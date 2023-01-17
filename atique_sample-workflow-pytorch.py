import os
import codecs
import PIL.Image as Image
import torchvision

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim

import matplotlib.pyplot as plt
import numpy as np

from tqdm.notebook import tqdm
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
!tar -xf /kaggle/input/qaida-dataset/train_200.tar.xz
!tar -xf /kaggle/input/qaida-dataset/test_200.tar.xz
class QaidaDataset(Dataset):
    def __init__(self, data_dir, transform=None, max_classes=0):

        if transform is None:
            self.transform = torchvision.transforms.ToTensor()
        else:
            self.transform = transform

        self.image_paths, self.labels = self.init(data_dir, max_classes)
        self.__len = len(self.labels)

    def init(self, data_dir, max_classes):
        """
        Prepare sorted hdf file readers
        """

        list_cls_dirs = sorted(os.listdir(data_dir), key=int)

        if max_classes == 0:
            print("Selecting max classes to : {}".format(len(list_cls_dirs)))

        elif max_classes > 0 and max_classes < len(list_cls_dirs):
            list_cls_dirs = list_cls_dirs[:max_classes]
            print("Selecting max classes to : {}".format(max_classes))

        else:
            print("Invalid arg max_classes: {}, Selecting max classes to : {}".format(max_classes, len(list_cls_dirs)))

        labels = []
        image_paths = []
        for cls_dir in list_cls_dirs:
            imgs_in_cls = [os.path.join(data_dir, cls_dir, img) for img in
                           os.listdir(os.path.join(data_dir, cls_dir))]

            num_imgs_in_cls = len(imgs_in_cls)
            cls_id = int(cls_dir)

            labels.extend([cls_id] * num_imgs_in_cls)
            image_paths.extend(imgs_in_cls)

        assert len(image_paths) == len(labels), "Size of images do not match size of labels"

        return image_paths, labels

    def __len__(self):
        return self.__len

    def __getitem__(self, idx):

        lbl = int(self.labels[idx])

        img_path = self.image_paths[idx]
        img = Image.open(img_path)
        img = img.convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img.double(), lbl
class QRN18(nn.Module):
    def __init__(self, target_classes):
        super(QRN18, self).__init__()
        self._model = torchvision.models.resnet18(pretrained=True)

        # Freeze all feature extraction layers
        for param in self._model.parameters():
            param.requires_grad = False

        # Replace the prediction head
        fc = nn.Linear(512, target_classes)
        fc.requires_grad = True
        
        self._model.fc = fc
        self._model.fc.requires_grad = True

    def forward(self, images):
        return self._model(images)
dataset = QaidaDataset("./train_200/")

# load ligatures
with codecs.open('/kaggle/input/qaida-dataset/ligatures_map_200', encoding='UTF-16LE') as ligature_file:
    ligatures_map = ligature_file.readlines()

# display an image grid with random samples
fig, ax = plt.subplots(5,5, figsize = (10,10))
for row in range(5):
    for col in range(5):
        idx = np.random.randint(len(dataset))
        img, lbl = dataset[idx]
        ax[row][col].imshow(np.transpose(img, (1,2,0)))
        ax[row][col].set_title(lbl)
        ax[row][col].axis("off")
        print ("Class: {} -> {}".format(lbl, ligatures_map[lbl]))

plt.show()
# create mini dataset instance 
small_dataset = QaidaDataset("./train_200/", max_classes= 10)

# create dataloader
small_dataloader = DataLoader(small_dataset, batch_size = 256, shuffle=True, num_workers=4)

# train on GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# create model_instance
QRN18_model = QRN18(target_classes=10)
QRN18_model = QRN18_model.double()
QRN18_model.to(device)

# create optimizer
trainable_parameters = [param for param in QRN18_model.parameters() if param.requires_grad]
optimizer = optim.Adam(QRN18_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


# Training loop
epochs = 20 # total epochs


train_loss =[]

for e in range(epochs):
    running_loss = 0
    for imgs, lbls in tqdm(small_dataloader):
        imgs, lbls = imgs.to(device), lbls.to(device)

        optimizer.zero_grad()
        img = QRN18_model(imgs)
        
        loss = criterion(img, lbls)
        running_loss += loss
        loss.backward()
        optimizer.step()
    running_loss /= len(small_dataloader)
    print("Epoch : {}/{}..".format(e+1, epochs),
      "Training Loss: {:.6f}".format(running_loss)) 
    train_loss.append(running_loss)
plt.plot(train_loss)
plt.title("Training loss curve")
plt.xlabel("Training epochs")
plt.ylabel("Cross entropy loss")
plt.show()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

QRN18_model.to(device)
QRN18_model.eval()

running_acc =0
for imgs, lbls in tqdm(small_dataloader):
    imgs = imgs.to(device)
    res = QRN18_model(imgs)
    pred = torch.argmax(res.to('cpu'), axis =1)
    acc = float(sum(pred == lbls))/lbls.shape[0]
    
    running_acc += acc

total_acc = round(running_acc/len(small_dataloader), 3)

print ("Total Acc: {}".format(total_acc))