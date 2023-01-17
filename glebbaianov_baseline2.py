import os

import cv2

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import torch, torchvision



from PIL import Image

from IPython.display import clear_output

from torchvision import transforms as T

from tqdm.notebook import tqdm

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score, accuracy_score

np.random.seed(0)

torch.manual_seed(0)

torch.backends.cudnn.deterministic = True

torch.backends.cudnn.benchmark = False





PATH = '/kaggle/input/avito-auto-moderation/'

TRAIN_FILE = 'train_v2.csv'

SUB_FILE = 'sample_submission_v2.csv'

BATCH_SIZE = 32

LR1 = 0.001

LR2 = 0.0001



train = pd.read_csv(os.path.join(PATH, TRAIN_FILE))

submission = pd.read_csv(os.path.join(PATH, SUB_FILE))
train.head()
train.label.mean()
image_file = train.image[0]

img = plt.imread(os.path.join(PATH, image_file))

plt.imshow(img)
image_file = train.image[1]

img = plt.imread(os.path.join(PATH, image_file))

plt.imshow(img)
MEAN = np.array([0.485, 0.456, 0.406])

STD = np.array([0.229, 0.224, 0.225])

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")





train_transforms = T.Compose([

    T.ToPILImage(),

    T.Resize(size=224),

    T.CenterCrop(size=(224, 224)),

    

    # место для того чтобы добавить аугментации

    

    T.ToTensor(),     

    T.Normalize(MEAN, STD)

])



val_transforms = T.Compose([

    T.ToPILImage(),

    T.Resize(size=224),

    T.CenterCrop(size=(224, 224)),

    

    # нужно ли добавлять аугментации на валидации?

    

    T.ToTensor(),

    T.Normalize(MEAN, STD)

])



img = plt.imread(os.path.join(PATH, image_file))

img_transformed = train_transforms(img)
plt.imshow(img_transformed.permute(1, 2, 0).numpy() * STD + MEAN)
class ImageDataset(torch.utils.data.Dataset):

    

    def __init__(self, images, labels, path, mode='fit'):

        

        super(ImageDataset).__init__()

        self.images = images

        self.labels = labels

        self.path = path

        self.mode = mode

        

    def __len__(self):

        return len(self.images)

    

    def __getitem__(self, index):

        

        if self.mode == 'fit':

            x = self.read_image(file=self.images[index])

            y = torch.tensor([self.labels[index]])

            return x, y

        

        else:

            x = self.read_image(file=self.images[index])

            return x

        

    def preprocess_image(self, img):

        

        if self.mode == 'fit':

            img = train_transforms(img)

        else:

            img = val_transforms(img)

        

        return img    

    

    def read_image(self, file):

        img = plt.imread(os.path.join(self.path, file))

        img = self.preprocess_image(img)

        return img
train_images, val_images, train_labels, val_labels = train_test_split(train.image.values, train.label.values, random_state=1)



train_dataset = ImageDataset(train_images, train_labels, path=PATH)

val_dataset = ImageDataset(val_images, val_labels, path=PATH)

test_dataset = ImageDataset(images=submission.image, labels=None, path=PATH, mode='predict')



train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE)

test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
model = torchvision.models.resnet18(pretrained=True)

model.fc = torch.nn.Sequential(torch.nn.Linear(512, 1))

model = model.to(DEVICE)

model.eval()



for param in model.parameters():

    param.requires_grad = False



for param in model.layer2.parameters():

    param.requires_grad = False



for param in model.layer3.parameters():

    param.requires_grad = False



for param in model.layer4.parameters():

    param.requires_grad = False

    

for param in model.fc.parameters():

    param.requires_grad = True


def train_model(train_dataloader, val_dataloader, model, n_epochs=5, lr=0.001):

    

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    criterion = torch.nn.BCEWithLogitsLoss()



    losses = []

    val_losses = []



    for epoch in tqdm(range(n_epochs)):

        

        # eval

        model.eval()

        val_epoch_loss = estimate_val_loss(val_dataloader, model, criterion)

        val_losses.append(val_epoch_loss)

        

        # train

        model.train()

        for inputs, labels in train_dataloader:

            inputs = inputs.to(DEVICE)

            labels = labels.to(DEVICE)

            optimizer.zero_grad()



            outputs = model(inputs)

            preds = outputs.round()

            loss = criterion(outputs, labels.float())



            loss.backward()

            optimizer.step()



            curr_loss = loss.item() * inputs.size(0)

            losses.append(curr_loss)

            

            plot_progress(losses, val_losses, epoch, len(train_dataloader))

    

    return model





def estimate_val_loss(dataloader, model, criterion):

    

    val_epoch_loss = 0

    

    for inputs, labels in dataloader:

        with torch.no_grad():

            inputs = inputs.to(DEVICE)

            labels = labels.to(DEVICE)



            outputs = model(inputs)

            preds = outputs.round()

            loss = criterion(outputs, labels.float())



            val_epoch_loss += loss.item() * inputs.size(0)

    return val_epoch_loss / len(dataloader)





def get_model_predict(dataloader, model):

    

    preds = []

    

    mode = dataloader.dataset.mode

    dataloader.dataset.mode = 'predict'

    

    model.eval()

    

    for inputs in tqdm(dataloader):

        

        with torch.no_grad():

            

            inputs = inputs.to(DEVICE)



            outputs = model(inputs)

            batch_preds = torch.sigmoid(outputs.round())

            preds.extend(list(batch_preds.cpu().detach().numpy()))

            

    dataloader.dataset.mode = mode

    preds = np.vstack(preds)

    return preds





def plot_progress(train_losses, val_losses, epoch, train_dataloader_len):

    clear_output(True)

    plt.figure(figsize=(12, 8))

    plt.plot(np.arange(len(train_losses)), train_losses, label='train_loss')

    plt.plot([i*train_dataloader_len for i in range(epoch+1)], val_losses, label='val_loss')

    plt.legend()

    plt.ylabel('Loss')

    plt.xlabel('Batch number')

    plt.show()

model = train_model(train_dataloader, val_dataloader, model, lr=LR1, n_epochs=5)
val_preds = get_model_predict(val_dataloader, model)

roc_auc_score(val_labels, val_preds)
accuracy_score(val_labels, val_preds > 0.5)
for param in model.parameters():

    param.requires_grad = False



for param in model.layer2.parameters():

    param.requires_grad = True



for param in model.layer3.parameters():

    param.requires_grad = True



for param in model.layer4.parameters():

    param.requires_grad = True

    

for param in model.fc.parameters():

    param.requires_grad = True

model = train_model(train_dataloader, val_dataloader, model, n_epochs=2, lr=LR2)
val_preds = get_model_predict(val_dataloader, model)

roc_auc_score(val_labels, val_preds)
accuracy_score(val_labels, val_preds>0.5)
test_preds = get_model_predict(test_dataloader, model)
submission.score = test_preds

submission.to_csv('submission.csv', index=False)

submission.head()