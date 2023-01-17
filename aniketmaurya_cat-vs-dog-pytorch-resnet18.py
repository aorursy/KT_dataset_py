import numpy as np # linear algebra

import torch

from torch import nn, optim

from torchvision import datasets, models, transforms

from torch import functional as F



from torch.utils.data import Dataset, DataLoader



from glob import glob

from PIL import Image



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



!unzip -q /kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip

!unzip -q /kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip
class CatDataSet(Dataset):

    def __init__(self, data_dir, transforms):

        self.image_pathlist = glob(os.path.join(data_dir, '*.jpg'))

        self.num_images = len(self.image_pathlist)

        

        self.transforms = transforms

        

    def __len__(self):

        return self.num_images

    

    def get_class(self, path):

        path = path.split('/')[-1]

        class_ = path.split('.')[0]

        if class_== 'cat':

            return 0.0

        return 1.0



    

    def __getitem__(self,idx):

        path= self.image_pathlist[idx]

        image=Image.open(path)

        if self.transforms:

            image = self.transforms(image)

        

        return image, self.get_class(path)
data_transform = transforms.Compose([

        transforms.RandomSizedCrop(224),

        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),

        transforms.Normalize(mean=[0.485, 0.456, 0.406],

                             std=[0.229, 0.224, 0.225])

    ])
train_ds = CatDataSet('train', data_transform)

val_ds = CatDataSet('val', data_transform)



dataset_size = {'train': len(train_ds), 'val': len(val_ds)}
train_dl = DataLoader(train_ds, 8)

val_dl = DataLoader(val_ds, 8)
model = models.resnet18(pretrained=True)
in_feats = model.fc.in_features



model.fc = nn.Linear(in_feats, 1)
epochs = 5

lr = 1e-4
criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters(), lr)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



model = model.to(device)
for epoch in range(epochs):

    print(f"{epoch}/{epochs-1}")

    print("-"*10)

    

    running_loss = 0.0

    for i, (images, labels) in enumerate(train_dl):

        images = images.to(device)

        labels = labels.to(device)

        labels = labels.view(len(labels), 1)

        

        optimizer.zero_grad()

        

        logits = model(images)

        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)

        

        loss.backward()

        optimizer.step()

        

        if i%200==0:

            print(f'Loss at step {i} - {loss.item()}')



    epoch_loss = running_loss / dataset_size['train']

    print(f"Train loss: {epoch_loss}")

    

    

    running_loss = 0.0

    for images, labels in val_dl:

        images = images.to(device)

        labels = labels.to(device)

        labels = labels.view(len(labels), 1)

                

        logits = model(images)

        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)



    epoch_loss = running_loss / dataset_size['val']

    print(f"Val loss: {epoch_loss}")

    

    print()