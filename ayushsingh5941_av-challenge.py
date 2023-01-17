import pandas as pd

from PIL import Image

import torch

from torch import nn, optim

from torchvision.models import googlenet

import torchvision.transforms as T

from torch.utils.data import DataLoader, Dataset

import numpy as np

import os

from skimage import io

import time
train_df = pd.read_csv('../input/av-emergency/train_SOaYf6m/train.csv')
test_df = pd.read_csv('../input/av-emergency/test_vc2kHdQ.csv')
train_df.head()
train_df.info()
test_df.info()
train_df['emergency_or_not'].value_counts()
model = googlenet(pretrained=True, transform_input=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.to(device)
model.fc
model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 100),

                         nn.ReLU(),

                         nn.Dropout(0.8),

                         nn.Linear(100, 1),

                        nn.Sigmoid())
model = model.to(device)

print(next(model.parameters()).is_cuda)
class LoadCustomData(Dataset):

    def __init__(self, csv_file, root_path, transform=None):

        self.annotation = pd.read_csv(csv_file)

        self.root_path = root_path

        self.transform = transform

    

    def __len__(self):

        return len(self.annotation)

    

    def __getitem__(self, index):

        img_path = os.path.join(self.root_path, self.annotation.iloc[index, 0])

        image = Image.open(img_path)

        y_label = torch.tensor(int(self.annotation.iloc[index, 1]))

        

        if self.transform:

            image = self.transform(image)

        

        return (image.float(), y_label.float())

    

    
# Transforms

transformations = T.Compose([T.Resize(225),

                             T.RandomHorizontalFlip(0.2),

                             T.RandomPerspective(),

                             T.RandomRotation(12),

                             T.ToTensor(),

                             T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]

                                     )
# Loading train data

train_data = LoadCustomData(csv_file='../input/av-emergency/train_SOaYf6m/train.csv', root_path='../input/av-emergency/train_SOaYf6m/images', transform=transformations)
train_set, valid_set = torch.utils.data.random_split(train_data, [1500, 146])
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=64, shuffle=True)
def check_accuracy(loader, model):

    num_correct = 0

    num_samples = 0

    model.eval()

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)

            y = y.to(device)

        

            scores = model(x)

            _, predictions = scores.max(1)

            num_correct += (predictions ==y).sum()

            num_samples += predictions.size(0)

        print(f'Got {num_correct/num_samples} with accuracy {float(num_correct)/float(num_samples) *100}')
optimizer = optim.Adam(model.parameters(), lr=0.00001)

criterion = nn.BCELoss()
epochs = 10

valid_loss_min = np.Inf # track change in validation loss





for epoch in range(epochs):

    train_loss = 0.0

    valid_loss = 0.0

    

    print('No. of epochs : ', epoch+1)

    

    # Training the model

    for batch_id, (images, label) in enumerate(train_loader):

        # zero grad 

        optimizer.zero_grad()

        # prediction of model

        images = images.to(device)

        label = label.to(device)

        

        output = model(images)

        # loss 

        loss = criterion(output, label.view(-1,1))

        # Backprop

        loss.backward()

        # updating weights

        optimizer.step()

        # Calculation training loss

        train_loss += loss.item()

    

    # Validation model

    for batch_id, (images, label) in enumerate(valid_loader):

        model.eval()

        # prediction of modelediction of model

        images = images.to(device)

        label = label.to(device)

            

            

        output = model(images)

        # Calculating loss

        valid_loss_model = criterion(output, label.view(-1,1))

        valid_loss += valid_loss_model.item()

        

    # Print training stats

    train_loss = train_loss /len(train_loader)

    valid_loss = valid_loss / len(valid_loader)

    

    print('Train loss :', train_loss)

    print('Valid loss :', valid_loss)



    

    # Save model if validation loss is less then train loss

    

    if valid_loss <= valid_loss_min:

        print('\nValidation loss decreases  (Saving model): ', valid_loss, 'Valid loss :', valid_loss_min)

        torch.save(model.state_dict(), 'model.pt')

        valid_loss_min = valid_loss

        print('valid_loss_min update :', valid_loss_min)

        print('------------------------------------------------')
    check_accuracy(valid_loader, model)