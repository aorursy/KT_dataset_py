# import packages

import os

import torch 

import torchvision

import torch.nn as nn

import torchvision.transforms as transforms

import torch.optim as optim

import matplotlib.pyplot as plt

import torch.nn.functional as F

 

from torchvision import datasets

from torch.utils.data import DataLoader

from torchvision.utils import save_image

from tqdm.notebook import tqdm
# leanring parameters

epochs = 10

batch_size = 64

lr = 0.0001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# image transformations

transform = transforms.Compose([

    transforms.ToTensor(),

])
train_data = datasets.FashionMNIST(

    root='./data',

    train=True, 

    download=True,

    transform=transform

)

val_data = datasets.FashionMNIST(

    root='./data',

    train=False,

    download=True,

    transform=transform

)



train_loader = DataLoader(

    train_data, 

    batch_size=batch_size,

)

val_loader = DataLoader(

    val_data, 

    batch_size=batch_size, 

)
class Autoencoder(nn.Module):

    def __init__(self):

        super(Autoencoder, self).__init__()



        # encoder

        self.enc1 = nn.Linear(in_features=784, out_features=512)

        self.enc2 = nn.Linear(in_features=512, out_features=32)



        # decoder 

        self.dec1 = nn.Linear(in_features=32, out_features=512)

        self.dec2 = nn.Linear(in_features=512, out_features=784)



    def forward(self, x):

        # encoding

        x = F.relu(self.enc1(x))

        x = F.relu(self.enc2(x))

        

        # decoding

        x = F.relu(self.dec1(x))

        x = torch.sigmoid(self.dec2(x))

        return x



model = Autoencoder().to(device)

print(model)
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=lr)
def fit(model, dataloader):

    model.train()

    running_loss = 0.0

    for i, data in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):

        data, _ = data

        data = data.to(device)

        data = data.view(data.size(0), -1)

        optimizer.zero_grad()

        reconstruction= model(data)

        loss = criterion(reconstruction, data)

        loss.backward()

        running_loss += loss.item()

        optimizer.step()



    train_loss = running_loss/len(dataloader.dataset)

    return train_loss





def validate(model, dataloader):

    model.eval()

    running_loss = 0.0

    with torch.no_grad():

        for i, data in tqdm(enumerate(dataloader), total=int(len(val_data)/dataloader.batch_size)):

            data, _ = data

            data = data.to(device)

            data = data.view(data.size(0), -1)

            reconstruction = model(data)

            loss = criterion(reconstruction, data)

            running_loss += loss.item()

        

            # save the last batch input and output of every epoch

            if i == int(len(val_data)/dataloader.batch_size) - 1:

                num_rows = 8

                both = torch.cat((data.view(batch_size, 1, 28, 28)[:8], 

                                  reconstruction.view(batch_size, 1, 28, 28)[:8]))

                save_image(both.cpu(), f"output{epoch}.png", nrow=num_rows)

                output = plt.imread(f"output{epoch}.png")

                plt.imshow(output)

                plt.show()



    val_loss = running_loss/len(dataloader.dataset)

    return val_loss
train_loss = []

val_loss = []

for epoch in range(epochs):

    print(f"Epoch {epoch+1} of {epochs}")

    train_epoch_loss = fit(model, train_loader)

    val_epoch_loss = validate(model, val_loader)

    train_loss.append(train_epoch_loss)

    val_loss.append(val_epoch_loss)

    print(f"Train Loss: {train_epoch_loss:.6f}")

    print(f"Val Loss: {val_epoch_loss:.6f}")