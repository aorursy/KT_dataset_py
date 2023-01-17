# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
gray_path = "/kaggle/input/image-colorization/l/gray_scale.npy"
gray = np.load(gray_path)[:10000]

print(gray.shape)
import matplotlib.pyplot as plt

import cv2



def display(img):

    plt.set_cmap('gray')

    plt.imshow(img)

    plt.show()

sample = gray[100]

display(sample)
gray = gray / 255
gray = gray.reshape((-1,224,224,1))
import torch

from torch import nn
gray = gray.reshape((-1,1,224,224))
class autoencoder(nn.Module):

    def __init__(self):

        super(autoencoder, self).__init__()

        self.encoder = nn.Sequential(

            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10

            nn.ReLU(True),

            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5

            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3

            nn.ReLU(True),

            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2

        )

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5

            nn.ReLU(True),

            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15

            nn.ReLU(True),

            nn.ConvTranspose2d(8, 1, 8, stride=2, padding=2),  # b, 1, 28, 28

            #nn.ConvTranspose2d()

            nn.Sigmoid()

        )



    def forward(self, x):

        x = self.encoder(x)

        x = self.decoder(x)

        return x



model = autoencoder()

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,weight_decay=1e-5)
num_epochs = 700

size = gray.shape[0]

batches = 128

print_every = 100

for epoch in range(num_epochs):

    total_loss = 0

    steps = size//batches

    for i in range(0,steps,batches):

        data = gray[i:i+batches]

        tensor_data = (torch.from_numpy(data)).float()

        generated_data = model(tensor_data)

        loss = criterion(generated_data,tensor_data)

        

        total_loss += loss.item()*data.shape[0]

        

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()       

    print("Epoch:",epoch,"/",num_epochs," Loss:",total_loss/steps)

        
model.eval()

def get_image_from_tensor(tval):

    np_img = tval.detach().numpy()

    np_img = np_img.reshape(224,224,1)

    np_img = np_img[:,:,0]

    return np_img



def show_img(grayi,tval):

    grayi = grayi.reshape(224,224)

    display(grayi)

    img = get_image_from_tensor(tval)

    img = img*255

    display(img)

    

sample_g = gray[0]

tval = torch.from_numpy(sample_g).float().unsqueeze(dim=0)

enc_result = model.encoder(tval)

dec_result = model.decoder(enc_result).squeeze()



show_img(sample_g,dec_result)



sample_g = gray[500]

tval = torch.from_numpy(sample_g).float().unsqueeze(dim=0)

enc_result = model.encoder(tval)

dec_result = model.decoder(enc_result).squeeze()



show_img(sample_g,dec_result)





sample_g = gray[600]

tval = torch.from_numpy(sample_g).float().unsqueeze(dim=0)

enc_result = model.encoder(tval)

dec_result = model.decoder(enc_result).squeeze()



show_img(sample_g,dec_result)