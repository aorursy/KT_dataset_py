import os
import numpy as np
import imageio
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm_notebook as tqdm
import torch.optim as optim
from IPython.display import Image

import warnings
warnings.filterwarnings('ignore')
Image(filename = '/kaggle/input/imagesforkernel/Autoencoder_fig.png')
Image(filename='/kaggle/input/imagesforkernel/autoenc.png')
IMG_DIR = '/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba/'
plt.figure(figsize=(15,10))
for i in range(6):
    plt.subplot(2,3,i+1)
    choose_img = np.random.choice(os.listdir(IMG_DIR))
    image_path = os.path.join(IMG_DIR,choose_img)
    image = imageio.imread(image_path)
    plt.imshow(image)
class Autoencoders(nn.Module):
    def __init__(self):
        super().__init__()
        ### encoder
        self.conv1 = nn.Conv2d(3,64,5)
        self.maxpool = nn.MaxPool2d(2,return_indices=True)
        self.conv2 = nn.Conv2d(64,64,5)
        self.conv3 = nn.Conv2d(64,128,5)
        ### decoder
        self.deconv1 = nn.ConvTranspose2d(128,64,5)
        self.unpool = nn.MaxUnpool2d(2)
        self.deconv2 = nn.ConvTranspose2d(64,64,5)
        self.deconv3 = nn.ConvTranspose2d(64,3,5)
    
    def forward(self,x):
        x = self.conv1(x)
        x,ind1 = self.maxpool(x)
        x = self.conv2(x)
        x,ind2 = self.maxpool(x)
        x = self.conv3(x)
        
        x = self.deconv1(x)
        x = self.unpool(x,ind2)
        x = self.deconv2(x)
        x = self.unpool(x,ind1)
        x = self.deconv3(x)
        return x
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_transform = torchvision.transforms.Compose([
    transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize
])

train_dataloader = torch.utils.data.DataLoader(dataset=torchvision.datasets.ImageFolder('/kaggle/input/celeba-dataset/img_align_celeba/',
                                                                                       transform=train_transform),
                                              shuffle=True,batch_size=32,num_workers=0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device
model = Autoencoders().to(device)

criterian = nn.MSELoss()
optimizer = optim.Adam(model.parameters())
n_epochs = 2 ### increase number of epochs for better result
for epoch in tqdm(range(n_epochs)):
    model.train()
    iteration = 0
    for data,_ in tqdm(train_dataloader):
        optimizer.zero_grad()
        data = data.to(device)
        output = model.forward(data)
        loss = criterian(output,data)
        loss.backward()
        optimizer.step()
        if iteration%1000 == 0:
            print(f'iteration: {iteration} , loss : {loss.item()}')
    print(f'epoch: {epoch} loss: {loss.item()}')
torch.save(model.state_dict(),'autoencoder.h5')
model1 = Autoencoders()
model1.load_state_dict(torch.load('autoencoder.h5'))
model1.eval()
for data,_ in train_dataloader:
    break
pred_img = model1(data)
pred_img = pred_img.detach().numpy()
pred_img = pred_img.reshape(32,224,224,3)
plt.imshow(pred_img[0])
plt.show()
new_data = data.reshape(32,224,224,3)

plt.imshow(new_data[0])
plt.show()