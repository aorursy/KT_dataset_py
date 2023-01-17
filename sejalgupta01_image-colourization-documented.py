# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

 #   for filename in filenames:

  #      print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from torchvision.datasets import ImageFolder

import torchvision.transforms as T

import torch

import torch.nn as nn

from torch.utils.data import DataLoader

from torch.utils.data.dataset import random_split

import torch.nn.functional as F

from tqdm.notebook import tqdm

import matplotlib.pyplot as plt

from skimage.color import rgb2lab, lab2rgb, rgb2gray
DATA_DIR = '../input/imagenet/imagenet/'
dataset = ImageFolder(DATA_DIR, transform=T.Compose([T.Resize((256, 256)),T.ToTensor()]))
len(dataset)
random_seed = 42

torch.manual_seed(random_seed)



val_size = 1000

train_size = len(dataset) - val_size



train_ds, val_ds = random_split(dataset, [train_size, val_size])

len(train_ds), len(val_ds)
batch_size = 128
train_loader = DataLoader(train_ds, batch_size = batch_size, shuffle = True, num_workers = 4, pin_memory = True)

val_loader = DataLoader(val_ds, batch_size = batch_size, num_workers = 4, pin_memory = True)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

device
def to_device(data, device):

    if isinstance(data, (list, tuple)):

        return [to_device(x, device) for x in data]

    return data.to(device, non_blocking = True)
class DeviceDataLoader():

    def __init__(self, dl, device):

        self.dl = dl

        self.device = device



    def __iter__(self):

        for batch in self.dl:

            yield to_device(batch, self.device)

  

    def __len__(self):

        return len(self.dl)
train_loader = DeviceDataLoader(train_loader, device)

val_loader = DeviceDataLoader(val_loader, device)
def generate_l_ab(images): 

    lab = rgb2lab(images.permute(0, 2, 3, 1).cpu().numpy())

    X = lab[:,:,:,0]

    X = X.reshape(X.shape+(1,))

    Y = lab[:,:,:,1:] / 128

    return to_device(torch.tensor(X, dtype = torch.float).permute(0, 3, 1, 2), device),to_device(torch.tensor(Y, dtype = torch.float).permute(0, 3, 1, 2), device)
class BaseModel(nn.Module):

    def training_batch(self, batch):

        images, _ = batch

        X, Y = generate_l_ab(images)

        outputs = self.forward(X)

        loss = F.mse_loss(outputs, Y)

        return loss



    def validation_batch(self, batch):

        images, _ = batch

        X, Y = generate_l_ab(images)

        outputs = self.forward(X)

        loss = F.mse_loss(outputs, Y)

        return {'val_loss' : loss.item()}



    def validation_end_epoch(self, outputs):

        epoch_loss = sum([x['val_loss'] for x in outputs]) / len(outputs)

        return {'epoch_loss' : epoch_loss}
def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1, **_) -> int:

    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2

    return padding
class Encoder_Decoder(BaseModel):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Conv2d(1, 64, kernel_size = 3, stride = 2, padding = get_padding(3, 2)),

            nn.ReLU(),

            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, kernel_size = 3, padding = get_padding(3)),

            nn.ReLU(),

            nn.BatchNorm2d(128),

        

            nn.Conv2d(128, 128, kernel_size = 3, stride = 2, padding = get_padding(3, 2)),

            nn.ReLU(),

            nn.BatchNorm2d(128),

            nn.Conv2d(128, 256, kernel_size = 3, padding = get_padding(3)),

            nn.ReLU(),

            nn.BatchNorm2d(256),

        

            nn.Conv2d(256, 256, kernel_size = 3, stride = 2, padding = get_padding(3, 2)),

            nn.ReLU(),

            nn.BatchNorm2d(256),

            nn.Conv2d(256, 512, kernel_size = 3, padding = get_padding(3)),

            nn.ReLU(),

            nn.BatchNorm2d(512),

            

            nn.Conv2d(512, 512, kernel_size = 3, padding = get_padding(3)),

            nn.ReLU(),

            nn.BatchNorm2d(512),

            nn.Conv2d(512, 256, kernel_size = 3, padding = get_padding(3)),

            nn.ReLU(),

            nn.BatchNorm2d(256),

        

            nn.Conv2d(256, 128, kernel_size = 3, padding = get_padding(3)),

            nn.Upsample(size = (64,64)),

            nn.Conv2d(128, 64, kernel_size = 3, padding = get_padding(3)),

            nn.Upsample(size = (128,128)),

            nn.Conv2d(64, 32, kernel_size = 3, padding = get_padding(3)),

            nn.Conv2d(32, 16, kernel_size = 3, padding = get_padding(3)),

            nn.Conv2d(16, 2, kernel_size = 3, padding = get_padding(3)),

            nn.Tanh(),

            nn.Upsample(size = (256,256))

    )



    def forward(self, images):

        return self.network(images)     

    
model = Encoder_Decoder()

to_device(model, device)
@torch.no_grad()

def validate(model, val_loader):

    model.eval()

    outputs = [model.validation_batch(batch) for batch in val_loader]

    return model.validation_end_epoch(outputs)



def fit(model, epochs, learning_rate, train_loader, val_loader, optimization_func = torch.optim.SGD):

    torch.cuda.empty_cache()

    history = []

    optimizer = optimization_func(model.parameters(), learning_rate)

    for epoch in range(epochs):

        train_losses = []

        model.train()

        for batch in tqdm(train_loader):

            loss = model.training_batch(batch)

            train_losses.append(loss)

            loss.backward()

            optimizer.step()

            optimizer.zero_grad()

        result = validate(model, val_loader)

        result['train_loss'] = torch.stack(train_losses).mean().item()

        history.append(result)

        print('Epoch: {}, Train loss: {:.4f}, Validation loss: {:.4f}'.format(epoch, result['train_loss'], result['epoch_loss']))

    return history
history = [validate(model, val_loader)]

history
history += fit(model, 5, 0.001, train_loader, val_loader, torch.optim.Adam)
checkpoint = {'model': Encoder_Decoder(),

              'state_dict': model.state_dict()}
torch.save(checkpoint, 'test.pth')
def load_checkpoint(filepath):

    checkpoint = torch.load(filepath)

    model = checkpoint['model']

    model.load_state_dict(checkpoint['state_dict'])

    for parameter in model.parameters():

        parameter.requires_grad = True

    

    model.eval()

    

    return model
model = load_checkpoint('../input/testing/test21.pth')

print(model)

to_device(model, device)
def to_rgb(grayscale_input, ab_output, save_path=None, save_name=None):

    color_image = torch.cat((grayscale_input, ab_output), 0).numpy() # combine channels

    color_image = color_image.transpose((1, 2, 0))  # rescale for matplotlib

    color_image[:, :, 0:1] = color_image[:, :, 0:1]

    color_image[:, :, 1:3] = (color_image[:, :, 1:3]) * 128

    color_image = lab2rgb(color_image.astype(np.float64))

    grayscale_input = grayscale_input.squeeze().numpy()

    return color_image
def prediction(img):

    a = rgb2lab(img.permute(1, 2, 0))

    a = torch.tensor(a[:,:,0]).type(torch.FloatTensor)

    a = a.unsqueeze(0)

    a = a.unsqueeze(0)

    xb = to_device(a, device)

    ab_img = model(xb)

    xb = xb.squeeze(0)

    ab_img = ab_img.squeeze(0)

    return to_rgb(xb.detach().cpu(), ab_img.detach().cpu())
import glob

from PIL import Image



images = glob.glob("../input/test-images/*jpg")
for img in range(len(images)):

    image = Image.open(images[img])

    trans1 = T.Resize((256, 256))

    trans2 = T.ToTensor()

    images[img] = trans2(trans1(image))
img = images[1]

f, arr = plt.subplots(1, 2, sharey=True)

arr[0].imshow(img.permute(1, 2, 0))

arr[1].imshow(prediction(img))