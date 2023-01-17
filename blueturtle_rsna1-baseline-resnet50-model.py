# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
!pip install pydicom

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
#for dirname, _, filenames in os.walk('/kaggle/input'):
    #for filename in filenames:
        #print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Load the csv files
train_csv_path = "/kaggle/input/rsna-str-pulmonary-embolism-detection/train.csv"
test_csv_path = "/kaggle/input/rsna-str-pulmonary-embolism-detection/test.csv"
#Create a dataframe object from csv file
train_csv = pd.read_csv(train_csv_path)
test_csv = pd.read_csv(test_csv_path)
train_csv["image_path"] = train_csv["StudyInstanceUID"] + "/" + train_csv["SeriesInstanceUID"] + "/" + train_csv["SOPInstanceUID"]
train_csv
#Initially I will load a subset of images to test the model as it will be quicker.
    #Study Instance ID: 6897fa9de148
    #Series Instance ID: 2bfbb7fd2e8b
    
StudyInstance = "6897fa9de148"
SeriesInstance = "2bfbb7fd2e8b"

Image_path = "/kaggle/input/rsna-str-pulmonary-embolism-detection/train/"

#List the file path for each image in one directory.
#for image in os.listdir(Image_path):
 #   print(os.path.join(Image_path + image))
#Display one image from the directory to check code is working properly.
import matplotlib.pyplot as plt #To view the .dcm files
import pydicom #To read the .dcm (dicom) data

#filename = "/kaggle/input/rsna-str-pulmonary-embolism-detection/train/6897fa9de148/2bfbb7fd2e8b/894706f0aa3e.dcm"
#slide = pydicom.dcmread(filename)
#plt.imshow(slide.pixel_array, cmap=plt.cm.bone)
import torch

#Set device to cuda/GPU as the GPU is considerably faster than the GPU at image tasks.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.utils.data import DataLoader, Dataset    #Create an efficient dataloader set to feed images to the model
from torch.utils.data.sampler import SequentialSampler
from torchvision import transforms

import albumentations as A #Package of transformations
from albumentations.pytorch.transforms import ToTensorV2

def get_train_transforms():
    return A.Compose([
        ToTensorV2(p=1.0) #Convert image and target to a tensor. Our CNN will only work with Tensors, not numpy arrays.
    ])

def get_valid_transforms():
    return A.Compose([
            A.Resize(height=1024, width=1024, p=1.0),
            ToTensorV2(p=1.0), #Convert image and target to a tensor. Our CNN will only work with Tensors, not numpy arrays.
        ], p=1.0)

class TrainData(Dataset):

    def __init__(self, dataframe, image_dir, transforms):
        super().__init__()
        
        self.df = dataframe
        self.image_ids = dataframe['image_path'].unique()
        self.image_dir = image_dir
        self.transforms = transforms

    def __getitem__(self, idx: int):
        #Generates one sample of the data
        image_id = self.image_ids[idx]
        image = pydicom.dcmread(f'{self.image_dir}{image_id}.dcm') #Read .dcm file into the loop
        image = image.reshape((512,512,1)).astype('float')

        labels =self.df[['negative_exam_for_pe', 'rv_lv_ratio_gte_1', 'rv_lv_ratio_lt_1',
                     'leftsided_pe', 'chronic_pe', 'rightsided_pe',
                     'acute_and_chronic_pe', 'central_pe', 'indeterminate']].loc[idx].values
        
        if self.transforms:
            image = {"image" : image,} #Create a dictionary of the image values. We can only force the kwargs through transform if they are in dict type
            image = self.transforms(**image)
            
        return image, labels #Return the transformed image, and the associated labels.
    
    def __len__(self) -> int:
        #The loader needs to know how many items we have in our dataset. Therefore we use the __len__ function as an upper-bound.
        return len(self.image_ids)
from sklearn.model_selection import train_test_split

#Split the training data into train and validate sets. We do this so we can assess the accuracy of our model.
train_meta, valid_meta = train_test_split(train_csv,test_size=0.2)
#print(train_meta.shape)
#print(valid_meta.shape)
train_dataset = TrainData(train_meta, Image_path, transforms = get_train_transforms())
valid_dataset = TrainData(valid_meta, Image_path, transforms = get_valid_transforms())
"""
DataLoader takes in the arguements:

    batch_size, which denotes the number of samples contained in each generated batch.
    shuffle. If set to True, we will get a new order of exploration at each pass.
        Shuffling the order in which examples are fed to the classifier is helpful so that batches between epochs do not look alike.
        Doing so will eventually make our model more robust.
    num_workers, which denotes the number of processes that generate batches in parallel.
        A high enough number of workers assures that CPU computations are efficiently managed, i.e. that the bottleneck is indeed the neural network's forward and backward operations on the GPU (and not data generation).

"""

train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True, num_workers = 0)
valid_loader = DataLoader(valid_dataset, batch_size = 32, shuffle = True, num_workers = 0)

"""
What is an AxesImage:
An image attached to axes. Rather than just a normal image without. Often used in medical situations.
"""
from torch import nn
from torch.nn import functional as F
import torchvision.models as models
from torch import optim
from torchvision import datasets, models

model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
    
model.fc = nn.Sequential(nn.Linear(2048, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, 10), #Compact the 512 incoming nodes into 10 output nodes
                                 nn.LogSoftmax(dim=1)) #Use softmax to classify the output into 1 of these 10 output nodes.
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
model.to(device)
#Install and import dependencies
#!conda install -c conda-forge gdcm -y
#import gdcm
epochs = 1
steps = 0
running_loss = 0
print_every = 10
train_losses, test_losses = [], []
for epoch in range(epochs):
    for image, label in train_loader:
        steps += 1
        image, label = image.to(device), label.to(device) #Send image and label to the device, cuda if enabled, otherwise gpu
        optimizer.zero_grad()                             #zero out the gradient
        logps = model.forward(image)                      #Do a forward pass of the model using the current batch of images and targets
        loss = criterion(logps, label)                    #Calculate the loss
        loss.backward()                                   #Backpropegate the loss
        optimizer.step()                                  #Take a step in the direction of the optimum, calculated by the optimiser - in this case Adam.
        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for images, category in valid_loader:
                    images, category = images.to(device), category.to(device)
                    logps = model.forward(images)
                    batch_loss = criterion(logps, category)
                    test_loss += batch_loss.item()
                    
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == category.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            train_losses.append(running_loss/len(train_loader))
            test_losses.append(test_loss/len(valid_loader))                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(valid_loader):.3f}.. "
                  f"Test accuracy: {accuracy/len(valid_loader):.3f}")
            running_loss = 0
            model.train()
torch.save(model, 'RSNAResNet50.pth')