''' This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        if filename.endswith('.jpg'):

            break

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session'''
#The kernal does not have the memory to train this model. But here's what I had done anyways.



'''import cv2

import re

from PIL import Image

import matplotlib.pyplot as plt

from keras.preprocessing.image import img_to_array, image

from keras.applications.resnet50 import preprocess_input, decode_predictions, resnet50

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D

from keras.layers import Dropout, Flatten, Dense, Activation

from keras.models import Sequential 

from sklearn.model_selection import train_test_split

from keras import optimizers

from keras.applications.vgg16 import VGG16

from keras.applications.vgg19 import VGG19

from keras.models import Model

from keras.preprocessing.image import ImageDataGenerator

from keras import backend as K'''
#%matplotlib inline

#import matplotlib.pyplot as plt

import numpy as np  

import pandas as pd

import torch



from torch import nn

from torch import optim

#import torch.nn.functional as F

from torchvision import datasets, transforms, models
import json



with open('/kaggle/input/herbarium-2020-fgvc7/nybg2020/train/metadata.json', 'r', errors='ignore') as f:

    train_metadata = json.load(f)
#train_metadata.keys()
with open('/kaggle/input/herbarium-2020-fgvc7/nybg2020/test/metadata.json', 'r', errors='ignore') as f:

    test_metadata = json.load(f)
#test_metadata.keys()
sub_samp = pd.read_csv('/kaggle/input/herbarium-2020-fgvc7/sample_submission.csv')
sub_samp
test_dir = '/kaggle/input/herbarium-2020-fgvc7/nybg2020/test/'

train_dir = '/kaggle/input/herbarium-2020-fgvc7/nybg2020/train/'
train_ann = pd.DataFrame(train_metadata['annotations'])
#train_ann
#train_cat = pd.DataFrame(train_metadata['categories'])
#train_cat
#train_imgs = pd.DataFrame(train_metadata['images'])
#train_imgs
#train_metadata['info']
#test_metadata['info']
#train_metadata['licenses']
#test_metadata['licenses']
#train_metadata['regions']
test_imgs = pd.DataFrame(test_metadata['images'])
#test_imgs
#train_files_names = train_imgs['file_name'].values
#train_files_names
#test_file_names = test_imgs['file_name'].values
#test_file_names
X = train_ann['image_id'].values
#X
y = train_ann['category_id'].values
#y.size
train_torch_y = torch.tensor(y, dtype=torch.long ).view(35543, 29)
#train_torch_y[0].shape 
#train_ann['category_id'].nunique()
test_X = test_imgs['id'].values
#test_X
'''def convert_to_tensor(path_img):

    img = image.load_img(path_img, target_size = (224,224))

    img_arr = image.img_to_array(img)

    return np.expand_dims(img_arr, axis = 0)'''
'''def convert_all_tensor(paths_imgs):

    tensor_list = [convert_to_tensor(i) for i in paths_imgs]

    return np.vstack(tensor_list)'''
#train_tensors = convert_all_tensor(train_dir+train_files_names).astype('float64')/255 #took too much memory

#test_tensors = convert_all_tensor(train_dir+train_files_names).astype('float64')/255
#train_tensors.shape, test_tensors.shape
data_transforms_train = transforms.Compose([transforms.RandomRotation(30),

                                       transforms.RandomResizedCrop(224),

                                       transforms.RandomHorizontalFlip(),

                                       transforms.ToTensor(),

                                       transforms.Normalize([0.485, 0.456, 0.406], 

                                                            [0.229, 0.224, 0.225])])







image_datasets_train =  datasets.ImageFolder(train_dir, transform=data_transforms_train)







dataloaders_train = torch.utils.data.DataLoader(image_datasets_train, batch_size=29)





#next(iter(dataloaders_train))[0].shape
data_transforms_test = transforms.Compose([transforms.Resize(256),

                                      transforms.CenterCrop(224),

                                      transforms.ToTensor(),

                                      transforms.Normalize([0.485, 0.456, 0.406], 

                                                           [0.229, 0.224, 0.225])])



image_datasets_test = datasets.ImageFolder(test_dir, transform=data_transforms_test)



dataloaders_test = torch.utils.data.DataLoader(image_datasets_test, batch_size=44)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = models.vgg16_bn(pretrained=True)



for param in model.parameters():

    param.requires_grad = False
model.classifier = nn.Sequential(nn.Linear(25088, 12544),

                                 nn.ReLU(),

                                 nn.Dropout(0.2),

                                 nn.Linear(12544, 32093),

                                 nn.LogSoftmax(dim=1))


criterion = nn.NLLLoss()

optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)
#train_torch_y[0]
model.to(device)
epochs = 2

steps = 0

running_loss = 0

print_every = 100

for epoch in range(epochs):

    for inputs, labels in zip(dataloaders_train,train_torch_y):

        steps += 1

        inputs, labels = inputs[0].to(device), labels.to(device)

        

        optimizer.zero_grad()

        

        logps = model.forward(inputs)

        loss = criterion(logps, labels)

        loss.backward()

        optimizer.step()



        running_loss += loss.item()

        

        if steps % print_every == 0:

               

            print(f"Epoch {epoch+1}/{epochs}.. "

                  f"Train loss: {running_loss/print_every:.3f}.. ")

            running_loss = 0

        if steps % 2000 == 0:

            break #to get out of loop before memory is full
#torch.cuda.is_available()
model.eval()

top_c = []

top_probs = []

with torch.no_grad():

    for inputs, labels in dataloaders_test:

        inputs  = inputs.to(device) 

        logps = model.forward(inputs)

                           

        ps = torch.exp(logps)

        top_p, top_class = ps.topk(1, dim=1)

        top_c.append(top_class)

        top_probs.append(top_p)

model.train()




#print(top_c)
#print(top_probs)
sub = sub_samp.copy()
topc_list = [torch.Tensor.cpu(i).numpy().tolist() for i in top_c]
from functools import reduce

topc_list1 = reduce(lambda x,y: x+y, topc_list)

topc_list2 = reduce(lambda x,y: x+y, topc_list1)
#topc_list2
sub['Predicted'] = list(map(int,topc_list2))
sub.to_csv('submission.csv', index=False)
sub