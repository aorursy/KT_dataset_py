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
!pip install gdown

!gdown "https://drive.google.com/uc?id=1-EcjB6mHCWkYdeCpg9PDmEed_CHwIohJ"

import pickle

inception_model = pickle.load(open('inception_model.dt','rb'))
import torch

import torch.nn as nn

import torch.optim as optim

import numpy as np

import torchvision

from torchvision import datasets, transforms

import matplotlib.pyplot as plt

import time

import os

import copy

print("PyTorch Version: ",torch.__version__)

print("Torchvision Version: ",torchvision.__version__)



num_classes = 30

batch_size = 256

num_epochs = 20
class MyModel(torch.nn.Module):

    def __init__(self,num_classes ):

        super(MyModel, self).__init__()

        inception_model = pickle.load(open('inception_model.dt','rb'))

        for param in inception_model.parameters():

            param.requires_grad = False

        num_ftrs = inception_model.AuxLogits.fc.in_features

        inception_model.AuxLogits.fc = nn.Linear(num_ftrs, 512)

        num_ftrs = inception_model.fc.in_features

        inception_model.fc = nn.Linear(num_ftrs, 512) 

        

        self.model = inception_model

        self.linear = torch.nn.Linear(512 ,num_classes)



        

    def forward(self, x):   

        if self.model.training:

            y, y_aux= self.model(x)    

        else:

            y = self.model(x)

        y_pred = self.linear(y)

        y_out = torch.relu(y_pred)

        return y_out    

from torch.utils.data import Dataset, DataLoader

from skimage.transform import resize

from skimage.color import gray2rgb

import skimage





class FacialDataset(Dataset):

    def __init__(self, filename):

        train_df = pd.read_csv(filename)

        self.x = train_df['Image'].to_numpy()

        train_df.drop(['Image'], axis=1, inplace=True)

        self.y = train_df.to_numpy()



    def __len__(self):

        return len(self.x)

    

    def __getitem__(self, index):

        y = torch.Tensor(self.y[index])

        y = y.type(torch.cuda.FloatTensor)

        x_pixels = np.array(self.x[index].split(' '), dtype = float).reshape(96,96)

        x_resized = skimage.transform.resize(x_pixels,(299,299))

        x = torch.Tensor(gray2rgb(x_resized)).permute(2,0,1)/255     

        x = x.type(torch.cuda.FloatTensor)

        return x,y

train_file = "/kaggle/input/facial-keypoints-detection/training.zip"

train_dataset = FacialDataset(train_file)

train_dl = DataLoader(train_dataset, batch_size, shuffle = True)

# for x,y in train_dl:

#     print(x,y)

len(train_dl.dataset)

def train_model(model, dataloaders, criterion, optimizer, num_epochs=20):

    since = time.time()

    best_acc = 0.0

    loss_hist = []

    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch+1, num_epochs))

        print('-' * 10)

        model.train()

        model.to("cuda")

        running_loss = 0.0

        running_corrects = 0

        for inputs, labels in dataloaders:

            optimizer.zero_grad()

            outputs = model(inputs)

            label_mask = (labels != labels)

            if label_mask.any():

                labels[label_mask] = outputs[label_mask]

            loss = criterion(outputs, labels)

            #loss2 = criterion(aux_outputs, labels)

            #loss = loss1 + 0.4 * loss2



            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            loss_hist.append(loss)

            



        epoch_loss = running_loss / len(dataloaders.dataset)

        print('Epoch {} Loss: {:.4f}'.format(epoch+1, epoch_loss))

        torch.save(model.state_dict(),'/kaggle/working/weights.dt')

        torch.save(loss, '/kaggle/working/loss.dt')



    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return model, loss_hist
my_model = MyModel(num_classes)

print(my_model)
params_to_update = my_model.parameters()

print("Params to learn:")

params_to_update = []    

for name,param in my_model.named_parameters():

    if param.requires_grad == True:

        params_to_update.append(param)

        print("\t",name)



# Observe that all parameters are being optimized

optimizer_ft = optim.Adam(params_to_update, lr=0.003)

criterion = nn.MSELoss()    #for multilabel classification

# Train and evaluate

model_ft, hist = train_model(my_model, train_dl, criterion, optimizer_ft, num_epochs=20)
%matplotlib inline

plt.figure(figsize = (15,10))

no_imgs = 5

no_cols = 5

no_rows = no_imgs//no_cols +1 

for i in range(no_imgs):

    x = train_dataset.x[i]

    x_pixels = np.array(x.split(' '), dtype = float).reshape(96,96)

    y_orig = train_dataset.y[i]

    y_orig = y_orig.reshape(-1,2)

    x_tensor, _ = train_dataset.__getitem__(i)

    model_ft.eval()

    y_pred = model_ft(x_tensor.unsqueeze(0))

    y_pred = y_pred.cpu().detach().numpy().reshape(-1,2)

    plt.subplot(no_rows, no_cols, i+1)

    plt.imshow(x_pixels)

    plt.title(f"Image {i+1}")

    plt.axis("off")

    

    plt.tight_layout()

    plt.scatter(y_orig[:, 0], y_orig[:, 1], marker = 'v', c = 'g')

    plt.scatter(y_pred[:, 0], y_pred[:, 1], marker = '.', c = 'r')

id_table = pd.read_csv('/kaggle/input/facial-keypoints-detection/IdLookupTable.csv')

test_data = pd.read_csv('/kaggle/input/facial-keypoints-detection/test.zip')



feature_map = {'left_eye_center_x': 0, 'left_eye_center_y': 1, 'right_eye_center_x' : 2,

       'right_eye_center_y' : 3, 'left_eye_inner_corner_x' : 4,

       'left_eye_inner_corner_y' : 5, 'left_eye_outer_corner_x' : 6,

       'left_eye_outer_corner_y' : 7, 'right_eye_inner_corner_x' : 8,

       'right_eye_inner_corner_y' : 9, 'right_eye_outer_corner_x' : 10,

       'right_eye_outer_corner_y' : 11, 'left_eyebrow_inner_end_x' : 12,

       'left_eyebrow_inner_end_y' : 13, 'left_eyebrow_outer_end_x' : 14,

       'left_eyebrow_outer_end_y' : 15, 'right_eyebrow_inner_end_x' : 16,

       'right_eyebrow_inner_end_y' : 17, 'right_eyebrow_outer_end_x' : 18,

       'right_eyebrow_outer_end_y' : 19, 'nose_tip_x' : 20, 'nose_tip_y' : 21,

       'mouth_left_corner_x' : 22, 'mouth_left_corner_y' : 23,

       'mouth_right_corner_x' : 24, 'mouth_right_corner_y' : 25,

       'mouth_center_top_lip_x' : 26, 'mouth_center_top_lip_y' : 27,

       'mouth_center_bottom_lip_x' : 28, 'mouth_center_bottom_lip_y' : 29

       }



model_ft.eval()



#img_data = test_data['Image'].to_numpy()

old_img_id = -1

for i in range(len(id_table)):

    new_img_id = id_table.loc[i,'ImageId']

    if (i == 0) or (new_img_id != old_img_id) :

        x = test_data.loc[new_img_id-1,'Image']

        img = np.array(x.split(' '), dtype = float).reshape(96,96)

        img = torch.Tensor(gray2rgb(skimage.transform.resize(img,(299,299)))).permute(2,0,1)/255

        img = img.type(torch.cuda.FloatTensor)

        features = model_ft(img.unsqueeze(0))

        features = features.cpu().detach().numpy().reshape(-1)

        old_img_id = new_img_id

    feature_location = features[feature_map[id_table.loc[i,'FeatureName']]]

    if feature_location > 96:

        feature_location = 96

    id_table.loc[i,'Location'] = feature_location 



id_table.drop(['ImageId'], axis=1, inplace=True)

id_table.drop(['FeatureName'], axis=1, inplace=True)

id_table.to_csv('/kaggle/working/submission.csv', index = False )

x = pd.read_csv('/kaggle/working/submission.csv')

x.head()