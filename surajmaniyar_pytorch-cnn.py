# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

import random

from sklearn.model_selection import train_test_split
!ls ../input/training/
import torch

torch.cuda.is_available()
path = '../input/training/training.csv'

df = pd.read_csv(path)



df.shape
train = df.sample(frac=0.9, random_state=200)

val = df.drop(train.index)



print(train.shape)

print(val.shape)
mean_coordinates = np.array([[66.02653075, 37.81405367],

                             [29.97346925, 37.81405367],

                             [59.2533041 , 37.9673122 ],

                             [73.47301818, 37.87028774],

                             [36.7466959 , 37.9673122 ],

                             [22.52698182, 37.87028774],

                             [56.37318808, 29.41783836],

                             [79.80662065, 30.0823681 ],

                             [39.62681192, 29.41783836],

                             [16.19337935, 30.0823681 ],

                             [48.        , 62.7158836 ],

                             [63.1926452 , 76.07526412],

                             [32.8073548 , 76.07526412],

                             [48.        , 72.9194426 ],

                             [48.        , 78.97014625]])



def get_image(index, df, display=False, flip=False, augment=False):

    img_list = df.iloc[index]['Image']

    img_list = img_list.split(' ')

    img_flat = list(map(int, img_list))

    img = np.reshape(img_flat, (96, 96))

    

    coords = []

    

    

    x1, y1 = (df.iloc[index]['left_eye_center_x'], df.iloc[index]['left_eye_center_y'])

    if x1 != x1:

        x1, y1 = mean_coordinates[0]

        

    x2, y2 = (df.iloc[index]['right_eye_center_x'], df.iloc[index]['right_eye_center_y'])

    if x2 != x2:

        x2, y2 = mean_coordinates[1]

    

    x3, y3 = (df.iloc[index]['left_eye_inner_corner_x'], df.iloc[index]['left_eye_inner_corner_y'])

    if x3 != x3:

        x3, y3 = mean_coordinates[2]

    

    x4, y4 = (df.iloc[index]['left_eye_outer_corner_x'], df.iloc[index]['left_eye_outer_corner_y'])

    if x4 != x4:

        x4, y4 = mean_coordinates[3]

    

    x5, y5 = (df.iloc[index]['right_eye_inner_corner_x'], df.iloc[index]['right_eye_inner_corner_y'])

    if x5 != x5:

        x5, y5 = mean_coordinates[4]

    

    x6, y6 = (df.iloc[index]['right_eye_outer_corner_x'], df.iloc[index]['right_eye_outer_corner_y'])

    if x6 != x6:

        x6, y6 = mean_coordinates[5]

    

    x7, y7 = (df.iloc[index]['left_eyebrow_inner_end_x'], df.iloc[index]['left_eyebrow_inner_end_y'])

    if x7 != x7:

        x7, y7 = mean_coordinates[6]

    

    x8, y8 = (df.iloc[index]['left_eyebrow_outer_end_x'], df.iloc[index]['left_eyebrow_outer_end_y'])

    if x8 != x8:

        x8, y8 = mean_coordinates[7]

    

    x9, y9 = (df.iloc[index]['right_eyebrow_inner_end_x'], df.iloc[index]['right_eyebrow_inner_end_y'])

    if x9 != x9:

        x9, y9 = mean_coordinates[8]

    

    x10, y10 = (df.iloc[index]['right_eyebrow_outer_end_x'], df.iloc[index]['right_eyebrow_outer_end_y'])

    if x10 != x10:

        x10, y10 = mean_coordinates[9]

    

    x11, y11 = (df.iloc[index]['nose_tip_x'], df.iloc[index]['nose_tip_y'])

    if x11 != x11:

        x11, y11 = mean_coordinates[10]

    

    x12, y12 = (df.iloc[index]['mouth_left_corner_x'], df.iloc[index]['mouth_left_corner_y'])

    if x12 != x12:

        x12, y12 = mean_coordinates[11]

    

    x13, y13 = (df.iloc[index]['mouth_right_corner_x'], df.iloc[index]['mouth_right_corner_y'])

    if x13 != x13:

        x13, y13 = mean_coordinates[12]

    

    x14, y14 = (df.iloc[index]['mouth_center_top_lip_x'], df.iloc[index]['mouth_center_top_lip_y'])

    if x14 != x14:

        x14, y14 = mean_coordinates[13]

    

    x15, y15 = (df.iloc[index]['mouth_center_bottom_lip_x'], df.iloc[index]['mouth_center_bottom_lip_y'])

    if x15 != x15:

        x15, y15 = mean_coordinates[14]

    

    if augment:

        flip = random.choice([True, False])

    

    if flip:

        coords = coords + [[96-x2, y2], [96-x1, y1]]

        coords = coords + [[96-x5, y5], [96-x6, y6]]

        coords = coords + [[96-x3, y3], [96-x4, y4]]

        coords = coords + [[96-x9, y9], [96-x10, y10]]

        coords = coords + [[96-x7, y7], [96-x8, y8]]

        coords = coords + [[96-x11, y11]]

        coords = coords + [[96-x13, y13], [96-x12, y12]]

        coords = coords + [[96-x14, y14]]

        coords = coords + [[96-x15, y15]]

        img = np.fliplr(img)

        

    else:

        coords = coords + [[x1, y1], [x2, y2]]

        coords = coords + [[x3, y3], [x4, y4]]

        coords = coords + [[x5, y5], [x6, y6]]

        coords = coords + [[x7, y7], [x8, y8]]

        coords = coords + [[x9, y9], [x10, y10]]

        coords = coords + [[x11, y11]]

        coords = coords + [[x12, y12], [x13, y13]]

        coords = coords + [[x14, y14]]

        coords = coords + [[x15, y15]]

    

    

    if display:

        plt.imshow(img, 'gray')

        for (x, y) in coords:

            if (not np.isnan(x)) and (not np.isnan(y)): 

                mcircle = mpatches.Circle((x, y), 1, color='r', fill=True)

                plt.gca().add_patch(mcircle)

    

    coords = np.array(coords)

    #coords = np.nan_to_num(coords)

    

    return img, coords
batch_size = 128

epochs = 50

learning_rate = 1e-4

weight_decay = 1e-4

dropout = 0.3
import torch

import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

import torch.nn.functional as F

from torch.autograd import Variable
class ImageDataset(Dataset):

    def __init__(self, df):

        self.df = df

        self.transform = transforms.ToTensor()

        print(self.df.shape)

        

    def __len__(self):

        return self.df.shape[0]

    

    def __getitem__(self, index):

        image, coords = get_image(index, self.df, augment=True)

        image = image/255.0

        coords = coords/96.0

        image = self.transform(image)

        coords = torch.Tensor(coords.reshape(-1))

        

        return image.float(), coords.float()
class ImageModel(nn.Module):

    def __init__(self):

        super(ImageModel, self).__init__()

        self.build_model()

        

    def build_model(self):

                                                            

        self.conv1 = nn.Conv2d(1, 32, stride=2, kernel_size=4)        

        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)    

        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)       

        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=2)  

        

        self.linear1 = nn.Linear(45*45*32, 128)

        #self.drop1 = nn.Dropout(dropout)

        self.linear2 = nn.Linear(128, 30)

        

    def forward(self, x):

        out = self.conv1(x)

        out = F.relu(out)

        out = self.conv2(out)

        out = F.relu(out)

        

        out = out.view(out.size(0), -1)

        

        out = self.linear1(out)

        out = F.relu(out)

        #out = self.drop1(out)

        out = self.linear2(out)

        

        return out

        
model = ImageModel()

train_dataset = ImageDataset(df)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = ImageDataset(val)

val_loader = DataLoader(val_dataset, batch_size=batch_size)
criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
if torch.cuda.is_available():

    model = model.cuda()
for epoch in range(20): #epochs):

    

    loss_train = 0

    for i, (images, coords) in enumerate(train_loader):

        model.train()

        if torch.cuda.is_available():

            images = images.cuda()

            coords = coords.cuda()

        

        images = Variable(images)

        coords = Variable(coords)

        

        optimizer.zero_grad()

        outputs = model(images)

        

        train_loss = criterion(outputs, coords)

        loss_train += train_loss.item()*images.size(0)

        

        train_loss.backward()

        optimizer.step()

        

        if i%10 == 0:

            print('[%d/%d]  \t Train Loss: %.4f ' % ( i, len(train_loader), loss_train/(i+1) ) )

            

            '''

            model.eval()

            loss_val = 0

            for images, coords in val_loader:

                if torch.cuda.is_available():

                    images = images.cuda()

                    coords = coords.cuda()

                    

                images = Variable(images)

                coords = Variable(coords)

                

                outputs = model(images)

                val_loss = criterion(outputs, coords)

                loss_val += val_loss.item()*images.size(0)

            '''

            

    print('-'*100)        

    print('Epoch: %d \t Train Loss: %.4f ' % ( epoch+1, loss_train/len(train_loader) ) ) 

    torch.save(model, 'model.pth')

    print('-'*100)        

    
path = '../input/test/test.csv'

df_test = pd.read_csv(path)
def get_test_image(df_test, index):

    arr = df_test.iloc[index]['Image'].split(' ')

    arr = [int(elem) for elem in arr]

    arr = np.array(arr)

    arr = np.reshape(arr, (96, 96))

    arr = arr / 255.0

    

    return arr
def predict(model, df_test, index, show=False):

    image = get_test_image(df_test, index)

    image_numpy = image

    

    image = torch.Tensor(image)

    image = torch.unsqueeze(image, 0)

    image = torch.unsqueeze(image, 0)

    

    model.eval()

    if torch.cuda.is_available():

        model = model.cuda()

        image = image.cuda()

    

    output = model(image)

    coords = output.detach().cpu().numpy() * 96.0

    coords[coords < 0] =  0

    coords = np.reshape(coords, (15, 2))

    

    if show:

        plt.imshow(image_numpy, 'gray')

        for (x, y) in coords:

            mcircle = mpatches.Circle((x, y), 1, color='r', fill=True)

            plt.gca().add_patch(mcircle)

        

    return coords

     
output = []

for i in range(df_test.shape[0]):

    coord = predict(model, df_test, i)

    output.append(coord.reshape(-1))
output = np.array(output)

output.shape
!ls
lookup = pd.read_csv('../input/IdLookupTable.csv')

lookup.head()
lookup.shape


for i in range(lookup.shape[0]):

    image_id = lookup.iloc[i, 1]-1

    

    feature = {}

    feature['left_eye_center_x'], feature['left_eye_center_y']                  =  output[image_id][0], output[image_id][1]

    feature['right_eye_center_x'], feature['right_eye_center_y']                =  output[image_id][2], output[image_id][3]

    feature['left_eye_inner_corner_x'], feature['left_eye_inner_corner_y']      =  output[image_id][4], output[image_id][5]

    feature['left_eye_outer_corner_x'], feature['left_eye_outer_corner_y']      =  output[image_id][6], output[image_id][7]

    feature['right_eye_inner_corner_x'], feature['right_eye_inner_corner_y']    =  output[image_id][8], output[image_id][9]

    feature['right_eye_outer_corner_x'], feature['right_eye_outer_corner_y']    =  output[image_id][10], output[image_id][11]          

    feature['left_eyebrow_inner_end_x'], feature['left_eyebrow_inner_end_y']    =  output[image_id][12], output[image_id][13]

    feature['left_eyebrow_outer_end_x'], feature['left_eyebrow_outer_end_y']    =  output[image_id][14], output[image_id][15]

    feature['right_eyebrow_inner_end_x'], feature['right_eyebrow_inner_end_y']  =  output[image_id][16], output[image_id][17]

    feature['right_eyebrow_outer_end_x'], feature['right_eyebrow_outer_end_y']  =  output[image_id][18], output[image_id][19]

    feature['nose_tip_x'], feature['nose_tip_y']                                =  output[image_id][20], output[image_id][21]

    feature['mouth_left_corner_x'], feature['mouth_left_corner_y']              =  output[image_id][22], output[image_id][23]

    feature['mouth_right_corner_x'], feature['mouth_right_corner_y']            =  output[image_id][24], output[image_id][25]

    feature['mouth_center_top_lip_x'], feature['mouth_center_top_lip_y']        =  output[image_id][26], output[image_id][27]

    feature['mouth_center_bottom_lip_x'], feature['mouth_center_bottom_lip_y']  =  output[image_id][28], output[image_id][29]

    

    feature_name = lookup.iloc[i, 2]

    

    lookup.iloc[i,3] = feature[feature_name] 
lookup.head()
submission = lookup[['RowId', 'Location']]
submission.to_csv('submission.csv', index=False)
!ls
!ls