# load the data if you need to; if you have already loaded the data, you may comment this cell out

# -- DO NOT CHANGE THIS CELL -- #

!mkdir /data

!wget -P /data/ https://s3.amazonaws.com/video.udacity-data.com/topher/2018/May/5aea1b91_train-test-data/train-test-data.zip

!unzip -n /data/train-test-data.zip -d /data
# import the usual resources

import matplotlib.pyplot as plt

import numpy as np



# import utilities to keep workspaces alive during model training

# from workspace_utils import active_session



# watch for any changes in model.py, if it changes, re-load it automatically

%load_ext autoreload

%autoreload 2
## TODO: Define the Net in models.py



import torch

import torch.nn as nn

import torch.nn.functional as F

# can use the below import should you choose to initialize the weights of your Net

import torch.nn.init as I



## TODO: define the convolutional neural network architecture



class Net(nn.Module):



    def __init__(self):

        super(Net, self).__init__()

        

        ## TODO: Define all the layers of this CNN, the only requirements are:

        ## 1. This network takes in a square (same width and height), grayscale image as input

        ## 2. It ends with a linear layer that represents the keypoints

        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs

        

        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:

        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel

        self.conv1 = nn.Conv2d(1, 32, 5)

        self.pool1 = nn.MaxPool2d(2,2)

#         self.dropout1 = nn.Dropout(p=0.1)

                

        ## Note that among the layers to add, consider including:

        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting

        self.conv2 = nn.Conv2d(32, 64, 3)

        self.pool2 = nn.MaxPool2d(2,2)

#         self.dropout2 = nn.Dropout(p=0.2)



        self.conv3 = nn.Conv2d(64, 128, 3)

        self.pool3 = nn.MaxPool2d(2,2)

#         self.dropout3 = nn.Dropout(p=0.2)



        self.conv4 = nn.Conv2d(128, 256, 3)

        self.pool4 = nn.MaxPool2d(2,2)

#         self.dropout4 = nn.Dropout(p=0.3)



        self.conv5 = nn.Conv2d(256, 512, 1)

        self.pool5 = nn.MaxPool2d(2,2)

#         self.dropout5 = nn.Dropout(p=0.3)



        self.fc1 = nn.Linear(512*6*6,512*6*3)

#         self.fc1_bn = nn.BatchNorm1d(1024)

        self.fc1_drop = nn.Dropout(p=0.4)

    

        self.fc2 = nn.Linear(512*6*3,1024)

#         self.fc1_bn = nn.BatchNorm1d(1024)

        self.fc2_drop = nn.Dropout(p=0.4)

    

        self.fc3 = nn.Linear(1024, 136)

        

    def forward(self, x):

        ## TODO: Define the feedforward behavior of this model

        ## x is the input image and, as an example, here you may choose to include a pool/conv step:

        ## x = self.pool(F.relu(self.conv1(x)))



        x = self.conv1(x)

        x = F.relu(x)

        x = self.pool1(x)

#         x = self.dropout1(x)

        

        x = self.conv2(x)

        x = F.relu(x)

        x = self.pool2(x)

#         x = self.dropout2(x)



        x = self.conv3(x)

        x = F.relu(x)

        x = self.pool3(x)

#         x = self.dropout3(x)



        x = self.conv4(x)

        x = F.relu(x)

        x = self.pool4(x)

#         x = self.dropout4(x)



        x = self.conv5(x)

        x = F.relu(x)

        x = self.pool5(x)

#         x = self.dropout5(x)

        

        #Prep for Linear layer

        x = x.view(x.size(0), -1)

        

        x = self.fc1(x)

        x = F.relu(x)

#         x = self.fc1_bn(x)

        x = self.fc1_drop(x)

    

        x = self.fc2(x)

        x = F.relu(x)

        x = self.fc2_drop(x)

    

        x = self.fc3(x)

        

        # a modified x, having gone through all the layers of your model, should be returned

        return x

## TODO: Once you've define the network, you can instantiate it

# one example conv layer has been provided for you

# from models import Net

net = Net()



if torch.cuda.is_available():

    net.cuda()

print(net)
import glob

import os

import torch

from torch.utils.data import Dataset, DataLoader

import numpy as np

import matplotlib.image as mpimg

import pandas as pd

import cv2





class FacialKeypointsDataset(Dataset):

    """Face Landmarks dataset."""



    def __init__(self, csv_file, root_dir, transform=None):

        """

        Args:

            csv_file (string): Path to the csv file with annotations.

            root_dir (string): Directory with all the images.

            transform (callable, optional): Optional transform to be applied

                on a sample.

        """

        self.key_pts_frame = pd.read_csv(csv_file)

        self.root_dir = root_dir

        self.transform = transform



    def __len__(self):

        return len(self.key_pts_frame)



    def __getitem__(self, idx):

        image_name = os.path.join(self.root_dir,

                                self.key_pts_frame.iloc[idx, 0])

        

        image = mpimg.imread(image_name)

        

        # if image has an alpha color channel, get rid of it

        if(image.shape[2] == 4):

            image = image[:,:,0:3]

        

        key_pts = self.key_pts_frame.iloc[idx, 1:].as_matrix()

        key_pts = key_pts.astype('float').reshape(-1, 2)

        sample = {'image': image, 'keypoints': key_pts}



        if self.transform:

            sample = self.transform(sample)



        return sample

    



    

# tranforms



class Normalize(object):

    """Convert a color image to grayscale and normalize the color range to [0,1]."""        



    def __call__(self, sample):

        image, key_pts = sample['image'], sample['keypoints']

        

        image_copy = np.copy(image)

        key_pts_copy = np.copy(key_pts)



        # convert image to grayscale

        image_copy = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        

        # scale color range from [0, 255] to [0, 1]

        image_copy=  image_copy/255.0

            

        

        # scale keypoints to be centered around 0 with a range of [-1, 1]

        # mean = 100, sqrt = 50, so, pts should be (pts - 100)/50

        key_pts_copy = (key_pts_copy - 100)/50.0





        return {'image': image_copy, 'keypoints': key_pts_copy}





class Rescale(object):

    """Rescale the image in a sample to a given size.



    Args:

        output_size (tuple or int): Desired output size. If tuple, output is

            matched to output_size. If int, smaller of image edges is matched

            to output_size keeping aspect ratio the same.

    """



    def __init__(self, output_size):

        assert isinstance(output_size, (int, tuple))

        self.output_size = output_size



    def __call__(self, sample):

        image, key_pts = sample['image'], sample['keypoints']



        h, w = image.shape[:2]

        if isinstance(self.output_size, int):

            if h > w:

                new_h, new_w = self.output_size * h / w, self.output_size

            else:

                new_h, new_w = self.output_size, self.output_size * w / h

        else:

            new_h, new_w = self.output_size



        new_h, new_w = int(new_h), int(new_w)



        img = cv2.resize(image, (new_w, new_h))

        

        # scale the pts, too

        key_pts = key_pts * [new_w / w, new_h / h]



        return {'image': img, 'keypoints': key_pts}





class RandomCrop(object):

    """Crop randomly the image in a sample.



    Args:

        output_size (tuple or int): Desired output size. If int, square crop

            is made.

    """



    def __init__(self, output_size):

        assert isinstance(output_size, (int, tuple))

        if isinstance(output_size, int):

            self.output_size = (output_size, output_size)

        else:

            assert len(output_size) == 2

            self.output_size = output_size



    def __call__(self, sample):

        image, key_pts = sample['image'], sample['keypoints']



        h, w = image.shape[:2]

        new_h, new_w = self.output_size



        top = np.random.randint(0, h - new_h)

        left = np.random.randint(0, w - new_w)



        image = image[top: top + new_h,

                      left: left + new_w]



        key_pts = key_pts - [left, top]



        return {'image': image, 'keypoints': key_pts}





class ToTensor(object):

    """Convert ndarrays in sample to Tensors."""



    def __call__(self, sample):

        image, key_pts = sample['image'], sample['keypoints']

         

        # if image has no grayscale color channel, add one

        if(len(image.shape) == 2):

            # add that third color dim

            image = image.reshape(image.shape[0], image.shape[1], 1)

            

        # swap color axis because

        # numpy image: H x W x C

        # torch image: C X H X W

        image = image.transpose((2, 0, 1))

        

        return {'image': torch.from_numpy(image),

                'keypoints': torch.from_numpy(key_pts)}
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, utils



# the dataset we created in Notebook 1 is copied in the helper file `data_load.py`

# from data_load import FacialKeypointsDataset

# the transforms we defined in Notebook 1 are in the helper file `data_load.py`

# from data_load import Rescale, RandomCrop, Normalize, ToTensor





## TODO: define the data_transform using transforms.Compose([all tx's, . , .])

# order matters! i.e. rescaling should come before a smaller crop

data_transform = transforms.Compose([Rescale(256), 

                                      RandomCrop(224), 

                                      Normalize(), 

                                      ToTensor()])



# testing that you've defined a transform

assert(data_transform is not None), 'Define a data_transform'
# create the transformed dataset

transformed_dataset = FacialKeypointsDataset(csv_file='/data/training_frames_keypoints.csv',

                                             root_dir='/data/training/',

                                             transform=data_transform)





print('Number of images: ', len(transformed_dataset))



# iterate through the transformed dataset and print some stats about the first few samples

for i in range(4):

    sample = transformed_dataset[i]

    print(i, sample['image'].size(), sample['keypoints'].size())
# load training data in batches

batch_size = 64



train_loader = DataLoader(transformed_dataset, 

                          batch_size=batch_size,

                          shuffle=True, 

                          num_workers=4)

# load in the test data, using the dataset class

# AND apply the data_transform you defined above



# create the test dataset

test_dataset = FacialKeypointsDataset(csv_file='/data/test_frames_keypoints.csv',

                                             root_dir='/data/test/',

                                             transform=data_transform)



# load test data in batches

batch_size = 10



test_loader = DataLoader(test_dataset, 

                          batch_size=batch_size,

                          shuffle=True, 

                          num_workers=4)
# test the model on a batch of test images



def net_sample_output():

    

    # iterate through the test dataset

    for i, sample in enumerate(test_loader):

        

        # get sample data: images and ground truth keypoints

        images = sample['image']

        key_pts = sample['keypoints']



        # convert images to FloatTensors

        if torch.cuda.is_available():

            images = images.type(torch.cuda.FloatTensor)

        else:

            images = images.type(torch.FloatTensor)



        # forward pass to get net output

        output_pts = net(images)

        

        # reshape to batch_size x 68 x 2 pts

        output_pts = output_pts.view(output_pts.size()[0], 68, -1)

        

        # break after first image is tested

        if i == 0:

            return images, output_pts, key_pts

            
# call the above function

# returns: test images, test predicted keypoints, test ground truth keypoints

test_images, test_outputs, gt_pts = net_sample_output()



# print out the dimensions of the data to see if they make sense

print(test_images.data.size())

print(test_outputs.data.size())

print(gt_pts.size())
def show_all_keypoints(image, predicted_key_pts, gt_pts=None):

    """Show image with predicted keypoints"""

    # image is grayscale

    plt.imshow(image, cmap='gray')

    plt.scatter(predicted_key_pts[:, 0], predicted_key_pts[:, 1], s=20, marker='.', c='m')

    # plot ground truth points as green pts

    if gt_pts is not None:

        plt.scatter(gt_pts[:, 0], gt_pts[:, 1], s=20, marker='.', c='g')

# visualize the output

# by default this shows a batch of 10 images

def visualize_output(test_images, test_outputs, gt_pts=None, batch_size=10):



    for i in range(batch_size):

        plt.figure(figsize=(20,10))

        ax = plt.subplot(1, batch_size, i+1)



        # un-transform the image data

        image = test_images[i].data   # get the image from it's Variable wrapper

        if torch.cuda.is_available():

            image = image.cpu()

            

        image = image.numpy()   # convert to numpy array from a Tensor

        image = np.transpose(image, (1, 2, 0))   # transpose to go from torch to numpy image



        # un-transform the predicted key_pts data

        predicted_key_pts = test_outputs[i].data

        if torch.cuda.is_available():

            predicted_key_pts = predicted_key_pts.cpu()

            

        predicted_key_pts = predicted_key_pts.numpy()

        # undo normalization of keypoints  

        predicted_key_pts = predicted_key_pts*50.0+100

        

        # plot ground truth points for comparison, if they exist

        ground_truth_pts = None

        if gt_pts is not None:

            ground_truth_pts = gt_pts[i]         

            ground_truth_pts = ground_truth_pts*50.0+100

        

        # call show_all_keypoints

        show_all_keypoints(np.squeeze(image), predicted_key_pts, ground_truth_pts)

            

        plt.axis('off')



    plt.show()

    

# call it

visualize_output(test_images, test_outputs, gt_pts)
## TODO: Define the loss and optimization

import torch.optim as optim



criterion = nn.SmoothL1Loss()



optimizer = optim.Adam(net.parameters(),lr=0.001)

def train_net(n_epochs):



    # prepare the net for training

    net.train()

    losses = []   

    for epoch in range(n_epochs):  # loop over the dataset multiple times

        

        running_loss = 0.0



        # train on batches of data, assumes you already have train_loader

        for batch_i, data in enumerate(train_loader):

            # get the input images and their corresponding labels

            images = data['image']

            key_pts = data['keypoints']



            # flatten pts

            key_pts = key_pts.view(key_pts.size(0), -1)



            # convert variables to floats for regression loss

            if torch.cuda.is_available():

                key_pts = key_pts.cuda().float()

                images = images.cuda().float()

            else:

                key_pts = key_pts.type(torch.FloatTensor)

                images = images.type(torch.FloatTensor)

            # forward pass to get outputs

            output_pts = net(images)



            # calculate the loss between predicted and target keypoints

            loss = criterion(output_pts, key_pts)



            # zero the parameter (weight) gradients

            optimizer.zero_grad()

            

            # backward pass to calculate the weight gradients

            loss.backward()



            # update the weights

            optimizer.step()



            # print loss statistics

            running_loss += loss.item()

            if batch_i % 10 == 9:    # print every 10 batches

#                 print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i+1, running_loss))

                losses.append(running_loss/10)

                running_loss = 0.0

            

        print(f'Epoch: {epoch + 1}, Avg. Training Loss: {running_loss:.5f}')

    print('Finished Training')

    return losses
import warnings

warnings.filterwarnings('ignore')
# train your network

n_epochs = 50 # start small, and increase when you've decided on your model structure and hyperparams



# this is a Workspaces-specific context manager to keep the connection

# alive while training your model, not part of pytorch



# with active_session():

losses = train_net(n_epochs)
plt.figure(figsize=(20, 10))

plt.plot(losses[1:])
plt.savefig('loss.png')
# get a sample of test data again

test_images, test_outputs, gt_pts = net_sample_output()



print(test_images.data.size())

print(test_outputs.data.size())

print(gt_pts.size())
def net_test_loss():

    avg_loss = 0.0

    with torch.no_grad():

        for i, sample in enumerate(test_loader):

        

            # get sample data: images and ground truth keypoints

            images = sample['image']

            key_pts = sample['keypoints']



            # flatten pts

            key_pts = key_pts.view(key_pts.size(0), -1)



            # convert variables to floats for regression loss

            if torch.cuda.is_available():

                key_pts = key_pts.cuda().float()

                images = images.cuda().float()

            else:

                key_pts = key_pts.type(torch.FloatTensor)

                images = images.type(torch.FloatTensor)



            # forward pass to get net output

            output_pts = net(images)

            loss = criterion(output_pts, key_pts)



            avg_loss += loss.item() / len(test_loader)

    return avg_loss
net_test_loss()
## TODO: visualize your test output

# you can use the same function as before, by un-commenting the line below:



# visualize_output(test_images, test_outputs, gt_pts)

visualize_output(test_images, test_outputs, gt_pts)
## TODO: change the name to something uniqe for each new model

# model_dir = 'saved_models/'

model_dir = '/kaggle/working'

model_name = 'keypoints_model_1.pt'



# after training, save your model parameters in the dir 'saved_models'

torch.save(net.state_dict(), model_name)
from IPython.display import FileLink

FileLink(r'keypoints_model_1.pt')
# Get the weights in the first conv layer, "conv1"

# if necessary, change this to reflect the name of your first conv layer

weights1 = net.conv1.weight.data



w = weights1.cpu().numpy()



filter_index = 0



print(w[filter_index][0])

print(w[filter_index][0].shape)



# display the filter weights

plt.imshow(w[filter_index][0], cmap='gray')

##TODO: load in and display any image from the transformed test dataset

import cv2

image = test_images[1].data   # get the image from it's Variable wrapper

image = image.cpu().numpy()   # convert to numpy array from a Tensor

image = np.squeeze(np.transpose(image, (1, 2, 0)))  # transpose to go from torch to numpy image

weights1 = net.conv1.weight.data

w = weights1.cpu().numpy()

filter_index = 31



## TODO: Using cv's filter2D function,

## apply a specific set of filter weights (like the one displayed above) to the test image

filtered = cv2.filter2D(image, -1, w[filter_index][0])

fig = plt.figure(figsize=(15,10))

ax = fig.add_subplot(221)

ax.imshow(image, cmap = 'gray')

ax.set_title("Image")

ax = fig.add_subplot(222)

ax.imshow(filtered, cmap = 'gray')

ax.set_title("Filtered Image")

ax = fig.add_subplot(223)

ax.imshow(w[filter_index][0], cmap = 'gray')

ax.set_title("Filter")

plt.show()