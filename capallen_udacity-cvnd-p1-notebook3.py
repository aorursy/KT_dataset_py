import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

%matplotlib inline
import cv2

# load in color image for face detection

image = cv2.imread('../input/udacitycvnd/obamas.jpg')



# switch red and blue color channels 

# --> by default OpenCV assumes BLUE comes first, not RED as in many images

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



# plot the image

fig = plt.figure(figsize=(9,9))

plt.imshow(image)
# load in a haar cascade classifier for detecting frontal faces

face_cascade = cv2.CascadeClassifier('../input/frontalface/haarcascade_frontalface_default.xml')



# run the detector

# the output here is an array of detections; the corners of each detection box

# if necessary, modify these parameters until you successfully identify every face in a given image

faces = face_cascade.detectMultiScale(image, 1.2, 2)



# make a copy of the original image to plot detections on

image_with_detections = image.copy()



# loop over the detected faces, mark the image where each face is found

for (x,y,w,h) in faces:

    # draw a rectangle around each detected face

    # you may also need to change the width of the rectangle drawn depending on image resolution

    cv2.rectangle(image_with_detections,(x,y),(x+w,y+h),(255,0,0),3) 



fig = plt.figure(figsize=(9,9))



plt.imshow(image_with_detections)
import torch

import torch.nn as nn

import torch.nn.functional as F

# can use the below import should you choose to initialize the weights of your Net

import torch.nn.init as I

# from models import Net

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



net = Net()



## TODO: load the best saved model parameters (by your path name)

## You'll need to un-comment the line below and add the correct name for *your* saved model

net.load_state_dict(torch.load('../input/udacity-cvnd-p1-notebook2/keypoints_model_1.pt'))



## print out your net and prepare it for testing (uncomment the line below)

net.eval()
def show_all_keypoints(image, predicted_key_pts):

    """Show image with predicted keypoints"""

    # image is grayscale

    plt.figure()

    plt.imshow(image, cmap='gray')

    plt.scatter(predicted_key_pts[:, 0], predicted_key_pts[:, 1], s=5, marker='.', c='m')
image_copy = np.copy(image)



# loop over the detected faces from your haar cascade

for (x,y,w,h) in faces:

    

    # Select the region of interest that is the face in the image 

    roi = image_copy[y-60:y+h+60, x-60:x+w+60]

    ## TODO: Convert the face region from RGB to grayscale

    roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

    ## TODO: Normalize the grayscale image so that its color range falls in [0,1] instead of [0,255]

    roi = roi/255.0

    ## TODO: Rescale the detected face to be the expected square size for your CNN (224x224, suggested)

    roi = cv2.resize(roi, (224, 224))  

    roi = np.reshape(roi,(224, 224, 1))

    ## TODO: Reshape the numpy image shape (H x W x C) into a torch image shape (C x H x W)

    roi = roi.transpose(2, 0, 1)

    roi_ = roi[0]

    ## TODO: Make facial keypoint predictions using your loaded, trained network 

    roi = torch.from_numpy(roi).type(torch.FloatTensor)

    roi = roi.unsqueeze(0)

    keypoints = net(roi)

    keypoints = keypoints.view(68, 2)

    keypoints = keypoints.data.numpy()

    keypoints = keypoints*50.0 + 100

    ## TODO: Display each detected face and the corresponding keypoints        

    plt.figure() 

    show_all_keypoints(roi_, keypoints)

    plt.show()
