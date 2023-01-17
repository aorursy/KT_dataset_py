### install facenet-pytorch to implement it later on in the code
!pip install facenet-pytorch
# importing libraries
from facenet_pytorch import MTCNN, InceptionResnetV1
from tqdm import tqdm
import numpy as np
from glob import glob

# load filenames for human and dog images
human_files = np.array(glob("/kaggle/input/dogimages/lfw/lfw/*/*"))
dog_files = np.array(glob("/kaggle/input/dogimages/dogImages/dogImages/*/*/*"))

# print number of images in each dataset
print('There are %d total human images.' % len(human_files))
print('There are %d total dog images.' % len(dog_files))
import cv2                
import matplotlib.pyplot as plt                        
%matplotlib inline                               

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('/kaggle/input/library/haarcascades/haarcascade_frontalface_alt.xml')
lbp_cascade = cv2.CascadeClassifier('/kaggle/input/lbpcascade/lbpcascade_frontalface.xml')

# load color (BGR) image
img = cv2.imread(human_files[0])
# convert BGR image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# find faces in image (faces is a numpy array)
faces = lbp_cascade.detectMultiScale(gray)

# print number of faces detected in the image
print('Number of faces detected:', len(faces))

# get bounding box for each detected face
for (x,y,w,h) in faces:
    # add bounding box to color image
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
# convert BGR image to RGB for plotting
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# display the image, along with bounding box
plt.imshow(cv_rgb)
plt.show()
# create face detector
mtcnn = MTCNN(keep_all=True)

# load color (BGR) image
img = cv2.imread(human_files[3])

# Detect face
boxes, probs, landmarks = mtcnn.detect(img, landmarks=True)

# Visualize
fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(img)
ax.axis('off')

for box, landmark in zip(boxes, landmarks):
    ax.scatter(*np.meshgrid(box[[0, 2]], box[[1, 3]]))
    ax.scatter(landmark[:, 0], landmark[:, 1], s=8)
fig.show()

def face_detector(img_path, algorithm = "face_cascade"):
    """Function to detect human face in an image
    Args: 
        img_path: string value file path
        algorithm: string name of what type of algorithm is used to detect the face (default = "face_cascade")
                   lbp_cascade, facenet 
        
    Returns: 
        boolean: 
            True if the a humann face is detected, False if it not
    """
    # read the image from the file path
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = list()
    
    if (algorithm == "face_cascade"):             
        faces = face_cascade.detectMultiScale(gray)
        
    elif (algorithm == "lbp_cascade"):
        faces = lbp_cascade.detectMultiScale(gray)
        
    elif (algorithm == "facenet"):
        mtcnn = MTCNN(keep_all=True)
        # Detect face(s)
        faces = mtcnn(img)
        
    return (faces is not None and len(faces) > 0)
def test_algorithm(algorithm):
    """Function to test face detection algorithm for a given images of human and dogs
    Args: 
        algorithm: string name of what type of algorithm is used to detect the face (default = "face_cascade")
                   lbp_cascade, facenet         
    Returns: 
       None
    """
    human_files_short = human_files[:100]
    dog_files_short = dog_files[:100]

    human_faces_human_files = 0
    human_faces_dog_files = 0
    for human_img in human_files_short:    
        # check if the face detector is detecting faces in the human faces dataset 
        if(face_detector(human_img, algorithm=algorithm)):
            human_faces_human_files += 1 

    for dog_img in dog_files_short:      
        # if it is not detecting faces in the dog images, then we are good to go   
        if(face_detector(dog_img, algorithm=algorithm)):
            human_faces_dog_files += 1


    print("Number of detected faces in human files : {}% \n Number of detected faces in the dog files : {}%".format(
                                                                    (human_faces_human_files/len(human_files_short))*100 ,
                                                                     (human_faces_dog_files/len(dog_files_short))*100))
test_algorithm("face_cascade")
test_algorithm("lbp_cascade")
test_algorithm("facenet")
import torch
import torchvision.models as models

# define VGG16 model
VGG16 = models.vgg16(pretrained=True)

# check if CUDA is available
use_cuda = torch.cuda.is_available()

# move model to GPU if CUDA is available
if use_cuda:
    VGG16 = VGG16.cuda()
from PIL import Image
import torchvision.transforms as transforms

def process_img(img_path, img_size):    
    '''Function to batched an image and process it as input for a model
    
    Args:
        img_path: string value path to an image
        img_size: integer value the desired image size
        
    Returns:
        batched image
    '''
    
    ## Load and pre-process an image from the given img_path
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    test_transforms  = transforms.Compose([transforms.CenterCrop(img_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=mean, std=std)])
    img = Image.open(img_path)
    img_t = test_transforms(img)
    batch_t = img_t.unsqueeze(0)
    
    return batch_t
    
def predict(model, img_path, img_size):
    '''
    Use pre-trained VGG-16 model to obtain index corresponding to 
    predicted ImageNet class for image at specified path
    
    Args:
         model: takes a pretrained model
         img_path: path to an image
         img_size: integer value for image size
        
    Returns:
        Index corresponding to VGG-16 model's prediction
    '''
    
    ## TODO: Complete the function.
    ## Load and pre-process an image from the given img_path
    
    batch_t = process_img(img_path, img_size)
    if use_cuda:
        batch_t = batch_t.cuda()
    model.eval()
    output = model(batch_t)
    _, pred = torch.max(output, 1)

    ## Return the *index* of the predicted class for that image
    
    return int(pred) # predicted class index
### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(model, img_path, img_size):
    ## TODO: Complete the function.
    pred = predict(model, img_path, img_size)
    if(pred >=151 and pred <=268):
        return True
    else:
        return False
### TODO: Test the performance of the dog_detector function
### on the images in human_files_short and dog_files_short.

from tqdm import tqdm

human_files_short = human_files[:100]
dog_files_short = dog_files[:100]

#-#-# Do NOT modify the code above this line. #-#-#

## TODO: Test the performance of the face_detector algorithm 
## on the images in human_files_short and dog_files_short.
def assess_dog_detector(model, img_size):
    dog_faces_human_files = 0
    dog_faces_dog_files = 0
    
    for human_img in human_files_short:
    
        # check if the face detector is detecting faces in the hyman faces dataset 
        if(dog_detector(model, human_img, img_size)):
            dog_faces_human_files += 1 

    for dog_img in dog_files_short:      
        # if it is not detecting faces in the dog images, then we are good to go   
        if(dog_detector(model, dog_img, img_size)):
            dog_faces_dog_files += 1


    print("Number of detected dogs in human files : {}% \nNumber of detected dogs in the dog files : {}%".format(
                                                                    (dog_faces_human_files/len(human_files_short))*100 ,
                                                                     (dog_faces_dog_files/len(dog_files_short))*100))

assess_dog_detector(VGG16,  img_size=224)
### (Optional) 
### TODO: Report the performance of another pre-trained network.
### Feel free to use as many code cells as needed.

# download a pretrained ResNet model
resnet = models.resnet50(pretrained=True)

# check if CUDA is available
use_cuda = torch.cuda.is_available()

# move model to GPU if CUDA is available
if use_cuda:
    resnet = resnet.cuda()
    

assess_dog_detector(resnet, img_size=224)
# download a pretrained Inception model
inception = models.inception_v3(pretrained=True)

# check if CUDA is available
use_cuda = torch.cuda.is_available()

# move model to GPU if CUDA is available
if use_cuda:
    inception = inception.cuda()
assess_dog_detector(inception,  img_size=224)
import os
from torchvision import datasets
import torch

### TODO: Write data loaders for training, validation, and test sets
## Specify appropriate transforms, and batch_sizes

import os
from torchvision import datasets

### TODO: Write data loaders for training, validation, and test sets
## Specify appropriate transforms, and batch_sizes
batch_size = 20

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.CenterCrop(112),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean = mean, std = std)
    ])

test_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.CenterCrop(112),
    transforms.ToTensor(),
    transforms.Normalize(mean = mean, std = std)
    ])

# Choose the trainig, test, and validation set
data_dir = "/kaggle/input/dogimages/dogImages/dogImages/"
# Pass transforms in here, then run the next cell to see how the transforms look
train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)
valid_data =  datasets.ImageFolder(data_dir + '/valid', transform=test_transforms)
# prepare the data loaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size)

dataset_size = len(train_data+test_data+valid_data)
classes = train_data.classes
number_classes = len(train_data.classes)
print("data set size: ", dataset_size)
print("Number of classes: ", number_classes)
import torch.nn as nn
import torch.nn.functional as F

# define the CNN architecture
class Net(nn.Module):
    ### TODO: choose an architecture, and complete the class
    def __init__(self):
        super(Net, self).__init__()
        ## Define layers of a CNN
        
        # image size 112X112
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)   
        self.bn1 = nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # image size after maxpooling 56X56 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) 
        self.bn2 = nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # image size after maxpooling 28X28     
        self.conv3 = nn.Conv2d(64, 64,  kernel_size=3, padding=1)  
        self.bn3 = nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # image size after maxpooling 14X14 
        self.conv4 = nn.Conv2d(64, 128,  kernel_size=3, padding=1)  
        self.bn4 = nn.BatchNorm2d(num_features=128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # image size after maxpooling 7X7 
        self.conv5 = nn.Conv2d(128, 256,  kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(num_features=256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        # define the pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        #define the linear layers (input 3 * 3 * 256)
        self.dropout = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(3 * 3 * 256, 4608)
        self.fc2 = nn.Linear(4608, 2304)
        self.fc3 = nn.Linear(2304, 1024)
        self.fc4 = nn.Linear(1024, number_classes)
    
    def forward(self, x):
        ## Define forward behavior
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        
        x = x.view(-1, 3 * 3* 256)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x

#-#-# You do NOT have to modify the code below this line. #-#-#

# instantiate the CNN
model_scratch = Net()


# apply normal distribution rule ro initialize the weights
def weights_init_normal(m):
    '''Takes in a module and initializes all linear layers with weight
       values taken from a normal distribution.'''
    
    classname = m.__class__.__name__
    # for every Linear layer in a model
    if classname.find('Linear') != -1:
        # m.weight.data shoud be taken from a normal distribution
        n = m.in_features
        y = 1.0/np.sqrt(n)    
        m.weight.data.normal_(0, y)
        # m.bias.data should be 0
        m.bias.data.fill_(0)
        
model_scratch.apply(weights_init_normal)

# move tensors to GPU if CUDA is available
if use_cuda:
    model_scratch.cuda()
import torch.optim as optim

### TODO: select loss function
criterion_scratch = nn.CrossEntropyLoss()

### TODO: select optimizer
optimizer_scratch = optim.SGD(model_scratch.parameters(), lr=0.005, momentum=0.9)
# the following import is required for training to be robust to truncated images
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def train(n_epochs, trainloader, testloader, validloader , model, optimizer, criterion, use_cuda, save_path):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf 
    
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        
        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(trainloader):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            # clear the gradient
            optimizer.zero_grad()
            output = model(data)
            
            ## find the loss and update the model parameters accordingly
            loss = criterion(output, target)
            loss.backward()
            # Perform the optimizer step
            optimizer.step()
            ## record the average training loss, using something like
            ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            train_loss += loss.item() * data.size(0)
            
        ######################    
        # validate the model #
        ######################
        model.eval()
        for batch_idx, (data, target) in enumerate(validloader):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()           
            # Pass data acrss the network
            output = model(data)
            loss = criterion(output, target)
            ## update the average validation loss
            valid_loss += loss.item() * data.size(0)
        
        # Calculate the average losses
        train_loss = train_loss/len(trainloader)
        valid_loss = valid_loss/len(validloader)
    
            
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))
        
        ## TODO: save the model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            valid_loss_min = valid_loss
            # print the decremnet in the validation
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min, 
            valid_loss))
            torch.save(model.state_dict(), save_path)
            
    # return trained model
    return model

# train the model
model_scratch = train(30, trainloader, testloader, validloader, model_scratch, optimizer_scratch, 
                      criterion_scratch, use_cuda, 'model_scratch.pt')
# load the model that got the best validation accuracy
model_scratch.load_state_dict(torch.load('model_scratch.pt'))
def test(testloader, model, criterion, use_cuda):
    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    model.eval()
    for batch_idx, (data, target) in enumerate(testloader):
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss 
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
            
    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))


# call test function    
test(testloader, model_scratch, criterion_scratch, use_cuda)
## TODO: Specify data loaders

batch_size = 20

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transfer_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean = mean, std = std)
    ])

test_transfer_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean = mean, std = std)
    ])

# Pass transforms in here, then run the next cell to see how the transforms look
train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transfer_transforms)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transfer_transforms)
valid_data =  datasets.ImageFolder(data_dir + '/valid', transform=test_transfer_transforms)
# prepare the data loaders
trainloader_transfer = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
testloader_transfer = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
validloader_transfer = torch.utils.data.DataLoader(valid_data, batch_size=batch_size)

# assigning the pretrained model
model_transfer = resnet

model_transfer
# Building the network
import torchvision.models as models
import torch.nn as nn

## TODO: Specify model architecture 

#freeze the model calssifier
for param in  model_transfer.parameters():
    param.requires_grad = False

from collections import OrderedDict

classifier = nn.Sequential(OrderedDict([                           
                          ('fc1', nn.Linear(2048, 1024)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(0.2)),
                          ('fc2', nn.Linear(1024, 512)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(0.2)),
                          ('fc3', nn.Linear(512, number_classes)),
                          ('output', nn.LogSoftmax(dim=1))]))

model_transfer.fc = classifier
model_transfer
if use_cuda:
    model_transfer = model_transfer.cuda()
model_transfer
# Initializing the weights
model_transfer.fc.apply(weights_init_normal)
from torch import optim
criterion_transfer = nn.NLLLoss()
optimizer_transfer = optim.SGD(model_transfer.fc.parameters(), lr=0.001, momentum=0.9)
# train the model
model_transfer = train(20, trainloader_transfer, testloader_transfer, validloader_transfer,
                       model_transfer, optimizer_transfer, criterion_transfer, use_cuda, 'model_transfer.pt')


# load the model that got the best validation accuracy
model_transfer.load_state_dict(torch.load('model_transfer.pt'))
test(testloader_transfer, model_transfer, criterion_transfer, use_cuda)
### TODO: Write a function that takes a path to an image as input
### and returns the dog breed that is predicted by the model.

# list of class names by index, i.e. a name can be accessed like class_names[0]
class_names = [item[4:].replace("_", " ") for item in train_data.classes]

def predict_breed_transfer(img_path):
    '''Function that takes a path to an image as input 
    and returns the dog breed that is predicted by the model
    Args:
         img_path: string path to an image
        
    Returns:
        Index corresponding to our model's prediction
    '''
    # load the image and return the predicted breed
    # pre_process the image
    t_batch = process_img(img_path, 224)
    model_transfer.eval()
    t_batch = t_batch.cuda()
    output = model_transfer(t_batch)
    _, pred = torch.max(output, 1)    
    class_name = classes[int(pred)]
    # Get the breed name
    class_name = class_name[class_name.find('.')+1 :].replace("_", " ")
    return class_name
### TODO: Write your algorithm.
### Feel free to use as many code cells as needed.

def run_app(img_path):
    # load color (BGR) image
    img = cv2.imread(img_path)
    # convert BGR image to RGB for plotting
    cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ## handle cases for a dog,human face, and neither
    if dog_detector(VGG16, img_path, img_size=224) == True:      
        # display the image, along with bounding box
        plt.imshow(cv_rgb)
        plt.suptitle("Welcome!") 
        plt.title("The {} gathering is over ther..".format(predict_breed_transfer(img_path)))
                
    elif face_detector(img_path) == True:             
        plt.imshow(cv_rgb)
        plt.suptitle("Got you human!")  
        plt.title("You are trying to pose as a {}".format(predict_breed_transfer(img_path)))
                         
    else:
        plt.imshow(cv_rgb)
        plt.suptitle( "Un-Identified object Aalert!") 
        plt.title("Posing as {}".format(predict_breed_transfer(img_path)))
                     
    plt.show()    
    

## TODO: Execute your algorithm from Step 6 on
## at least 6 images on your computer.
## Feel free to use as many code cells as needed.

## suggested code, below
for file in np.hstack((human_files[-4:-1], dog_files[-4:-1])):
    run_app(file)
img_path = "/kaggle/input/testimgs/me.jpg"
run_app(img_path)
img_path = "/kaggle/input/testimgs/Rocky.jpg"
run_app(img_path)
