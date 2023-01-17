import os

os.listdir('../input/')
import numpy as np

from glob import glob



# load filenames for human and dog images

human_files = np.array(glob("../input/dogclassification/lfw/lfw/*/*"))

dog_files = np.array(glob("../input/dogclassification/dogimages/dogImages/*/*/*"))



# print number of images in each dataset

print('There are %d total human images.' % len(human_files))

print('There are %d total dog images.' % len(dog_files))
import cv2             

import matplotlib.pyplot as plt                        

%matplotlib inline                               



# extract pre-trained face detector

face_cascade = cv2.CascadeClassifier('../input/haarcascades/haarcascade_frontalface_alt.xml')



# load color (BGR) image

img = cv2.imread(human_files[0])

# convert BGR image to grayscale

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



# find faces in image

faces = face_cascade.detectMultiScale(gray)



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
# returns "True" if face is detected in image stored at img_path

def face_detector(img_path):

    img = cv2.imread(img_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray)

    return len(faces) > 0
from tqdm import tqdm



human_files_short = human_files[:100]

dog_files_short = dog_files[:100]



#-#-# Do NOT modify the code above this line. #-#-#



## TODO: Test the performance of the face_detector algorithm 

## on the images in human_files_short and dog_files_short.

def detect_human(files):

    count = 0

    total_count = len(files)

    for file in files:

        if face_detector(file):

            count+=1

    return (count,total_count)
print("Human in Human Files is {} / {}".format(detect_human(human_files_short)[0],detect_human(human_files_short)[1]))

print("Human in Dog Files is {} / {}".format(detect_human(dog_files_short)[0],detect_human(dog_files_short)[1]))
### (Optional) 

### TODO: Test performance of anotherface detection algorithm.

### Feel free to use as many code cells as needed.
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



def load_image(img_path):

    image = Image.open(img_path).convert('RGB')

    # resize to (244,244) because VGG16 accepts that dimension

    in_transform = transforms.Compose([

                        transforms.Resize(size=(244,244)),

                        transforms.ToTensor()]) #Normalization parameter from PyTorch Doc

    # discard the transparent, alpha channel (that's the :3) and add the batch dimension

    image = in_transform(image)[:3,:,:].unsqueeze(0)

    return image
def VGG16_predict(img_path):

    '''

    Use pre-trained VGG-16 model to obtain index corresponding to 

    predicted ImageNet class for image at specified path

    

    Args:

        img_path: path to an image

        

    Returns:

        Index corresponding to VGG-16 model's prediction

    '''

    ## TODO: Complete the function.

    ## Load and pre-process an image from the given img_path

    ## Return the *index* of the predicted class for that image

    img = load_image(img_path)

    if use_cuda:

        img = img.cuda()

    ret = VGG16(img)

    return torch.max(ret,1)[1].item() # predicted class index

VGG16_predict(dog_files[0])
### returns "True" if a dog is detected in the image stored at img_path

def dog_detector(img_path):

    ## TODO: Complete the function.

    chk = VGG16_predict(img_path)

    if chk >=151 and chk<=268:

        return True

    return False # true/false
print(dog_detector(dog_files_short[0]))

print(dog_detector(human_files_short[0]))
### TODO: Test the performance of the dog_detector function

### on the images in human_files_short and dog_files_short.

def dog_detect(files):

    detect_count = 0

    total_count = len(files)

    for dog in files:

        if dog_detector(dog):

            detect_count += 1

    return (detect_count, total_count)
print("Dogs in Human Files : {} / {}".format(dog_detect(human_files_short)[0],len(human_files_short)))

print("Dogs in Dog Files : {} / {}".format(dog_detect(dog_files_short)[0],len(dog_files_short)))
### (Optional) 

### TODO: Report the performance of another pre-trained network.

### Feel free to use as many code cells as needed.
import os

from torchvision import datasets

import torchvision.transforms as transforms



### TODO: Write data loaders for training, validation, and test sets

## Specify appropriate transforms, and batch_sizes



batch_size = 20

num_workers = 0

train_dir = "../input/dogclassification/dogimages/dogImages/train"

valid_dir = "../input/dogclassification/dogimages/dogImages/valid"

test_dir = "../input/dogclassification/dogimages/dogImages/test"







standard_normalization = transforms.Normalize(mean=[0.485,0.456,0.406],

                                              std=[0.299,0.244,0.255])

data_transforms = {'train': transforms.Compose([transforms.RandomResizedCrop(224),

                                     transforms.RandomHorizontalFlip(),

                                     transforms.ToTensor(),

                                     standard_normalization]),

                   'val': transforms.Compose([transforms.Resize(256),

                                     transforms.CenterCrop(224),

                                     transforms.ToTensor(),

                                     standard_normalization]),

                   'test': transforms.Compose([transforms.Resize(size=(224,224)),

                                     transforms.ToTensor(), 

                                     standard_normalization])

                  }
train_data = datasets.ImageFolder(train_dir, transform=data_transforms['train'])

valid_data = datasets.ImageFolder(valid_dir, transform=data_transforms['val'])

test_data = datasets.ImageFolder(test_dir, transform=data_transforms['test'])
train_loader = torch.utils.data.DataLoader(train_data,

                                           batch_size=batch_size, 

                                           num_workers=num_workers,

                                           shuffle=True)

valid_loader = torch.utils.data.DataLoader(valid_data,

                                           batch_size=batch_size, 

                                           num_workers=num_workers,

                                           shuffle=False)

test_loader = torch.utils.data.DataLoader(test_data,

                                           batch_size=batch_size, 

                                           num_workers=num_workers,

                                           shuffle=False)

loaders_scratch = {

    'train': train_loader,

    'valid': valid_loader,

    'test': test_loader

}
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True



num_classes = 133 # total classes of dog breeds
import torch.nn as nn

import torch.nn.functional as F

import numpy as np



# define the CNN architecture

class Net(nn.Module):

    ### TODO: choose an architecture, and complete the class

    def __init__(self):

        super(Net, self).__init__()

        ## Define layers of a CNN

        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)



        # pool

        self.pool = nn.MaxPool2d(2, 2)

        

        # fully-connected

        self.fc1 = nn.Linear(7*7*128, 500)

        self.fc2 = nn.Linear(500, num_classes) 

        

        # drop-out

        self.dropout = nn.Dropout(0.3)

    

    def forward(self, x):

        ## Define forward behavior

        x = F.relu(self.conv1(x))

        x = self.pool(x)

        x = F.relu(self.conv2(x))

        x = self.pool(x)

        x = F.relu(self.conv3(x))

        x = self.pool(x)

        

        # flatten

        x = x.view(-1, 7*7*128)

        

        x = self.dropout(x)

        x = F.relu(self.fc1(x))

        

        x = self.dropout(x)

        x = self.fc2(x)

        return x



#-#-# You so NOT have to modify the code below this line. #-#-#



# instantiate the CNN

model_scratch = Net()

print(model_scratch)



# move tensors to GPU if CUDA is available

if use_cuda:

    model_scratch.cuda()
import torch.optim as optim



### TODO: select loss function

criterion_scratch = nn.CrossEntropyLoss()



### TODO: select optimizer

optimizer_scratch = optim.SGD(model_scratch.parameters(), lr = 0.05)
def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path, last_validation_loss=None):

    """returns trained model"""

    # initialize tracker for minimum validation loss

    if last_validation_loss is not None:

        valid_loss_min = last_validation_loss

    else:

        valid_loss_min = np.Inf

    

    for epoch in range(1, n_epochs+1):

        # initialize variables to monitor training and validation loss

        train_loss = 0.0

        valid_loss = 0.0

        

        ###################

        # train the model #

        ###################

        model.train()

        for batch_idx, (data, target) in enumerate(loaders['train']):

            # move to GPU

            if use_cuda:

                data, target = data.cuda(), target.cuda()

            ## find the loss and update the model parameters accordingly

            ## record the average training loss, using something like

            ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))



            # initialize weights to zero

            optimizer.zero_grad()

            

            output = model(data)

            

            # calculate loss

            loss = criterion(output, target)

            

            # back prop

            loss.backward()

            

            # grad

            optimizer.step()

            

            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))

            

            if batch_idx % 100 == 0:

                print('Epoch %d, Batch %d loss: %.6f' %

                  (epoch, batch_idx + 1, train_loss))

            

        ######################    

        # validate the model #

        ######################

        model.eval()

        for batch_idx, (data, target) in enumerate(loaders['valid']):

            # move to GPU

            if use_cuda:

                data, target = data.cuda(), target.cuda()

            ## update the average validation loss

            output = model(data)

            loss = criterion(output, target)

            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))



            

        # print training/validation statistics 

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(

            epoch, 

            train_loss,

            valid_loss

            ))

        

        ## TODO: save the model if validation loss has decreased

        if valid_loss < valid_loss_min:

            torch.save(model.state_dict(), save_path)

            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(

            valid_loss_min,

            valid_loss))

            valid_loss_min = valid_loss

            

    # return trained model

    return model
model_scratch = train(100, loaders_scratch, model_scratch, optimizer_scratch, 

                      criterion_scratch, use_cuda, 'model_scratch.pt')
#load the model that got the best validation accuracy

model_scratch.load_state_dict(torch.load('model_scratch.pt'))
def test(loaders, model, criterion, use_cuda):



    # monitor test loss and accuracy

    test_loss = 0.

    correct = 0.

    total = 0.



    for batch_idx, (data, target) in enumerate(loaders['test']):

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

test(loaders_scratch, model_scratch, criterion_scratch, use_cuda)
## TODO: Specify data loaders

loaders_transfer = loaders_scratch.copy()
import torchvision.models as models

import torch.nn as nn



## TODO: Specify model architecture 

model_transfer = models.resnet50(pretrained=True)



for param in model_transfer.parameters():

    param.requires_grad = False



model_transfer.fc = nn.Linear(2048, 133, bias=True)



fc_parameters = model_transfer.fc.parameters()



for param in fc_parameters:

    param.requires_grad = True

    

model_transfer
if use_cuda:

    model_transfer = model_transfer.cuda()
criterion_transfer = nn.CrossEntropyLoss()

optimizer_transfer = optim.SGD(model_transfer.fc.parameters(), lr=0.001)
# train the model

# train(n_epochs, loaders_transfer, model_transfer, optimizer_transfer, criterion_transfer, use_cuda, 'model_transfer.pt')



def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):

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

        for batch_idx, (data, target) in enumerate(loaders['train']):

            # move to GPU

            if use_cuda:

                data, target = data.cuda(), target.cuda()



            # initialize weights to zero

            optimizer.zero_grad()

            

            output = model(data)

            

            # calculate loss

            loss = criterion(output, target)

            

            # back prop

            loss.backward()

            

            # grad

            optimizer.step()

            

            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))

            

            if batch_idx % 100 == 0:

                print('Epoch %d, Batch %d loss: %.6f' %

                  (epoch, batch_idx + 1, train_loss))

        

        ######################    

        # validate the model #

        ######################

        model.eval()

        for batch_idx, (data, target) in enumerate(loaders['valid']):

            # move to GPU

            if use_cuda:

                data, target = data.cuda(), target.cuda()

            ## update the average validation loss

            output = model(data)

            loss = criterion(output, target)

            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))



            

        # print training/validation statistics 

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(

            epoch, 

            train_loss,

            valid_loss

            ))

        

        ## TODO: save the model if validation loss has decreased

        if valid_loss < valid_loss_min:

            torch.save(model.state_dict(), save_path)

            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(

            valid_loss_min,

            valid_loss))

            valid_loss_min = valid_loss

            

    # return trained model

    return model
train(20, loaders_transfer, model_transfer, optimizer_transfer, criterion_transfer, use_cuda, 'model_transfer.pt')
# load the model that got the best validation accuracy

model_transfer.load_state_dict(torch.load('model_transfer.pt'))
test(loaders_transfer, model_transfer, criterion_transfer, use_cuda)
### TODO: Write a function that takes a path to an image as input

### and returns the dog breed that is predicted by the model.



# list of class names by index, i.e. a name can be accessed like class_names[0]

class_names = [item[4:].replace("_", " ") for item in loaders_transfer['train'].dataset.classes]
loaders_transfer['train'].dataset.classes[:10]

class_names[:10]
from PIL import Image

import torchvision.transforms as transforms



def load_input_image(img_path):    

    image = Image.open(img_path).convert('RGB')

    prediction_transform = transforms.Compose([transforms.Resize(size=(224, 224)),

                                     transforms.ToTensor(), 

                                     standard_normalization])



    # discard the transparent, alpha channel (that's the :3) and add the batch dimension

    image = prediction_transform(image)[:3,:,:].unsqueeze(0)

    return image
def predict_breed_transfer(model, class_names, img_path):

    # load the image and return the predicted breed

    img = load_input_image(img_path)

    model = model.cpu()

    model.eval()

    idx = torch.argmax(model(img))

    return class_names[idx]
### TODO: Write your algorithm.

### Feel free to use as many code cells as needed.



def run_app(img_path):

    ## handle cases for a human face, dog, and neither

    img = Image.open(img_path)

    plt.imshow(img)

    plt.show()

    if dog_detector(img_path) is True:

        prediction = predict_breed_transfer(model_transfer, class_names, img_path)

        print("Dogs Detected!\nIt looks like a {0}".format(prediction))  

    elif face_detector(img_path) > 0:

        prediction = predict_breed_transfer(model_transfer, class_names, img_path)

        print("Hello, human!\nIf you were a dog..You may look like a {0}".format(prediction))

    else:

        print("Error! Can't detect anything..")
## TODO: Execute your algorithm from Step 6 on

## at least 6 images on your computer.

## Feel free to use as many code cells as needed.



## suggested code, below

for file in np.hstack((human_files[:3], dog_files[:3])):

    run_app(file)