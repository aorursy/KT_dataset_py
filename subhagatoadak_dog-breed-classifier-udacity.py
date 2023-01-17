

#import os

#for dirname in os.walk('/kaggle/input'):

#    print(dirname)
import numpy as np

from glob import glob



# load filenames for human and dog images

human_files = np.array(glob("/kaggle/input/dogudacity/dog/Dog/lfw/lfw/*/*"))

dog_files = np.array(glob("/kaggle/input/dogudacity/dog/Dog/dogImages/dogImages/*/*/*"))



# print number of images in each dataset

print('There are %d total human images.' % len(human_files))

print('There are %d total dog images.' % len(dog_files))

human_files[1]
import cv2                

import matplotlib.pyplot as plt  

%matplotlib inline                               



# extract pre-trained face detector

face_cascade = cv2.CascadeClassifier('/kaggle/input/cascade/cascade/Cascade/haarcascade_frontalface_alt.xml')



# load color (BGR) image

img = cv2.imread(human_files[0])

# convert BGR image to grayscale

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



# find faces in image

#faces = face_cascade.detectMultiScale(gray)



# Detect faces in the image

faces = face_cascade.detectMultiScale(

    gray,

    scaleFactor=1.2,

    minNeighbors=5,

    minSize=(30, 30)

)



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
#from tqdm import tqdm



human_files_short = human_files[:100]

dog_files_short = dog_files[:100]



face_dec = np.vectorize(face_detector)

#-#-# Do NOT modify the code above this line. #-#-#



## TODO: Test the performance of the face_detector algorithm 

## on the images in human_files_short and dog_files_short.

human_face_detected =face_dec(human_files_short)

dog_detected = face_dec(dog_files_short)



print(" Human Face Detected with {:.1f}% accuracy".format(sum(human_face_detected)))

print("Dog detected with  {:.1f}% error".format(sum(dog_detected)))
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
use_cuda = torch.cuda.is_available()
from PIL import Image

import torchvision.transforms as transforms

from torch.autograd import Variable

VGG16.eval()





def transfrom(image_path):

    img =Image.open(image_path)

    

    



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

    ## Normalize the data according to the needs of the models

    transform_img = transforms.Compose([

        transforms.Resize(size=(224,224)),

        transforms.ToTensor(),

        transforms.Normalize(mean=[0.485, 0.456, 0.406],

                                 std=[0.229, 0.224, 0.225])

        

    ])

    img = Image.open(img_path)

    img = transform_img(img)

    

    # PyTorch pretrained models expect the Tensor dims to be (num input imgs, num color channels, height, width).

    # Currently however, we have (num color channels, height, width); let's fix this by inserting a new axis.

    img = img.unsqueeze(0)  # Insert the new axis at index 0 i.e. in front of the other axes/dims. 



    # Now that we have preprocessed our img, we need to convert it into a 

    # Variable; PyTorch models expect inputs to be Variables. A PyTorch Variable is a  

    # wrapper around a PyTorch Tensor.

    img = Variable(img)

    

    prediction = VGG16(img.cuda())  # Returns a Tensor of shape (batch, num class labels)

    prediction = prediction.cpu().data.numpy().argmax()  # Our prediction will be the index of the class label with the largest value.

    

    return prediction # predicted class index
from PIL import Image

import glob

image = Image.open(human_files[0])

# summarize some details about the image

print(image.format)

print(image.mode)

print(image.size)

# show the image

image.show()
image.show()


# load and display an image with Matplotlib

from matplotlib import image

from matplotlib import pyplot

# load image as pixel array

data = image.imread(human_files[0])

# summarize shape of the pixel array

print(data.dtype)

print(data.shape)

# display the array of pixels as an image

pyplot.imshow(data)

pyplot.show()
### returns "True" if a dog is detected in the image stored at img_path

def dog_detector(img_path):

    ## TODO: Complete the function.

    ## TODO: Complete the function.

    pred=VGG16_predict(img_path)

    if pred >= 151 and pred <= 268:

        return True

    else:

        return False

    

    #return None # true/false
print(dog_detector(dog_files[0]))
from tqdm import tqdm

### TODO: Test the performance of the dog_detector function

### on the images in human_files_short and dog_files_short.

human_files_short = human_files[:100]

dog_files_short = dog_files[:100]



counter_human = 0

counter_dog   = 0



for human_file in tqdm(human_files_short):

    if(dog_detector(human_file)):

        counter_human +=1

for dog_file in tqdm(dog_files_short):

    if(dog_detector(dog_file)):

        counter_dog +=1

print("Detected dogs in human files ",counter_human,"%")

print("Detected dogs in dogs files ",counter_dog,"%")

### (Optional) 

### TODO: Report the performance of another pre-trained network.

### Feel free to use as many code cells as needed.
import os

import torch

from torchvision import datasets

import torchvision.transforms as transforms



### TODO: Write data loaders for training, validation, and test sets

## Specify appropriate transforms, and batch_sizes

data_dir = "/kaggle/input/dogudacity/dog/Dog/dogImages/dogImages/"

num_workers = 0

batch_size = 10

data_transforms = {

    'train' : transforms.Compose([

    transforms.Resize(256),

    transforms.RandomResizedCrop(224),

    transforms.RandomHorizontalFlip(), # randomly flip and rotate

    transforms.RandomRotation(15),

    transforms.ToTensor(),

    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    ]),

    # no need of image augmentation for the validation test set    

    'valid' : transforms.Compose([

    transforms.Resize(256),

    transforms.CenterCrop(224),

    transforms.ToTensor(),

    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    ]),

    # test dataset flips can  be found out in fast.ai

    'test' : transforms.Compose([

    transforms.Resize(256),

    transforms.CenterCrop(224),

    transforms.ToTensor(),

    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    ]),

}





train_dir = data_dir + '/train'

valid_dir = data_dir + '/valid'

test_dir = data_dir + '/test'



image_datasets = {

    'train' : datasets.ImageFolder(root=train_dir,transform=data_transforms['train']),

    'valid' : datasets.ImageFolder(root=valid_dir,transform=data_transforms['valid']),

    'test' : datasets.ImageFolder(root=test_dir,transform=data_transforms['test'])

}



# Loading Dataset

loaders_scratch = {

    'train' : torch.utils.data.DataLoader(image_datasets['train'],batch_size = batch_size,shuffle=True),

    'valid' : torch.utils.data.DataLoader(image_datasets['valid'],batch_size = batch_size),

    'test' : torch.utils.data.DataLoader(image_datasets['test'],batch_size = batch_size)    

}





print(loaders_scratch['train'])
import torch.nn as nn

import torch.nn.functional as F



# define the CNN architecture

class Net(nn.Module):

    ### TODO: choose an architecture, and complete the class

    def __init__(self):

        super(Net, self).__init__()

        ## Follow the architecture of VGG-16

        self.conv1 = nn.Conv2d(3, 16, 3)

        self.conv2 = nn.Conv2d(16, 32, 3)

        self.conv3 = nn.Conv2d(32, 64, 3)

        self.conv4 = nn.Conv2d(64, 128, 3)

        self.conv5 = nn.Conv2d(128, 256, 3)

        

        self.fc1 = nn.Linear(256 * 6 * 6, 133)

        self.max_pool = nn.MaxPool2d(2, 2,ceil_mode=True)

        

        self.dropout = nn.Dropout(0.20)

        

        self.conv_bn1 = nn.BatchNorm2d(224,3)

        self.conv_bn2 = nn.BatchNorm2d(16)

        self.conv_bn3 = nn.BatchNorm2d(32)

        self.conv_bn4 = nn.BatchNorm2d(64)

        self.conv_bn5 = nn.BatchNorm2d(128)

        self.conv_bn6 = nn.BatchNorm2d(256)

        

    def forward(self, x):

        ## Define forward behavior

        x = F.relu(self.conv1(x))

        x = self.max_pool(x)

        x = self.conv_bn2(x)



        x = F.relu(self.conv2(x))

        x = self.max_pool(x)

        x = self.conv_bn3(x)



        x = F.relu(self.conv3(x))

        x = self.max_pool(x)

        x = self.conv_bn4(x)



        x = F.relu(self.conv4(x))

        x = self.max_pool(x)

        x = self.conv_bn5(x)



        x = F.relu(self.conv5(x))

        x = self.max_pool(x)

        x = self.conv_bn6(x)



        x = x.view(-1, 256 * 6 * 6)



        x = self.dropout(x)

        x = self.fc1(x)

        return x



#-#-# You so NOT have to modify the code below this line. #-#-#



# instantiate the CNN

model_scratch = Net()



# move tensors to GPU if CUDA is available

if use_cuda:

    model_scratch.cuda()

model_scratch
import torch.optim as optim



### TODO: select loss function

criterion_scratch = nn.CrossEntropyLoss()



### TODO: select optimizer

optimizer_scratch = optim.SGD(model_scratch.parameters(), lr=0.005)
import numpy as np

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True



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

            ## find the loss and update the model parameters accordingly

            ## record the average training loss, using something like

            ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))

            optimizer.zero_grad()

            # forward pass

            output = model(data)

            # Loss

            loss = criterion(output, target)

            # backward pass

            loss.backward()

            # Optimization

            optimizer.step()

            # update training loss

            # train_loss += loss.item()*data.size(0)

            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))



            

        ######################    

        # validate the model #

        ######################

        model.eval()

        for batch_idx, (data, target) in enumerate(loaders['valid']):

            # move to GPU

            if use_cuda:

                data, target = data.cuda(), target.cuda()

          

            output = model(data)

            loss = criterion(output, target)

            # update average validation loss 

            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))

            

                

            # calculate average losses

        train_loss = train_loss/len(loaders['train'].dataset)

        valid_loss = valid_loss/len(loaders['valid'].dataset)

            



            

        # print training/validation statistics 

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(

            epoch, 

            train_loss,

            valid_loss

            ))

        

        ## TODO: save the model if validation loss has decreased

        if valid_loss <= valid_loss_min:

            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(

            valid_loss_min,

            valid_loss))

            torch.save(model.state_dict(), save_path)

            valid_loss_min = valid_loss  

            

    # return trained model

    return model





# train the model

model_scratch = train(20, loaders_scratch, model_scratch, optimizer_scratch, 

                      criterion_scratch, use_cuda, 'model_scratch.pt')



# load the model that got the best validation accuracy

model_scratch.load_state_dict(torch.load('model_scratch.pt'))
def test(loaders, model, criterion, use_cuda):



    # monitor test loss and accuracy

    test_loss = 0.

    correct = 0.

    total = 0.



    model.eval()

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

from torchvision import datasets

import torchvision.transforms as transforms



data_dir = "/kaggle/input/dogudacity/dog/Dog/dogImages/dogImages/"

num_workers = 0

batch_size = 10

data_transforms = {

    'train' : transforms.Compose([

    transforms.Resize(256),

    transforms.RandomResizedCrop(224),

    transforms.RandomHorizontalFlip(), # randomly flip and rotate

    transforms.RandomRotation(15),

    transforms.ToTensor(),

    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    ]),

    # no need of image augmentation for the validation test set    

    'valid' : transforms.Compose([

    transforms.Resize(256),

    transforms.CenterCrop(224),

    transforms.ToTensor(),

    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    ]),

    # test dataset flips can  be found out in fast.ai

    'test' : transforms.Compose([

    transforms.Resize(256),

    transforms.CenterCrop(224),

    transforms.ToTensor(),

    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    ]),

}





train_dir = data_dir + '/train'

valid_dir = data_dir + '/valid'

test_dir = data_dir + '/test'



image_datasets = {

    'train' : datasets.ImageFolder(root=train_dir,transform=data_transforms['train']),

    'valid' : datasets.ImageFolder(root=valid_dir,transform=data_transforms['valid']),

    'test' : datasets.ImageFolder(root=test_dir,transform=data_transforms['test'])

}

class_names = image_datasets['train'].classes

# Loading Dataset

loaders_transfer= {

    'train' : torch.utils.data.DataLoader(image_datasets['train'],batch_size = batch_size,shuffle=True),

    'valid' : torch.utils.data.DataLoader(image_datasets['valid'],batch_size = batch_size),

    'test' : torch.utils.data.DataLoader(image_datasets['test'],batch_size = batch_size)    

}





import torchvision.utils as torchutil

import matplotlib.pyplot as plt

plt.ion()

def imshow(inp, title=None):

    """Imshow for Tensor."""

    inp = inp.numpy().transpose((1, 2, 0))

    mean = np.array([0.485, 0.456, 0.406])

    std = np.array([0.229, 0.224, 0.225])

    inp = std * inp + mean

    inp = np.clip(inp, 0, 1)

    fig = plt.figure(figsize=(50, 25))

    plt.imshow(inp)

    if title is not None:

        plt.title(title)

    plt.pause(0.001)  # pause a bit so that plots are updated





# Get a batch of training data

inputs, classes = next(iter(loaders_transfer['train']))



# Make a grid from batch

out = torchutil.make_grid(inputs)



imshow(out, title=[class_names[x] for x in classes])
import torchvision.models as models

import torch.nn as nn



## TODO: Specify model architecture 

model_transfer = models.vgg16(pretrained=True)

# Freeze the pre-trained weights

for param in model_transfer.features.parameters():

    param.required_grad = False

    

# Get the input of the last layer of VGG-16

n_inputs = model_transfer.classifier[6].in_features



# Create a new layer(n_inputs -> 133)

# The new layer's requires_grad will be automatically True.

last_layer = nn.Linear(n_inputs, 133)



# Change the last layer to the new layer.

model_transfer.classifier[6] = last_layer



# Print the model.

print(model_transfer)







if use_cuda:

    model_transfer = model_transfer.cuda()
criterion_transfer = nn.CrossEntropyLoss()

optimizer_transfer = optim.SGD(model_transfer.classifier.parameters(),lr=0.001,momentum=0.9)
import numpy as np

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# train the model

n_epochs = 5





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

            ## find the loss and update the model parameters accordingly

            ## record the average training loss, using something like

            ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))

            optimizer.zero_grad()

            # forward pass

            output = model(data)

            # Loss

            loss = criterion(output, target)

            # backward pass

            loss.backward()

            # Optimization

            optimizer.step()

            # update training loss

            # train_loss += loss.item()*data.size(0)

            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))



            

        ######################    

        # validate the model #

        ######################

        model.eval()

        for batch_idx, (data, target) in enumerate(loaders['valid']):

            # move to GPU

            if use_cuda:

                data, target = data.cuda(), target.cuda()

          

            output = model(data)

            loss = criterion(output, target)

            # update average validation loss 

            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))

            

                

            # calculate average losses

        train_loss = train_loss/len(loaders['train'].dataset)

        valid_loss = valid_loss/len(loaders['valid'].dataset)

            



            

        # print training/validation statistics 

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(

            epoch, 

            train_loss,

            valid_loss

            ))

        

        ## TODO: save the model if validation loss has decreased

        if valid_loss <= valid_loss_min:

            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(

            valid_loss_min,

            valid_loss))

            torch.save(model.state_dict(), save_path)

            valid_loss_min = valid_loss  

            

    # return trained model

    return model





model_transfer = train(n_epochs, loaders_transfer, model_transfer, optimizer_transfer, criterion_transfer, use_cuda, 'model_transfer.pt')



# load the model that got the best validation accuracy (uncomment the line below)

model_transfer.load_state_dict(torch.load('model_transfer.pt'))
test(loaders_transfer, model_transfer, criterion_transfer, use_cuda)
### TODO: Write a function that takes a path to an image as input

### and returns the dog breed that is predicted by the model.



# list of class names by index, i.e. a name can be accessed like class_names[0]

class_names = [item[4:].replace("_", " ") for item in image_datasets['train'].classes]

# Load the trained model 'model_transfer.pt'

model_transfer.load_state_dict(torch.load('model_transfer.pt', map_location='cpu'))

def predict_breed_transfer(img_path):

    # load the image and return the predicted breed

    image = Image.open(img_path).convert('RGB')

    prediction_transform = transforms.Compose([transforms.Resize(size=(224, 224)),

                                     transforms.ToTensor(), 

                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])



    # discard the transparent, alpha channel (that's the :3) and add the batch dimension

    image = prediction_transform(image)[:3,:,:].unsqueeze(0)

    image = image.cuda()

    

    

    model_transfer.eval()

    idx = torch.argmax(model_transfer(image))

    return class_names[idx]
### TODO: Write your algorithm.

### Feel free to use as many code cells as needed.

from PIL import Image

def run_app(img_path):

    ## handle cases for a human face, dog, and neither

    img = Image.open(img_path)

    plt.imshow(img)

    plt.show()

    

    if dog_detector(img_path) is True:

        prediction = predict_breed_transfer(img_path)

        print("A dog has been detected which most likely to be {0} breed".format(prediction))  

    elif face_detector(img_path) > 0:

        prediction = predict_breed_transfer(img_path)

        print("This is a Human who looks like a {0}".format(prediction))

    else:

        print("Neither Human nor Dog")

## TODO: Execute your algorithm from Step 6 on

## at least 6 images on your computer.

## Feel free to use as many code cells as needed.



## suggested code, below

for file in np.hstack((human_files[2], dog_files[2])):

    run_app(file)