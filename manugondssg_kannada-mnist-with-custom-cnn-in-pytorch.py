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
import os

import pandas as pd

import numpy as np

from PIL import Image

import time

import errno

import torch

import torch.nn.functional as F

import matplotlib.pyplot as plt

from torchvision import models,datasets,transforms

from torch import nn,optim

from torch.utils.data.sampler import SubsetRandomSampler

from collections import OrderedDict
# Select GPU if available

device='cpu'

if(torch.cuda.is_available()):

    device='cuda'
extract_folder_location='/kaggle/input/Kannada-MNIST/'
df_train=pd.read_csv(extract_folder_location+'train.csv') #Train data csv file

df_test=pd.read_csv(extract_folder_location+'Dig-MNIST.csv') # Test data csv file
df_train.head(5)
df_train=df_train.to_numpy() #convert both dataframes to numpy array to iterate over each row seperately

df_test=df_test.to_numpy()
os.mkdir('myData')      

os.mkdir('myData/train')  # Create the necessary train folder

for i in range(10):

    os.mkdir('myData/train/'+str(i)) # since Image is divided into 10 classes, generate 10 distinct folder for each class images

    

os.mkdir('myData/test')

for i in range(10):

    os.mkdir('myData/test/'+str(i))
def generateImage(imageArray,imageWidth,imageHeight):

    '''

    Parameters :

        imageArray: numpy 1d array with pixel intensity values

        imageWidth: Width of target image

        imageHeight: Height of target image

    

    Returns: Generated grey scale PIL image with resolution Width x Height 

    '''

    image=np.zeros(shape=(imageWidth,imageHeight))

    index=0

    for i in range(imageWidth):

        for j in range(imageHeight):

            image[i][j]=imageArray[index]

            index+=1

    img=Image.fromarray(image)

    img=img.convert("L")

    return img
# Function to save PIL image to file

def saveImage(image,saveLocation):

    image.save(saveLocation)
# Function to iterate over whole train dataset, convert each row of pixel values into iamge and save the image into its corresponding class id folder

for i in range(len(df_train)):

    saveImage(generateImage(df_train[i][1:],28,28),'myData/train/'+str(df_train[i][0])+'/'+str(i)+'.png')
# Function to iterate over whole test dataset, convert each row of pixel values into iamge and save the image into its corresponding class id folder

for i in range(len(df_test)):

    saveImage(generateImage(df_test[i][1:],28,28),'myData/test/'+str(df_test[i][0])+'/'+str(i)+'.png')
data_dir='myData'  # Train and test data location directory

num_workers=0      # This is set to 0 becasue we are only using 1 hardware accelerator

batch_size=128     

valid_size=0.2     # Fraction of data that will be taken from training data for validation set





# Image transformations for greay scale image, to convert it into a tensor and normalize it

train_transforms = transforms.Compose([transforms.Grayscale(num_output_channels=1),

                                       transforms.ToTensor(),

                                       transforms.Normalize([0.5],

                                                            [0.5])])



test_transforms = transforms.Compose([transforms.Grayscale(num_output_channels=1),

                                      transforms.ToTensor(),

                                      transforms.Normalize([0.5],

                                                            [0.5])])



train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms) #Use PyTorch datasets.ImageFolder to load and transform images into tensor

test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)





# Split the train set into train and validation set and shuffle it

num_train = len(train_data)   

indices = list(range(num_train))

np.random.shuffle(indices)

split = int(np.floor(valid_size * num_train))

train_idx, valid_idx = indices[split:], indices[:split]



train_sampler = SubsetRandomSampler(train_idx)

valid_sampler = SubsetRandomSampler(valid_idx)





# Put the data into dataloader

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,

    sampler=train_sampler, num_workers=num_workers)

valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 

    sampler=valid_sampler, num_workers=num_workers)



test_loader = torch.utils.data.DataLoader(test_data,batch_size=batch_size,num_workers=num_workers)
len(train_loader),len(valid_loader),len(test_loader)
# Print the class to id mapping 

print(train_data.class_to_idx)
class KannadaClassifierCNN(torch.nn.Module):

    def __init__(self):

        super(KannadaClassifierCNN,self).__init__()

        

        self.conv1 = torch.nn.Conv2d(1,50,kernel_size=3,stride=1,padding=1)

        self.pool = torch.nn.MaxPool2d(kernel_size=2,stride=2,padding=0)

        self.fc1 = torch.nn.Linear(50*14*14,512)

        self.fc2 = torch.nn.Linear(512,256)

        self.fc3 = torch.nn.Linear(256,128)

        self.fc4 = torch.nn.Linear(128,64)

        self.fc5 = torch.nn.Linear(64,10)

        self.dropout = torch.nn.Dropout(0.25)

        

        # Function to make forward pass into CNN

    def forward(self,x):

        x = F.relu(self.conv1(x))

        x = F.relu(self.pool(x))

        x = F.relu(self.fc1(x.view(-1,50*14*14)))

        x = self.dropout(x)

        x = F.relu(self.fc2(x))

        x = self.dropout(x)

        x = F.relu(self.fc3(x))

        x = self.dropout(x)

        x = F.relu(self.fc4(x))

        x = self.fc5(x)

        return F.log_softmax(x,dim=1)
def createModel():

    model=KannadaClassifierCNN()

    

    # Unfreeze all parameters to allow gradient calculation for backpropagate

    for param in model.parameters():

        param.requires_grad=True

        

    # Loss and optimizer for model

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    return model,criterion,optimizer
def trainNetwork(model,epochs):

    # Passes model has be already on GPU if GPU was available

    train_on_gpu=False

    if device=='cuda':

        train_on_gpu=True

    n_epochs = epochs

    valid_loss_min = np.Inf  # Set initial validation loss to MAX 

    Training_Loss=[]

    Validation_Loss=[]

    Iteration=[]

    print('Train on gpu is : ',train_on_gpu)





    for epoch in range(1, n_epochs+1):



        train_loss = 0.0

        valid_loss = 0.0

        Iteration.append(epoch+1)



        model.train()

        for data, target in train_loader:



            if train_on_gpu:   # GPU is available then move data to GPU and train

                data, target = data.cuda(), target.cuda()



            optimizer.zero_grad()

            output = model(data)

            loss = criterion(output, target)

            loss.backward()

            optimizer.step()

            train_loss += loss.item()*data.size(0)





        model.eval()

        for data, target in valid_loader:



            if train_on_gpu:

                data, target = data.cuda(), target.cuda()



            output = model(data)

            loss = criterion(output, target)

            valid_loss += loss.item()*data.size(0)



        train_loss = train_loss/len(train_loader.dataset)

        valid_loss = valid_loss/len(valid_loader.dataset)

        Training_Loss.append(train_loss)

        Validation_Loss.append(valid_loss)





        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(

          epoch, train_loss, valid_loss))



        if valid_loss <= valid_loss_min: #Each time validation loss decreases save the model

            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(

          valid_loss_min,

          valid_loss))

            torch.save(model.state_dict(), 'checkpointFinal.pt')

            saveModel(model,'CheckpointFinal2.pth')

            valid_loss_min = valid_loss



    plt.plot(Iteration,Training_Loss)

    plt.plot(Iteration,Validation_Loss)

    plt.xlabel('Epoch')

    plt.ylabel('Loss')

    plt.ylim(0.0,1.0)

    plt.xlim(0,epochs)

    plt.legend(['Training Loss','Validation Loss'], loc='upper left')

    plt.show()

    return model

def saveModel(model,path):

    model.class_to_idx = train_data.class_to_idx

    checkpoint = {'input_size': 28*28,

                'output_size': 10,

                'model': model,

                'state_dict': model.state_dict(),

                'optimizer_state_dict': optimizer.state_dict,

                'criterion': criterion,

                'class_to_idx':model.class_to_idx

               }

    torch.save(checkpoint, path)
# Function to save model checkpoint

def load_checkpoint(filepath):

    checkpoint=torch.load(filepath)

    model=checkpoint["model"]

    model.class_to_idx = checkpoint['class_to_idx']

    model.load_state_dict(checkpoint['state_dict'],strict=False)

    return model
# Function to check accuracy by passing model and corresponding dataloader (it can be train_loader, validation_loader or test_loader)

def check_accuracy(model,testloader):    

    correct = 0

    total = 0

    model.to('cuda')

    with torch.no_grad():

        for data in testloader:

            images, labels = data

            images, labels = images.to('cuda'), labels.to('cuda')

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()



    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
# For single image prediction , image processing function

def process_image(img_pil):



    adjustments = transforms.Compose([transforms.ToTensor(),

        transforms.Normalize([0.5],[0.5])

    ])

    

    img_tensor = adjustments(img_pil)

    

    return img_tensor
# Prediction function to reutrn the top k prob for classes 

def predict(image_path, model, topk=1):   

    model.to(device)

    img_torch = process_image(image_path)

    img_torch = img_torch.unsqueeze_(0)

    img_torch = img_torch.float()

    

    with torch.no_grad():

        output = model.forward(img_torch.cuda())

        

    probability = F.softmax(output.data,dim=1)

    

    return probability.topk(topk)
model,criterion,optimizer=createModel()
model.cuda()
model=trainNetwork(model,10)
check_accuracy(model,test_loader)
model=trainNetwork(model,5)
check_accuracy(model,test_loader)
optimizer = optim.Adam(model.parameters(), lr=0.000001)
model=trainNetwork(model,45)
model=load_checkpoint('CheckpointFinal2.pth')
check_accuracy(model,test_loader)
df_submission=pd.read_csv(extract_folder_location+'sample_submission.csv')

df_submission_mapping=pd.read_csv(extract_folder_location+'test.csv')
df_submission.head()
df_submission_mapping.head()
df_submission_mapping=df_submission_mapping.to_numpy()
temp=[]

for i in range(len(df_submission_mapping)):

    image=generateImage(df_submission_mapping[i][1:],imageWidth=28,imageHeight=28)

    img_torch = process_image(image)

    img_torch = img_torch.unsqueeze_(0)

    img_torch = img_torch.float()

    outputs=model.forward(img_torch.cuda())

    _, predicted = torch.max(outputs.data, 1)

    temp.append(predicted)
result=[]

for element in temp:

    result.append(element.item())
result[:10]
df_submission['label']=result
df_submission.to_csv('submission.csv',index=False)
!rm -rf myData
!rm CheckpointFinal2.pth
!rm checkpointFinal.pt
!ls