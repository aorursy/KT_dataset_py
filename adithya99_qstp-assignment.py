# All the required imports



import pandas as pd

import numpy as np

import os

import torch

import torchvision

from torchvision import transforms

from torch.utils.data import Dataset

import matplotlib.pyplot as plt

from torch import nn

import torch.nn.functional as F

from torch import optim

from skimage import io, transform



from PIL import Image



%matplotlib inline 
# Exploring train.csv file

df = pd.read_csv('../input/train.csv')

df.head()
#Dataset class



class ImageDataset(Dataset):

    



    def __init__(self, csv_file, root_dir, transform=None):

        """

        Args:

            csv_file (string): Path to the csv file with labels.

            root_dir (string): Directory with all the images.

            transform (callable, optional): Optional transform to be applied

                on a sample.

        """

        self.data_frame = pd.read_csv(csv_file)

        self.root_dir = root_dir

        self.transform = transform



    def __len__(self):

        return len(self.data_frame)



    def __getitem__(self, idx):

        img_name = os.path.join(self.root_dir, self.data_frame['Id'][idx])         # getting path of image

        image = Image.open(img_name).convert('RGB')                                # reading image and converting to rgb if it is grayscale

        label = np.array(self.data_frame['Category'][idx])                         # reading label of the image

        

        if self.transform:            

            image = self.transform(image)                                          # applying transforms, if any

        

        sample = (image, label)        

        return sample
# Transforms to be applied to each image (you can add more transforms), resizing every image to 3 x 224 x 224 size and converting to Tensor

transform =transforms.Compose([

         transforms.RandomResizedCrop(size=256,scale=(0.8,1.0)),

           transforms.RandomRotation(degrees=15),

        transforms.RandomHorizontalFlip(),

        transforms.ColorJitter(),

        transforms.CenterCrop(size=224),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ])



trainset = ImageDataset(csv_file = '../input/train.csv', root_dir = '../input/data/data/', transform=transform)     #Training Dataset

trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=0)                     #Train loader, can change the batch_size to your own choice
#Checking training sample size and label

for i in range(len(trainset)):

    sample = trainset[i]

    print(i, sample[0].size(), " | Label: ", sample[1])

    if i == 9:

        break
# Visualizing some sample data

# obtain one batch of training images

dataiter = iter(trainloader)

images, labels = dataiter.next()

images = images.numpy() # convert images to numpy for display



# plot the images in the batch, along with the corresponding labels

fig = plt.figure(figsize=(25, 4))

for idx in np.arange(16):                       #Change the range according to your batch-size

    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])

    plt.imshow(np.transpose(images[idx], (1, 2, 0)))
# check if CUDA / GPU is available, if unavaiable then turn it on from the right side panel under SETTINGS, also turn on the Internet

train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:

    print('CUDA is not available.  Training on CPU ...')

else:

    print('CUDA is available!  Training on GPU ...')
from torchvision import models

model = models.resnet152(pretrained=True)



for param in model.parameters():

    param.requires_grad = False

num_ftrs = model.fc.in_features

print(num_ftrs)

fc = nn.Sequential(

    nn.Linear(num_ftrs, 1024),

    nn.LogSoftmax(dim=1),

    nn.Dropout(0.1),

    nn.Linear(1024,67),

    nn.LogSoftmax(dim=1)

    

)

#num_ftrs = model.fc.in_features

model.fc = fc
 #...



# Define your CNN model here, shift it to GPU if needed

#class ImageClassifierModel(nn.Module):

  #  def __init__(self):

        

        

        

 #   def forward(self, x):





#your_model = ImageClassifierModel

#if train_on_gpu:

 #   your_model.cuda()

    

  #...   

if train_on_gpu:

    model.cuda()



# Loss function to be used

criterion = nn.CrossEntropyLoss()   # You can change this if needed



# Optimizer to be used, replace "your_model" with the name of your model and enter your learning rate in lr=0.001

optimizer = optim.SGD(model.parameters(), lr=0.01)



'''



# Training Loop (You can write your own loop from scratch)

n_epochs = 10    #number of epochs, change this accordingly



for epoch in range(1, n_epochs+1):

  

    # WRITE YOUR TRAINING LOOP HERE

    

'''   
#Loading test data to make predictions

transform1= transforms.Compose([

        transforms.RandomResizedCrop(size=256, scale=(0.8,1.0)),

        transforms.CenterCrop(size=224),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ])

testset = ImageDataset(csv_file = '../input/sample_sub.csv', root_dir = '../input/data/data/', transform=transform1)

testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=0)
num_epochs=20

for epoch in range(num_epochs):

    for i, (images, labels) in enumerate(trainloader):

           if torch.cuda.is_available():

                images = images.cuda()

                labels = labels.cuda()

        

        # Clear gradients w.r.t. parameters

           optimizer.zero_grad()

        

        # Forward pass to get output/logits

           outputs = model(images)

        

        # Calculate Loss: softmax --> cross entropy loss

           loss = criterion(outputs, labels)

        

        # Getting gradients w.r.t. parameters

           loss.backward()

        

        # Updating parameters

           optimizer.step()

    correct = 0

    total = 0

    # Iterate through test dataset

    for images, labels in trainloader:

        if torch.cuda.is_available():

            images = images.cuda()

            labels = labels.cuda()

                

                

                # Forward pass only to get logits/output

        outputs = model(images)

               

                # Get predictions from the maximum value

        _, predicted = torch.max(outputs.data, 1)

                

                # Total number of labels

        total += labels.size(0)

                

                # Total correct predictions

        correct += (predicted == labels).sum()

                

    accuracy = 100 * correct / total

            

            # Print Loss

    print('Iteration: {}. Loss: {}. Accuracy: {}. Correct: {}'.format(epoch, loss.item(), accuracy, correct))

    
# Reading sample_submission file to get the test image names

submission = pd.read_csv('../input/sample_sub.csv')

submission.head()




#Exit training mode and set model to evaluation mode

model.eval() # eval mode
predictions=[]

# iterate over test data to make predictions

for data, target in testloader:

    # move tensors to GPU if CUDA is available

    

    if train_on_gpu:

        data, target = data.cuda(), target.cuda()

    # forward pass: compute predicted outputs by passing inputs to the model

    output = model(data)

    _, pred = torch.max(output, 1)

    for i in range(len(pred)):

        predictions.append(int(pred[i]))

        



submission['Category'] = predictions             #Attaching predictions to submission file



      


#saving submission file

submission.to_csv('submission.csv', index=False, encoding='utf-8')