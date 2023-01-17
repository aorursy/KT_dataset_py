# All the required imports



import pandas as pd

import numpy as np

import os

import torch

import torchvision

from torchvision import transforms, models

from torch.utils.data import Dataset

import matplotlib.pyplot as plt

from torch import nn

import torch.nn.functional as F

from torch import optim

from skimage import io, transform

from torch.utils.data.sampler import SubsetRandomSampler





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

transform = transforms.Compose([transforms.RandomResizedCrop(224),

                                torchvision.transforms.RandomHorizontalFlip(p=0.5),

                                torchvision.transforms.RandomRotation(degrees = 15),

                                transforms.ToTensor()

                                ])



transform_t = transforms.Compose([transforms.RandomResizedCrop(224),

                                transforms.ToTensor()

                                ])



trainset = ImageDataset(csv_file = '../input/train.csv', root_dir = '../input/data/data/', transform=transform)  #Training Dataset
#Checking training sample size and label

for i in range(len(trainset)):

    sample = trainset[i]

    print(i, sample[0].size(), " | Label: ", sample[1])

    if i == 9:

        break
valid_size = 0.1

batch_size = 16

num_train = len(trainset)

indices = list(range(num_train))

np.random.shuffle(indices)

split = int(np.floor((valid_size)* num_train))

train_idx , valid_idx = indices[split:], indices[:split]



train_sampler = SubsetRandomSampler(train_idx)

valid_sampler = SubsetRandomSampler(valid_idx)



trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=train_sampler)

validloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=valid_sampler)
print(len(validloader))

print(len(trainloader))
# Visualizing some sample data

# obtain one batch of training images

dataiter = iter(trainloader)

images, labels = dataiter.next()

images = images.numpy() # convert images to numpy for display



# plot the images in the batch, along with the corresponding labels

fig = plt.figure(figsize=(25, 4))

for idx in np.arange(16):                                             #Change the range according to your batch-size

    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])

    plt.imshow(np.transpose(images[idx], (1, 2, 0)))
# check if CUDA / GPU is available, if unavaiable then turn it on from the right side panel under SETTINGS, also turn on the Internet

train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:

    print('CUDA is not available.  Training on CPU ...')

else:

    print('CUDA is available!  Training on GPU ...')
#creating resnet model

your_model = models.resnet50(pretrained=True)
your_model
fc = nn.Sequential(

    nn.Linear(2048, 1024),

    nn.ReLU(),

    nn.Dropout(0.25),

    nn.Linear(1024, 512),

    nn.ReLU(),

    nn.Dropout(0.25),

    nn.Linear(512, 67),

    nn.LogSoftmax(dim=1)

)



your_model.fc = fc
your_model.cuda()

# Loss function to be used

criterion = nn.NLLLoss()   # You can change this if needed



# Optimizer to be used, replace "your_model" with the name of your model and enter your learning rate in lr=0.001

optimizer = optim.Adadelta(your_model.parameters(), lr = 0.01)
# Training Loop (You can write your own loop from scratch)

n_epochs = 5 #number of epochs, change this accordingly

valid_loss_min = 1.18 #model has been trained for 25 epochs already



for e in range(n_epochs):

    

    train_loss = 0.0

    your_model.train()



    for images, labels in trainloader:

        #images, labels = Variable(images), Variable(labels)

        images, labels = images.cuda(), labels.cuda()

        optimizer.zero_grad()

        output = your_model(images)

        loss = criterion(output, labels)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        

    your_model.eval()

    valid_loss = 0.0

    valid_total = 0.0

    valid_correct = 0.0

    

    with torch.no_grad():

        for images, labels in validloader:

            images, labels = images.cuda(), labels.cuda()

            output = your_model(images)

            _, pred = torch.max(output.data, 1)

            loss = criterion(output, labels)

            valid_loss += loss.item()

            valid_total += labels.size(0)

            valid_correct += (pred == labels).sum()



    train_loss = train_loss/len(trainloader)

    valid_loss = valid_loss/len(validloader)

    accuracy = float(valid_correct)/float(valid_total)



    if valid_loss <= valid_loss_min:

        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(

        valid_loss_min,

        valid_loss))

        torch.save(your_model.state_dict(), 'indoor_scene_model.pt')

        valid_loss_min = valid_loss

    

    print("Epoch : {}/{}\tTrain loss : {}\tValid Loss : {}\tAccuracy : {}".format(e+1, n_epochs, train_loss, valid_loss, accuracy))
your_model.load_state_dict(torch.load("indoor_scene_model.pt"))


#Exit training mode and set model to evaluation mode

your_model.eval() # eval mode

# Reading sample_submission file to get the test image names

submission = pd.read_csv('../input/sample_sub.csv')

submission.head()
#Loading test data to make predictions



testset = ImageDataset(csv_file = '../input/sample_sub.csv', root_dir = '../input/data/data/', transform=transform_t)

testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=0)
predictions = []

# iterate over test data to make predictions

for data, target in testloader:

    # move tensors to GPU if CUDA is available

    

    if train_on_gpu:

        data, target = data.cuda(), target.cuda()

    # forward pass: compute predicted outputs by passing inputs to the model

    output = your_model(data)

    _, pred = torch.max(output, 1)

    for i in range(len(pred)):

        predictions.append(int(pred[i]))

        



submission['Category'] = predictions             #Attaching predictions to submission file

 


#saving submission file

submission.to_csv('submission.csv', index=False, encoding='utf-8')

import os

os.environ['KAGGLE_USERNAME'] = 'sham21'

os.environ['KAGGLE_KEY'] = 'd7a5e3643925c9acbdf13b8b3464e0e3'
!pip install kaggle
!kaggle competitions submit -c qstp-deep-learning-2019 -f submission.csv -m "resnet50_ft_val"