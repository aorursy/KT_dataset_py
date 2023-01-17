# All the required imports



import pandas as pd

import numpy as np

import os

import torch

import torchvision

from torchvision import transforms

from torchvision import models

from torch.utils.data import Dataset

import matplotlib.pyplot as plt

from torch import nn

import torch.nn.functional as F

from torch import optim

from skimage import io, transform

from torch.autograd import Variable



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

                                transforms.RandomHorizontalFlip(),

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

for idx in np.arange(16):                                             #Change the range according to your batch-size

    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])

    plt.imshow(np.transpose(images[idx], (1, 2, 0)))
# check if CUDA / GPU is available, if unavaiable then turn it on from the right side panel under SETTINGS, also turn on the Internet

train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:

    print('CUDA is not available.  Training on CPU ...')

else:

    print('CUDA is available!  Training on GPU ...')
model = models.densenet121(pretrained=True)

for param in model.parameters():

    param.requires_grad = False

n_inputs = model.classifier.in_features



#in_ftrs = model.classifier[1].in_channels

#out_ftrs = model.classifier[1].out_channels



'''# Add on classifier

model.classifier[6] = nn.Sequential(

    nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.4),

    nn.Linear(256, 67), nn.LogSoftmax(dim=1))



model.classifier'''





'''vgg16 = models.vgg16_bn()

#vgg16.load_state_dict(torch.load("../input/vgg16bn/vgg16_bn.pth"))

print(vgg16.classifier[6].out_features)''' 



#num_features = vgg16.classifier[3].in_channels

features = list(model.classifier.children())[:-1] 

features.extend([nn.Linear(n_inputs, 67)]) 

model.classifier = nn.Sequential(*features) # Replace the model classifier

#num_ftrs = model.AuxLogits.fc.in_features

#model.AuxLogits.fc = nn.Linear(num_ftrs, 67)

#model.AuxLogits.fc = nn.Linear(n_inputs, 67)

#model.fc = nn.Linear(n_inputs, 67)

#input_size = 299





#features = list(model.classifier.children())

#features[1] = nn.Conv2d(in_ftrs, 67, 3)

#features[3] = nn.AvgPool2d(12, stride=1)

#model.classifier = nn.Sequential(*features)

#model.num_classes = 67

print(model)





'''vgg16.classifier[6] = nn.Sequential(

                      nn.Linear(num_features, 256), 

                      nn.ReLU(), 

                      nn.Dropout(0.4),

                      nn.Linear(256,67),                   

                      nn.LogSoftmax(dim=1))'''







"""for param in your_model.parameters():

    param.requires_grad = False

your_model.classifier[6] = nn.Sequential(

                      nn.Linear(n_inputs, 256), 

                      nn.ReLU(), 

                      nn.Dropout(0.4),

                      nn.Linear(256, n_classes),                   

                      nn.LogSoftmax(dim=1))

# Define your CNN model here, shift it to GPU if needed

class ImageClassifierModel(nn.Module):

    def __init__(self):

        

        

        

    def forward(self, x):





your_model = ImageClassifierModel"""

if train_on_gpu:

        model.cuda()


criterion = nn.CrossEntropyLoss()   # You can change this if needed



# Optimizer to be used, replace "your_model" with the name of your model and enter your learning rate in lr=0.001

optimiser = optim.RMSprop(model.parameters(), lr=0.001, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)




# Training Loop (You can write your own loop from scratch)

#iter = 0

n_epochs = 20    #number of epochs, change this accordingly



for epoch in range(1, n_epochs+1):

    for i, (images,labels) in enumerate(trainloader):

    

    #images=images.unsqueeze(0)

        if train_on_gpu:

            images= Variable(images.cuda())

            labels= Variable(labels.cuda())

        else:

            images= Variable(images)

            labels= Variable(labels)



        #print(images.size())



        optimiser.zero_grad()



        outputs= model(images)



        loss= criterion(outputs,labels)



        loss.backward()



        optimiser.step()

        

        #iter+=1

    

        '''if iter%2000==0:



            correct= 0

            total= 0



            for images,labels in test_loader:

                images= Variable(images)

                outputs= model(images)



                _, predicted = torch.max(outputs.data, 1)

                total+= labels.size(0)



                correct+= (predicted==labels).sum()



                accuracy = 100*correct/total



                print ('Iteration: {}. Loss: {}. Accuracy: {}.'.format(iter,loss.data[0],accuracy))'''





    # WRITE YOUR TRAINING LOOP HERE

    

   




#Exit training mode and set model to evaluation mode

model.eval() # eval mode



# Reading sample_submission file to get the test image names

submission = pd.read_csv('../input/sample_sub.csv')

submission.head()
#Loading test data to make predictions



testset = ImageDataset(csv_file = '../input/sample_sub.csv', root_dir = '../input/data/data/', transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=0)
predictions = []

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

        #os.system('echo' + str(i) + ' ' + str(pred))



submission['Category'] = predictions             #Attaching predictions to submission file

     


#saving submission file

submission.to_csv('submission.csv', index=False, encoding='utf-8')
