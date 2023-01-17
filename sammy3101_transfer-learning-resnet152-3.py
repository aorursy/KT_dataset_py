# Importing necessary libraries



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



import os

import time

import copy



import torch

import torchvision

from torchvision import transforms, models

from torch import optim, nn

from torch.utils.data import Dataset, DataLoader

from torch.autograd import Variable



from PIL import Image



%matplotlib inline
df = pd.read_csv('../input/train.csv')

df.head()



train, validate = np.split(df, [int(.8*len(df))])

validate.index = range(len(validate))
# Making an image datset class



class ImageDataset(Dataset):

    def __init__(self, data, root_dir, transform=None):

        

        self.df = data

        self.dir = root_dir

        self.transform = transform

    

    def __len__(self):

        return len(self.df)

    

    def __getitem__(self, index):

        img_path = os.path.join(self.dir, self.df['Id'][index])

        image = Image.open(img_path).convert('RGB')

        label = np.array(self.df['Category'][index])

        

        if self.transform:

            image = self.transform(image)

        

        sample = (image, label)

        return sample

        
# Defining some transforms to standardise and augment the dataset



transform = transforms.Compose([transforms.Resize(256),

                                transforms.RandomRotation(24),

                                transforms.ColorJitter(brightness=0.8, contrast=0.2),

                                transforms.CenterCrop(224),

                                transforms.RandomHorizontalFlip(),

                                transforms.ToTensor(),

                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])



trainset = ImageDataset(data=train, root_dir='../input/data/data', transform=transform)

valset = ImageDataset(data=validate, root_dir='../input/data/data', transform=transform)



datasets = {'train' : trainset, 'val': valset}



dataloaders = {x : DataLoader(datasets[x], batch_size=32, num_workers=4, shuffle=True) 

               for x in ['train', 'val']}

# Check if the GPU is available



gpu = torch.cuda.is_available()



if gpu:

    print("CUDA is available. Training on GPU")

else:

    print("CUDA unavailable. Training on CPU")
model = models.resnet152(pretrained=True)

num_ft = model.fc.in_features



model.fc = nn.Sequential(nn.Linear(num_ft, 67),

                         nn.Dropout2d(p=0.2))



model.cuda()



criterion = nn.CrossEntropyLoss()



optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)



scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
def train_model(model, criterion, optimizer, scheduler, num_epochs=20):

    since = time.time()

    

    best_acc = 0.0

    best_model_state = copy.deepcopy(model.state_dict())

    

    for epoch in range(num_epochs):

        epoch += 1

        print("Epoch : {}/{}".format(epoch, num_epochs))

        print("-"*10)

        

        for phase in ['train', 'val']:

            if phase is 'train':

                scheduler.step()

                model.train()

            else:

                model.eval()

            

            running_loss = 0.0

            running_correct = 0

             

            for inputs, labels in dataloaders[phase]:

                if gpu:

                    inputs = Variable(inputs.cuda())

                    labels = Variable(labels.cuda())

                else:

                    inputs = Variable(inputs)

                    labels = Variable(labels)

                

                optimizer.zero_grad()

                

                with torch.set_grad_enabled(phase is 'train'):

                    outputs = model(inputs)

                    _, predicted = torch.max(outputs.data, 1)

                    loss = criterion(outputs, labels)

                    

                    if phase is 'train':

                        loss.backward()

                        optimizer.step()

                

                running_loss += loss.item() * inputs.size(0)

                running_correct += torch.sum(predicted.cpu() == labels.cpu())

            

            epoch_loss = running_loss / len(datasets[phase])

            epoch_acc = running_correct.double() / len(datasets[phase])

            

            print("{}:  Loss: {:.4f}, Accuracy: {:.4f}".format(phase, epoch_loss, epoch_acc))

            

            if phase == 'val' and epoch_acc > best_acc:

                best_acc = epoch_acc

                best_model_state = copy.deepcopy(model.state_dict())

                

        print()

        

    time_elapsed = time.time() - since

    print("Training finished in {} seconds!".format(time_elapsed))

    print("Best eval accuracy: {:.4f}".format(100*best_acc))

    

    model.load_state_dict(best_model_state)

    return model          
model = train_model(model, criterion, optimizer, scheduler, 25)
# Loading dataset for generating predictions



testset = ImageDataset(data=pd.read_csv('../input/sample_sub.csv'), root_dir='../input/data/data/', transform=transform)

testloader = DataLoader(testset, batch_size=32, num_workers=4, shuffle=False)
# Load the submission file



submission = pd.read_csv('../input/sample_sub.csv')

submission.head()
# Run the testing



predictions = []



for inputs, labels in testloader:

    if gpu:

        inputs = Variable(inputs.cuda())

    else:

        inputs = Variable(inputs)

    

    outputs = model(inputs)

    _, predicted = torch.max(outputs.data, 1)

    

    for i in range(len(predicted)):

        predictions.append(int(predicted[i]))

    

submission['Category'] = predictions
submission.to_csv('Submission5.csv', index=False, encoding='utf-8')