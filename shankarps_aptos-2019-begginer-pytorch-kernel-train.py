import pandas as pd



import time

import torchvision

import torch.nn as nn

from tqdm import tqdm_notebook as tqdm



from PIL import Image, ImageFile

import matplotlib.pyplot as plt

%matplotlib inline

from torch.utils.data import Dataset

import torch

import torch.optim as optim

from torchvision import transforms

from torch.optim import lr_scheduler

import os

import numpy as np

device = torch.device("cuda:0")

ImageFile.LOAD_TRUNCATED_IMAGES = True

from torch.utils.data.sampler import SubsetRandomSampler

from collections import OrderedDict



print(os.listdir('../input'))


path_0 ="../input/aptos2019-blindness-detection/train_images/002c21358ce6.png"

img_0 = Image.open(path_0)

#plt.figure(figsize = (10,10))



path_1="../input/aptos2019-blindness-detection/train_images/0024cdab0c1e.png"

img_1 = Image.open(path_1)

print(type(img_1))

img_1 = transforms.functional.adjust_contrast(img_1, contrast_factor=3)

print(type(img_1))

img_1 = transforms.functional.adjust_gamma(img_1, gamma=3)

print(type(img_1))

#plt.figure(figsize = (10,10))



path_2="../input/aptos2019-blindness-detection/train_images/000c1434d8d7.png"

img_2 = Image.open(path_2)

#plt.figure(figsize = (10,10))



path_3="../input/aptos2019-blindness-detection/train_images/03c85870824c.png"

img_3 = Image.open(path_3)



path_4="../input/aptos2019-blindness-detection/train_images/02685f13cefd.png"

img_4 = Image.open(path_4)



#plt.figure(figsize = (10,10))

# Four polar axes

f, axarr = plt.subplots(1, 5, figsize=(20, 10) )

axarr[0].imshow(img_0)

axarr[0].set_title('0 - No DR')

axarr[1].imshow(img_1)

axarr[1].set_title('1 - Mild')

axarr[2].imshow(img_2)

axarr[2].set_title('2 - Moderate')

axarr[3].imshow(img_3)

axarr[3].set_title('3 - Severe')

axarr[4].imshow(img_4)

axarr[4].set_title('4 - Proliferative DR')



#Apply transformations



#f.subplots_adjust(hspace=10)
class RetinopathyDatasetTrain(Dataset):



    def __init__(self, csv_file, transform):



        self.data = pd.read_csv(csv_file)

        self.transform = transform

        self.labels = torch.eye(5)[self.data['diagnosis']]

        



    def __len__(self):

        return len(self.data)



    def __getitem__(self, idx):

        img_name = os.path.join('../input/aptos2019-blindness-detection/train_images', self.data.loc[idx, 'id_code'] + '.png')

        image = Image.open(img_name)

        image = transforms.functional.adjust_saturation(image, saturation_factor=0)

        image = transforms.functional.adjust_contrast(img=image, contrast_factor=2)

        image = transforms.functional.adjust_gamma(img=image, gamma=2)

        image = self.transform(image)

        label = torch.tensor(self.data.loc[idx, 'diagnosis'])

        #label = self.labels[idx]

        return {'image': image,

                'labels': label

                }

    

transformer = transforms.Compose([

    transforms.Resize((128, 128)),

    transforms.CenterCrop((64, 64)),

    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),

    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

])    



loadSavedModel = False

#model = torchvision.models.resnet101(pretrained=True)

model = []

if (not loadSavedModel):

    model = torchvision.models.vgg16_bn(pretrained=True)

    model = model.to(device)

    #Disable grad for the hidden layers

    for params in model.parameters():

        params.requires_grad = False

    #Add custom layer - after setting other layers no grad.

    custom_classifier = nn.Sequential(OrderedDict([

                          ('fc1',nn.Linear(25088, 4096)),

                          ('r1',nn.ReLU()),

                          ('fc2',nn.Linear(4096, 512)),

                          ('r2',nn.ReLU()),

                          ('fc3',nn.Linear(512, 5)),

                          ('s3',nn.LogSoftmax(dim=1)) ]               

                          ))

    model.classifier = custom_classifier

    

else:    

    model = torchvision.models.vgg16_bn(pretrained=False)

    saved_checkpoint = torch.load('saved_model.pth')

    #Create empty model

    #Add custom layer

    temp_model.classifier = nn.Sequential(OrderedDict([

                          ('fc1',nn.Linear(25088, 4096)),

                          ('r1',nn.ReLU()),

                          ('d1',nn.Dropout(p=0.3)),  

                          ('fc2',nn.Linear(4096, 512)),

                          ('r2',nn.ReLU()),

                          ('d2',nn.Dropout(p=0.3)),

                          ('fc3',nn.Linear(512, 5)),

                          ('s3',nn.LogSoftmax(dim=1)) ]               

                          ))

    model.classifier = custom_classifier

    #Load model

    temp_model.load_state_dict(saved_checkpoint['state_dict'])



#print(model)

params = list(model.parameters())

print(len(params))

print(params[0].size())  # conv1's .weight

print(params[1].size())  # conv1's .weight

print(params[2].size())  # conv1's .weight

print(params[4].size())  # conv1's .weight
train_dataset = RetinopathyDatasetTrain(csv_file='../input/aptos2019-blindness-detection/train.csv',transform=transformer)

#data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)



batch_size = 16

validation_split = .1

shuffle_dataset = True

random_seed= 42



# Creating data indices for training and validation splits:

dataset_size = len(train_dataset)

indices = list(range(dataset_size))

split = int(np.floor(validation_split * dataset_size))

np.random.seed(random_seed)

np.random.shuffle(indices)

train_indices, val_indices = indices[split:], indices[:split]



# Creating PT data samplers and loaders:

train_sampler = SubsetRandomSampler(train_indices)

valid_sampler = SubsetRandomSampler(val_indices)



train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 

                                           sampler=train_sampler, num_workers=4)

validation_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,

                                                sampler=valid_sampler, num_workers=4)



#TODO Split the data set into train and validate sets.

#https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets



im = next(iter(train_loader))

image = im["image"]

labels = im["labels"]

print(f"image shape {image.shape}")

print(f"labels shape {labels.shape}")

print(f"labels {labels}")



criterion = nn.NLLLoss()



optimizer = optim.Adam(model.classifier.parameters(), lr = 0.001)



debugTrain = False

debugVal = False

device = torch.device(device)

model = model.to(device)

#Train - number of epochs

epochs = 10

acc_loss = 0

for epoch in range(epochs):

    batch_count = 0

    tk0 = tqdm(train_loader, total=int(len(train_loader)))

    counter = 0

    for bi, d in enumerate(tk0):

        batch_count += 1

        #print(f"batch {batch_count}")

        #print(images.shape)

        inputs = d["image"]

        #labels = d["labels"].view(-1, 1)

        labels = d["labels"]

        inputs = inputs.to(device, dtype=torch.float)

        labels = labels.to(device, dtype=torch.float)

        labels_val = labels.long()

        optimizer.zero_grad()

             

        logps = model.forward(inputs)

        loss = criterion(logps, labels_val)

        if(debugTrain):

            #print(f"train - logps {logps.shape}")

            print(f"logps[0][0] {logps[0][0]} logps[0][1] {logps[0][1]}")

            #print(logps.type())

            #print(labels_val.type())    

            print(f" loss {loss}")

            

        loss.backward()

        optimizer.step()

        acc_loss += loss.item()

        

    #End of epoch - print loss and accuracy

    val_loss = 0

    accuracy = 0

    model.eval()

    with torch.no_grad():

        tk1 = tqdm(validation_loader, total=int(len(validation_loader)))

        for bi, d in enumerate(tk1):

            inputs = d["image"]

            #labels = d["labels"].view(-1, 1)

            labels = d["labels"]

            inputs = inputs.to(device, dtype=torch.float)

            labels = labels.to(device, dtype=torch.float)

            labels_val = labels.long()

            logps = model.forward(inputs)

            



            batch_loss = criterion(logps, labels_val)

            val_loss+= batch_loss.item()

            

            ps = torch.exp(logps)

            top_p, top_class = ps.topk(1, dim =1)

            if(debugVal):

                #print(logps.type())

                #print(labels_val.type())

                #print(f" val - labelabels_valls shape {labels_val.shape}")

                #print(f" val - logps shape {logps.shape}")

                print(f" top_class {top_class} labels {labels}")

                #print(top_class.type())

                #print(labels.type())

                #print(labels_val.type())

                #top_class = top_class.type(torch.DoubleTensor)

                

            equals = top_class == labels_val.view(-1, 1)

            #print(f"equals {equals}")

            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            

    #Set model to train again

    model.train()    

    #End of epoch - print loss and accuracy

    print(f"training loss {acc_loss/len(train_loader):.3f}")

    print(f"val loss {val_loss/len(validation_loader):.3f}")

    print(f"val accuracy {accuracy/len(validation_loader):.3f}")



    acc_loss =0

    

    
#Save model for loading without internet

checkpoint ={"state_dict":model.state_dict()}

torch.save(checkpoint, 'saved_model_vgg16_bn_aptos_competition.pth')





class RetinopathyDatasetTest(Dataset):



    def __init__(self, csv_file, transform):

        self.data = pd.read_csv(csv_file)

        self.transform = transform



    def __len__(self):

        return len(self.data)



    def __getitem__(self, idx):

        img_name = os.path.join('../input/aptos2019-blindness-detection/test_images', self.data.loc[idx, 'id_code'] + '.png')

        image = Image.open(img_name)

        image = self.transform(image)

        return {'image': image}
test_dataset = RetinopathyDatasetTest(csv_file='../input/aptos2019-blindness-detection/test.csv', transform = transformer)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

test_preds = np.zeros((len(test_dataset), 1))

model.eval()

with torch.no_grad():

    tk2 = tqdm(test_loader, total=int(len(test_loader)))

    for i, d in enumerate(tk2):

         #print(bi)

         inputs = d["image"]

         inputs = inputs.to(device, dtype=torch.float)

         #print(f" Test inputs shape {inputs.shape}")

         #print(f" Test labels shape {labels.shape}")

         logps = model.forward(inputs)

         print(f"logps {logps}")

         ps = torch.exp(logps)

         top_p, top_class = ps.topk(1, dim =1)

         #print(logps.shape)   

         print(f" top_class {top_class} top_p {top_p}")

         #print(top_class.type())

         top_class = top_class.type(torch.DoubleTensor)

         #print(f"top class {top_class}")

         print(top_class.type())

         #print(type(top_class.numpy().ravel()))

         print(top_class.detach().cpu().squeeze().numpy().ravel().reshape(-1, 1))

         test_preds[i * 16:(i + 1) * 16] = top_class.detach().cpu().squeeze().numpy().ravel().reshape(-1, 1)

         





sample = pd.read_csv("../input/aptos2019-blindness-detection/sample_submission.csv")

sample.diagnosis = test_preds.astype(int)

sample.to_csv("submission.csv", index=False)

#Check if the file present

os.listdir('.')