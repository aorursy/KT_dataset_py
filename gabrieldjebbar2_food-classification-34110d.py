# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


#Import Pytorch
import torch
import torchvision
from torchvision import  datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
#We need the display function from IPython for Notebook
from IPython.display import display
from torch.optim.lr_scheduler import ReduceLROnPlateau



# others
import numpy as np
import random
from PIL import Image
from IPython.display import display
#A package to make beautiful progress bars :) 
from tqdm import tqdm_notebook

import matplotlib.pyplot as plt

import pandas as pd
# library to do bash-like wildcard expansion
import glob

# a little helper function do directly display a Tensor
def display_tensor(t):
    trans = transforms.ToPILImage()
    display(trans(t))
    



import matplotlib.pyplot as plt
import time
import os
import copy
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    i = 0
    for filename in filenames:
        i = i + 1
        if i > 3: break
        #print(os.path.join(dirname, filename))


# Any results you write to the current directory are saved as output.
# custom datset loader

class FoodDataset(torch.utils.data.Dataset):    
    def __init__(self,img_dir, evaluation=False,augmentation=False):
        super().__init__()
        self.img_dir = img_dir    
        self.evaluation = evaluation
        self.augmentation = augmentation
        self.img_names = np.asarray([x.split("/")[-1] for x in glob.glob(img_dir + "/*")])
        if ("evaluation" not in img_dir):
            self.labels = np.asarray([int(x.split("_")[0]) for x in self.img_names])
            self.ids = np.asarray([x.split("_")[1].split(".")[0] for x in self.img_names])
        else: 
            self.labels = np.asarray([np.nan for x in self.img_names])
            self.ids = np.asarray([x.split(".")[0] for x in self.img_names])            
        
        if self.augmentation:
            self.transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                             transforms.RandomVerticalFlip(p=0.5),
                                             transforms.RandomRotation(10),
                                transforms.Resize(299),
                                transforms.CenterCrop(299),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
        else:
            self.transform = transforms.Compose([
                                transforms.Resize(299),
                                transforms.CenterCrop(299),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
           
    def __len__(self):
            return len(self.img_names)
        
    def __getitem__(self,i):
        img = self.img_dir + "/" + self.img_names[i]
        img = Image.open(img)
        img_as_tensor = self.transform(img)        
        lbl = self.labels[i]
        if self.evaluation:
            return img_as_tensor, lbl, self.ids[i] 
        return img_as_tensor, lbl
    
    def get_class(self,i):        
        lbl=self.img_names[i].split("_")[0]
        return int(lbl)
    
    def get_img(self,i):
        img = self.img_dir + "/" + self.img_names[i]
        img = Image.open(img)
        return img
    
    def get_dataframe(self):
        df = pd.DataFrame(columns=['label', 'img'])
        df['label'] = self.labels
        df['img'] = self.img_names
        return df
    
imgs_dir = "/kaggle/input/polytech-ds-2019/polytech-ds-2019/"

train_data = FoodDataset(imgs_dir + "training", augmentation=True)
validation_data = FoodDataset(imgs_dir + "validation")
evaluation_data = FoodDataset(imgs_dir + "kaggle_evaluation", evaluation=True)


def make_weights_for_balanced_classes(images, nclasses):                        
    count = [0] * nclasses                                                      
    for item in images:                                                         
        count[item[1]] += 1                                                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
    weight = [0] * len(images)                                              
    for idx, val in enumerate(images):                                          
        weight[idx] = weight_per_class[val[1]]                                  
    return weight    

weights = make_weights_for_balanced_classes(train_data , 11)                                                                
weights = torch.DoubleTensor(weights)                                       
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))                     
                       
train_dl = torch.utils.data.DataLoader(train_data, batch_size=32, num_workers=10, sampler=sampler)
val_dl = torch.utils.data.DataLoader(validation_data, batch_size=32, num_workers=10)


evaluation_dl = torch.utils.data.DataLoader(evaluation_data, batch_size=32, num_workers=10)
dataloaders_dict = {'train': train_dl, 'val' : val_dl}

#display(train_data[0])

#plt.imshow(plt.imread(train_data.get_img(7)))
train_data.get_img(5)
display_tensor(train_data[7][0])
#l = [plt.imread(imgs_dir +"training/" + x).shape for x in train_data.img_names]

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224


    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=0)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
        scheduler.step(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history
# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "inception"

# Number of classes in the dataset
num_classes = 11

# Batch size for training (change depending on how much memory you have)
batch_size = 32

# Number of epochs to train for
num_epochs = 1

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize the model for this run
model_ft, input_size = initialize_model(model_name= model_name, num_classes = num_classes, feature_extract = feature_extract, use_pretrained=True)




# Send the model to GPU
model_ft = model_ft.to(device)

params_to_update = model_ft.parameters()

if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)

           


# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

# Train and evaluate
model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))
ids = []
predictions = []

for data in evaluation_dl:
    outputs = model_ft(data[0].to(device))

    _, predicted = torch.max(outputs, 1)
    predictions.extend(predicted.tolist())
    ids.extend(data[2])
    
predictions = [" " + str(el) for el in predictions]
res = pd.DataFrame({"Id":ids, "Category":predictions})
res.to_csv("sample_submission.csv", index=False)

