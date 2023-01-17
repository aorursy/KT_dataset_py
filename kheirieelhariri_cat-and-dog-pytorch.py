import numpy as np 
import pandas as pd 
import random
import matplotlib.pyplot as plt

from PIL import Image

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.optim as optim

import os
class  MakeDataset(Dataset): 
    
    
    def __init__(self, paths = None, DataList = None, classes=["cat","dog"], transform = None):        
        self.data = []
        self.classes = classes
        self.paths = paths
        self.datalist = DataList # The main goal of this argument is to serve the _split_dataset method
                                 # DataList is a list to be converted to type Dataset. 
                                 # This List should contain a 2 element list the 0th index is the Image (PIL, numpy or tensor) & 1st index is the label as digits or one hot encoded 
        self.transform = transform
        self._init_dataset()
                        
    def _init_dataset(self):
        if self.datalist: 
            self.data = self.datalist
        if self.paths:
            for path in self.paths:
                for d in os.listdir(path):
                    if "jpg" not in d:
                        continue
                    class_, num, ext = d.split('.')
                    image = Image.open(path + d)
                    
                    self.data.append([image, self.classes.index(class_)])
        
    def _split_dataset(self, split = 0.2):
        len_first_frac = self.__len__() * (1-split)
        len_second_frac = self.__len__() - len_first_frac
        self.first_fraction = random.sample(self.data, int(len_first_frac))
        self.second_fraction = random.sample(self.data, int(len_second_frac))
        return (MakeDataset(DataList=self.first_fraction, transform = self.transform), 
               MakeDataset(DataList=self.second_fraction, transform = self.transform))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        if self.transform: 
            x = self.transform(x)
        return x, y
train_transform = transforms.Compose([
                            transforms.Resize([256,256]), 
                            transforms.RandomHorizontalFlip(),
                            transforms.CenterCrop(size=224), # Imagenet standards
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])  # Imagenet standards
                                    ])

test_transform = transforms.Compose([
                            transforms.Resize([256,256]), 
                            transforms.CenterCrop(size=224), # Imagenet standards
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])  # Imagenet standards
                                    ])
classes = ["cat","dog"]
train_paths = ["../input/cat-and-dog/training_set/training_set/cats/", "../input/cat-and-dog/training_set/training_set/dogs/"]
test_path = ["../input/cat-and-dog/test_set/test_set/cats/", "../input/cat-and-dog/test_set/test_set/dogs/"]
TrainData = MakeDataset(paths = train_paths, classes = classes, transform = train_transform)
TestData = MakeDataset(paths = test_path, classes = classes, transform = test_transform)
print("len of TrainData: ", TrainData.__len__())
print("len of TestData: ", TestData.__len__())
# Test the __getitem__ method on TrainData
TrainData.__getitem__(8000)
# Test the __getitem__ method on TestData
TestData.__getitem__(5)
# Split the TrainData into Train and Valid 
Train, Valid = TrainData._split_dataset(split = 0.2)
print("Train is of type", type(Train)) 
print("Valid is of type", type(Valid)) 
print("len of Train: ", Train.__len__())
print("len of Valid: ", Valid.__len__())
def show_image(data, idx):
    plt.figure(figsize=(6,6))
    label = classes[data.__getitem__(idx)[1]]
    plt.title(label)
    plt.imshow(np.transpose(data.__getitem__(idx)[0], (1,2,0))) #make the color channel the 3rd dimension
    plt.show()
show_image(Train, 5000)
show_image(Valid, 0)
DataLoaders = {
    'train': DataLoader(Train, batch_size=100, shuffle=True),
    'valid': DataLoader(Valid, batch_size=100, shuffle=True),
    'test': DataLoader(TestData, batch_size=100, shuffle=True)
}
train_features, train_labels = next(iter(DataLoaders['train']))
train_features.shape # torch.Size([batch_size, channels, height, width])
train_labels.shape
val_features, val_labels = next(iter(DataLoaders['valid']))
val_features.shape
val_labels.shape
class Model(nn.Module):
    def __init__(self, num_classes = 2):
        
        super(Model, self).__init__()
        
        self.num_classes = num_classes
        self.network = models.resnet18(pretrained=True)
        
        # freeze early parameters
        for param in self.network.parameters():
            param.requires_grad = False
            
        #get number of features of the fully connected layer
        num_ftrs = self.network.fc.in_features
        
        #update fully connected layer
        self.network.fc = nn.Linear(num_ftrs, self.num_classes)
                                
    def forward(self, x): 
        return self.network(x)
                                
if torch.cuda.is_available():
    device = torch.device("cuda:0") 
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")
my_model = Model(num_classes = 2).to(device)
optimizer = optim.Adam(my_model.parameters(), lr=0.001)
loss_function = nn.CrossEntropyLoss()   # use NLLLoss if last layer in network is log_softmax()
                                        # crossentropy = log_softmax() + NLLLoss()
EPOCHS = 20

def train_model(model = my_model, 
                Data = DataLoaders, 
                criterion = loss_function , 
                optimizer = optimizer, 
                max_epochs_stop = 3, 
                num_epochs = EPOCHS, 
               save_file_name = "resnet18_transfer_pytorch.pt" # A common pytorch convention is to save models using .pt or .pth 
               ): 
            
    minimum_loss = np.inf # positive infinity
    epoch_no_improve = 0.0 # to count epochs with no improvement
    
    for epoch in range(num_epochs): 
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        for phase in ["train", "valid"]: 
            
            correct = 0.0
            losses = 0.0
            
            if phase == "train":
                model = model.train() # model is set to training mode
            
            if phase == "valid": 
                model = model.eval() # model is set to evaluate mode
                
            for features, labels in Data[phase]: 
                features = features.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad() # set the parameter gradients to zero
                
                outputs = model(features)
                _, predictions = torch.max(outputs, 1)
                loss = criterion(outputs, labels) # validation loss
                 
                correct += torch.sum(predictions == labels) 
                
                losses += loss.item() * features.size(0) # multiply average loss times number of examples in batch
                                                         # item() returns the value of a tensor as a standard python number
                
                if phase == "train": 
                    loss.backward()
                    optimizer.step()    
            
            epoch_accuracy = correct/len(Data[phase])
            epoch_loss = loss/len(Data[phase])
            print(("{} accuracy: {:.4f} loss: {:.4f}").format(phase, epoch_accuracy, epoch_loss))
            
            # Early stopping
            if phase == "valid":
                if epoch_loss < minimum_loss: 
                    torch.save(model.state_dict(), save_file_name) # Save the model
                    minimum_loss = epoch_loss
                    epoch_no_improve = 0
                else:
                    epoch_no_improve += 1
                    if epoch_no_improve >= max_epochs_stop:
                        model.load_state_dict(torch.load(save_file_name)) # Load best state dict
                        model.optimizer = optimizer # attach optimizer
                        print("Early Stopping!")
                        return model   
                    
    return model       
my_model = train_model()
# Function to display predicted images with their labels 

def display_pridicted(model, data, num_of_imgs):
    
    counter = 0
    
    for features, labels in data:
        
        features = features.to(device)
        labels = labels.to(device)
        
        output = model(features)
        _, predictions = torch.max(output.data, 1)
    
    
    for j in range(features.size()[0]):
        
        counter += 1
         
        plt.figure(figsize=(12,12))
        plt.subplot(num_of_imgs//2, 2, counter)
        plt.title("Prediced label: {}, True label: {}".format(classes[predictions[j]], classes[labels[j]]))
        plt.imshow(np.transpose(features[j].cpu(), (1,2,0))) 
        
        if counter == num_of_imgs: 
            plt.show()
            return
# Trying the model on unseen data

display_pridicted(model = my_model, data = DataLoaders['test'], num_of_imgs = 6)