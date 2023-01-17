import torchvision.models as models

import torch.nn as nn





#load in pretrained model

resnet = models.resnet50(pretrained=True)

alexnet = models.alexnet(pretrained=True)



#change fc layers for finetuning

output_size = 10

resnet.fc = nn.Linear(2048,output_size)

alexnet.classifier[6] = nn.Linear(4096,output_size)
from tqdm.notebook import tqdm #for progressbar 

#half precision training

def fit(model, data, criterion, optimizer, epochs = 10):

    pb = tqdm(range(epochs))

    for e in pb:

        for x,y in data:

            model.train() #set model to train

            model.half() #set model to half precision (fp16)

            prediction = model(x.cuda().half()) #make prediction on fp16 data (note:convert x to cuda)

            loss = criterion(prediction.float(), y.cuda().long()) #calculate loss with scaled prediction and target

            #target needs to be long() because of crossentropy and cuda() because of half precision

            model.float() #rescale model to full precision (fp32)

            optimizer.zero_grad() #reset optimizer

            loss.backward() #calculate gradient

            optimizer.step() #step optimizer with gradients

            pb.set_description(f"Epoch: {e} || Loss: {round(loss.item(),3)} || Acc: {accuracy(model,data)}")

            

def accuracy(model, data): #gets accuracy of model on data

    correct, total = 0.0,0.0

    for x,y in data:

        model.eval()

        with torch.no_grad():

            prediction = torch.argmax(model(x.cuda()), dim=1)

            correct+= sum(prediction == y.cuda())

            total += y.shape[0]

        return round((correct/total).item() * 100,4)

            
import torch

import torch.optim as optim

from torchvision.datasets import CIFAR10 as load_data

import torchvision.transforms as transforms

convert = transforms.Compose([transforms.Pad((128-32)//2),transforms.ToTensor()]) #converts to torch tensor data type

path = load_data(root='../working', train=False, download=True, transform=convert) #downloads data and applies conversion

data = torch.utils.data.DataLoader(path, batch_size=512, shuffle=True) #put into dataloader for convenience



criterion = nn.CrossEntropyLoss() #use cross entropy loss for multiclass

optimizer = optim.Adam(alexnet.parameters(), lr = 1e-3) #arbitiary choice of optimizer (train on alexnet parameters)

fit(alexnet.cuda(), data, criterion, optimizer) #note: need to use cuda() <- only works on GPU
optimizer = optim.Adam(resnet.parameters(), lr = 1e-3) #arbitiary choice of optimizer (train on resnet parameters)

fit(resnet.cuda(), data, criterion, optimizer) #note: need to use cuda() <- only works on GPU