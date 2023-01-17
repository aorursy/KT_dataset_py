import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, cohen_kappa_score
import sys
import numpy as np
import pandas as pd
import torchvision
import torch.nn as nn
from tqdm import tqdm, notebook
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torch import optim, cuda
import torch
from torchvision import transforms
import os
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler
ImageFile.LOAD_TRUNCATED_IMAGES = True
from timeit import default_timer as timer
import scipy as sp
from functools import partial
from sklearn import metrics
from collections import Counter
import cv2
import random
from os.path import isfile
import math
import torch.nn.functional as F
import scipy.misc
from skimage.transform import resize
import numpy as np
import gc
import pickle
from sklearn.utils import shuffle
print(torch.cuda.is_available())
if(torch.cuda.is_available()):
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(2020)
learning_rate=1e-5
epochs=100
num_workers=2
batch_size=16
IMG_SIZE    = 256
path= "../input/covid19chestxray/"
def cleanUp(Df):
    Df["Present"]=0
    for idx in Df.index:
        name=Df["Image Name"][idx]
        if(isfile(path+"images/images/"+name)):
            Df["Present"][idx]=1
        else:
            Df["Present"][idx]=0
    Df=Df[Df["Present"]==1]
    return Df
trainDf=pd.read_csv(path+"Train_Combined.csv")
testDf=pd.read_csv(path+"Test_Combined.csv")
trainDf=cleanUp(trainDf)
testDf=cleanUp(testDf)
trainDf=trainDf.rename(columns={"Image Name":"Path"})
chexpertPath="../input/syde522pjte/CheXpert-v1.0-small/"
trainDf2=pd.read_csv(chexpertPath+"train.csv")
trainDf2=trainDf2[(trainDf2["Frontal/Lateral"]=="Frontal")  & (trainDf2["No Finding"]==1)]
trainDf2=trainDf2.sample(168,replace=False)
trainDf2=trainDf2.reset_index(drop=True)
Df1=trainDf[["Path"]]
Df1["Label"]=1
Df2=trainDf2[["Path"]]
Df2["Label"]=0
Df=Df1.append(Df2)
Df=shuffle(Df)
Df=Df.reset_index(drop=True)

Covid19Path= "../input/covid19chestxray/images/images/"
chexpertImagePath="../input/syde522pjte/"
def expand_path(p):
    p = str(p)
    if isfile(Covid19Path + p ):
        return (Covid19Path+p)
    if isfile(chexpertImagePath + p):
        return (chexpertImagePath+p)
    return False
traintransforms = transforms.Compose([
                #transforms.Grayscale(num_output_channels=1),
                transforms.Resize((224,224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor()
                ])

testtransforms = transforms.Compose([
                #transforms.Grayscale(num_output_channels=1),
                transforms.Resize((224,224)),
                transforms.ToTensor()
                ])
class CovidDataset(Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name=expand_path(self.data.loc[idx, "Path"])
        image=Image.open(img_name)
        image=image.convert('RGB')
        image=self.transform(image)
        label=self.data.loc[idx, "Label"]
        return image,label
    
chexpertPath="../input/syde522pjte/CheXpert-v1.0-small/"
testDf2=pd.read_csv(chexpertPath+"train.csv")
testDf2=testDf2[(testDf2["Frontal/Lateral"]=="Frontal")  & (testDf2["No Finding"]==1)]
testDf2=testDf2.sample(168,replace=False)
testDf2=testDf2.reset_index(drop=True)
tDf2=testDf2[["Path"]]
tDf2["Label"]=0
testDf["Path"]=testDf["Image Name"]
testDf["Label"]=testDf["COVID-19"]
testDf=testDf[["Path","Label"]]
testDf=testDf.append(tDf2)
shuffle(testDf)
testDf=testDf.reset_index(drop=True)
trainSet=CovidDataset(data=Df,transform=traintransforms)
trainLoader=torch.utils.data.DataLoader(trainSet, batch_size=batch_size, num_workers=num_workers)
testSet=CovidDataset(data=testDf,transform=testtransforms)
testLoader=torch.utils.data.DataLoader(testSet, batch_size=batch_size, num_workers=num_workers)
#baseModel =torchvision.models.resnext101_32x8d(pretrained=True, progress=True)
modelPath="../input/covid-19/CovidNetModel"
baseModel= torch.load(modelPath)
def train(model,learningRate,dataLoader,cvLoader,n_epochs=2):
        criterion=nn.BCEWithLogitsLoss()
        optimizer=optim.Adam(model.parameters(),lr=learningRate,weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        save_file_name='covidnetweight.pth'
        max_epochs_stop=5
        print_every=1
        #model.epochs=0
        model=model.to(device)
        ct=0
        for child in model.children():
            if ct < 2:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True
            ct=ct+1
        optimizer=optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=learningRate,weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.95)
        epochs_no_improve = 0
        valid_loss_min = np.Inf
        valid_max_acc = 0
        history = []
        try:
            print(f'Model has been trained for: {model.epochs} epochs.\n')
        except:
            model.epochs = 0
            print(f'Starting Training from Scratch.\n')
        overall_start = timer()
        for epoch in (range(n_epochs)):
            # keep track of training and validation loss each epoch
            if(epoch==8):
                for param in model.parameters():
                        param.requires_grad = True 
                optimizer=optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=learningRate,weight_decay=1e-5)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.95)
            train_loss = 0.0
            valid_loss = 0.0

            train_mse = 0
            valid_mse = 0

            # Set to training
            model.train()
            start = timer()

            # Training loop
            for ii, (data, target) in enumerate((dataLoader)):
                target=target.to(device).unsqueeze(1)
                data=data.to(device)
                #target=target.unsqueeze(1)
                # Clear gradients
                optimizer.zero_grad()
                # Predicted outputs are log probabilities
                output = model(data)
                # Loss and backpropagation of gradients
                loss=criterion(output,(target.float()))
                loss.backward()

                # Update the parameters
                optimizer.step()
                # Track train loss by multiplying average loss by number of examples in batch
                train_loss += loss.item() 
                # Calculate accuracy by finding max log probability
                #_, pred = torch.max(output, dim=1)
                #correct_tensor = pred.eq(target.data.view_as(pred))
                # Need to convert correct tensor from int to float to average
                #accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
                # Multiply average accuracy times the number of examples in batch
                #train_acc += accuracy.item() * data.size(0)

                # Track training progress
                print(
                    f'Epoch: {epoch}\t{100 * (ii + 1) / len(dataLoader):.2f}% complete. {timer() - start:.2f} seconds elapsed in epoch.',
                    end='\r')
            model.epochs += 1
            # After training loops ends, start validation
            if(epoch%2==0):
                    # Don't need to keep track of gradients
                with torch.no_grad():
                    # Set to evaluation mode
                    model.eval()

                    # Validation loop
                    for data, target in cvLoader:
                        # Tensors to gpu
                        target=target.to(device).unsqueeze(1)
                        data = data.to(device)
                        # Forward pass
                        output = model(data)

                        # Validation loss
                        loss=criterion(output,target.float())
                        # Multiply average loss times the number of examples in batch
                        valid_loss += loss.item()
                        # Calculate validation accuracy
                        

                    # Calculate average losses
                    train_loss = train_loss / len(dataLoader)
                    valid_loss = valid_loss / len(cvLoader)

                    # Calculate average accuracy
                    #train_acc = train_acc / len(train_loader.dataset)
                    #valid_acc = valid_acc / len(valid_loader.dataset)

                    history.append([train_loss, valid_loss,scheduler.get_lr()[0],model.epochs])

                    # Print training and validation results
                    if (epoch + 1) % print_every == 0:
                        print(
                            f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f}'
                        )
                        print('lr:', scheduler.get_lr()[0])
                        #print(
                        #    f'\t\tTraining Accuracy: {100 * train_acc:.2f}%\t Validation Accuracy: {100 * valid_acc:.2f}%'
                        #)

                    # Save the model if validation loss decreases
                    if valid_loss < valid_loss_min:
                        # Save model
                        torch.save(model.state_dict(), save_file_name)
                        torch.save(model,"CovidNetModel")
                        # Track improvement
                        epochs_no_improve = 0
                        valid_loss_min = valid_loss
                        best_epoch = epoch

                    # Otherwise increment count of epochs with no improvement
                    else:
                        epochs_no_improve += 1
                        # Trigger early stopping
                        if epochs_no_improve >= max_epochs_stop:
                            print(f'\nEarly Stopping! Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {valid_loss_min:.2f}')
                            total_time = timer() - overall_start
                            print(f'{total_time:.2f} total seconds elapsed. {total_time / (epoch+1):.2f} seconds per epoch.')

                            # Load the best state dict
                            model.load_state_dict(torch.load(save_file_name))
                            # Attach the optimizer
                            model.optimizer = optimizer

                            # Format history
                            history = pd.DataFrame(history,columns=['train_loss', 'valid_loss','Learning_rate','epochs'])
                            return model,history
            scheduler.step()
        # Attach the optimizer
        model.optimizer = optimizer
        # Record overall time and print out stats
        total_time = timer() - overall_start
        print(f'\nBest epoch: {best_epoch} with loss: {valid_loss_min:.2f} ')
        print(f'{total_time:.2f} total seconds elapsed. {total_time / (epoch+1):.2f} seconds per epoch.')
        # Format history
        history = pd.DataFrame(history,columns=['train_loss', 'valid_loss','Learning_rate','epochs'])
        return model,history
def plotLoss(history):
    plt.figure(figsize=(8, 6))
    for c in ['train_loss', 'valid_loss']:
        plt.plot(history['epochs'],history[c],label=c)
        plt.legend()
        plt.xlabel('no of epochs')
        plt.ylabel('MSE Losses')
        plt.title('Training and Validation Losses')
def saveHistory(fileName,history):
    history.to_pickle(fileName)
def retriveHistory(fileName):
    history=pd.read_pickle(fileName)
    return history
#baseModel.fc=nn.Sequential(nn.Linear(2048,out_features=1),nn.Sigmoid())
model,history=train(baseModel,1e-5,trainLoader,testLoader,n_epochs=600)
plotLoss(history)
saveHistory("History.pkl",history)
def checkAccuracy(model,cvLoader):
    acc=0
    with torch.no_grad():
        # Set to evaluation mode
        model.eval()

        # Validation loop
        for data, target in cvLoader:
            # Tensors to gpu
            target=target.to(device).unsqueeze(1)
            data = data.to(device)
            # Forward pass
            output = model(data)
            thresh=0.5
            acc+=((output.cpu()>thresh).float()==target.cpu().float()).sum().cpu()
        acc=float(acc)/ len(cvLoader.dataset)
        print(acc*100)
        return acc

    
acc=checkAccuracy(baseModel,trainLoader)
import torch
import torch.nn.functional as F
class ScoreCAM():
    def __init__(self, model):
        
        def forward_hook(module, input, output):
            self.activations= output.to(device)
            return None
        self.model=model
        self.activations = None
        self.model.layer4[2].relu.register_forward_hook(forward_hook)

    def forward(self, input):
        b, c, h, w = input.size()
        
        # predication on raw input
        baseScore = self.model(input)
        activations = self.activations.to(device)
        b, k, u, v = activations.size()
        finalAcivationMap = torch.zeros((b, 1, h, w)).to(device)
        contributionStack=torch.zeros(b,k,1).to(device)
        activationMapStack=torch.zeros((b, k,1, h, w)).to(device)
        with torch.no_grad():
            for i in tqdm(range(k)):

                # upsampling
                #activationMap = torch.unsqueeze(activations[:, i, :, :], 1)
                #saliency_map=activationMap
                activationMap=activations[:, i, :, :]
                activationMap = torch.unsqueeze(activationMap, 1)
                activationMap = F.interpolate(activationMap, size=(h, w), mode='bilinear', align_corners=False)
                activationMapStack[:,i]=activationMap
                if activationMap.max() == activationMap.min():
                    continue
              
              # normalize to 0-1
                norm_saliency_map = (activationMap - activationMap.min()) / (activationMap.max() - activationMap.min())

                # how much increase if keeping the highlighted region
                # predication on masked input
                score = self.model(input * norm_saliency_map)
                contributionStack[:,i]=score[0]-baseScore[0]
                #score_saliency_map +=  contribution * saliency_map
        contributionStack= F.softmax(contributionStack) 
        contributionStack=torch.squeeze(contributionStack,2)
        activationMapStack=torch.squeeze(activationMapStack,2)
        for j in tqdm(range(b)):
            for i in range (k):
                finalAcivationMap[j] +=  contributionStack[j,i] * activationMapStack[j,i]
        #finalAcivationMap = F.relu(finalAcivationMap)
        finalAcivationMapMin, finalAcivationMapMax = finalAcivationMap.min(), finalAcivationMap.max()

#         if finalAcivationMapMin == finalAcivationMapMax:
#             return None

#         finalAcivationMap = (finalAcivationMap - finalAcivationMapMin).div(finalAcivationMapMax - finalAcivationMapMin).data

        return finalAcivationMap

    def __call__(self, input):
        return self.forward(input)
def plotScorecam(image,image_normalized,cam_image,size,label):
    fig, axes = plt.subplots(size, 1,figsize=(15,15))
    plt.axis('off')
    cam_image=torch.squeeze(cam_image,1)
    for row in range(size):
        axes[row].text(0.5, 1.2,"Images And Attention Map of Covid Presence "+str(label[row]))
        axes[row].tick_params(labelcolor=(1.,1.,1., 0.0), top='off', bottom='off', left='off', right='off')
        axes[row].frameon = False
    cam_image=torch.squeeze(cam_image,1)
    for i in range(size):
        ax1=fig.add_subplot(size,4,(i*4)+1)
        ax1.tick_params(labelcolor=(1.,1.,1., 0.0), top='off', bottom='off', left='off', right='off')
        ax1.imshow(np.transpose(image[i].cpu(), (1, 2, 0)))
        ax2=fig.add_subplot(size,4,(i*4)+2)
        ax2.tick_params(labelcolor=(1.,1.,1., 0.0), top='off', bottom='off', left='off', right='off')
        ax2.imshow(np.transpose(image_normalized[i].cpu(), (1, 2, 0)))
        ax3=fig.add_subplot(size,4,(i*4)+3)
        ax3.tick_params(labelcolor=(1.,1.,1., 0.0), top='off', bottom='off', left='off', right='off')
        ax3.imshow((cam_image[i].cpu()))
        ax4=fig.add_subplot(size,4,(i*4)+4)
        ax4.tick_params(labelcolor=(1.,1.,1., 0.0), top='off', bottom='off', left='off', right='off')
        ax4.imshow(np.transpose(image[i].cpu(), (1, 2, 0)))
        ax4.imshow((cam_image[i].cpu()),alpha=0.4)
        if(i==0):
            ax1.set_title("Original Image")
            ax2.set_title("Normalized Image")
            ax3.set_title("Attention Map")
            ax4.set_title("Super imposed Image")
    plt.tight_layout()
    plt.show()
positiveCase=testDf[testDf["Label"]==1].sample(2)
negetiveCase=testDf[testDf["Label"]==0].sample(2)
camDf=negetiveCase.append(positiveCase)
camDf=camDf.reset_index(drop=True)
camSet=CovidDataset(data=camDf,transform=testtransforms)
camLoader=torch.utils.data.DataLoader(camSet, batch_size=4, num_workers=num_workers)

image,label=next(iter(camLoader))
image=image.to(device)
score_cam=ScoreCAM(baseModel)
activationMap=score_cam(image)
plotScorecam(image,image,activationMap,4,label)
