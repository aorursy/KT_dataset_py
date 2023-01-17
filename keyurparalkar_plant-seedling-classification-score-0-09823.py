import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.models as models

import os
from PIL import Image
plt.ion()
PATH = '../input'
#loadingg test set data:
def load_test_data(data_path,transform):
    temp = []
    
    allTestImages = os.listdir(data_path)
    for x in allTestImages:
        img = Image.open(data_path+'/'+allTestImages[1])
        temp.append(transform(np.array(img)))
        
    return temp
#Loading train dataset
transform = {'train': transforms.Compose([
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]),
  'test':transforms.Compose([
    transforms.ToPILImage(),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
}
trainData = torchvision.datasets.ImageFolder(root=PATH+'/train',transform=transform['train'])
trainLen = len(trainData)
trainData1, valData = torch.utils.data.dataset.random_split(trainData,[int((trainLen*4)/5),int(trainLen/5)])

trainData1Loader = torch.utils.data.DataLoader(dataset=trainData1, shuffle=False, batch_size=4)
valDataLoader = torch.utils.data.DataLoader(dataset=valData, shuffle=False, batch_size=4)
len(trainData1)
#Loading test dataset
testData = torch.stack(load_test_data(PATH+'/test',transform=transform['test'])) #For converting list to tensor
# testData = transform(testData)
testDataLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("TRAIN DATASET === ")
print("No. of examples = ",len(trainData1Loader.dataset))
print("VAL SET ==== ")
print("No. of examples =",len(valDataLoader.dataset))
print("\nTEST DATASET ===")
print("No. of exmaples = ",testDataLoader.dataset.size()[0])
#Visualizing Train dataset
'''
In trainDataLoader Dimensions are given as
dim. index              0    1    2
Actual Dims.           [3   128  128]

These dimension are not suitable for plt.imshow() it needs dimensions in the format HxWxC but we have CxHxW
So to change this we need our this dim. sequence = 0,1,2 in this format i.e. new dim. sequence 1,2,0 i.e. HxWxC
So thats what np.transpose(img,(1,2,0)) is doing its changing the dims to suitable format.
'''

def imageShow(img):
    img = [0.229, 0.224, 0.225]*np.transpose(img.numpy(),(1,2,0)) + [0.485, 0.456, 0.406]
    plt.imshow(img)
    plt.xlabel('Train images batch = 4')
    
iterator = iter(trainData1Loader)
image, label =  iterator.next()

imageShow(torchvision.utils.make_grid(image))
print('Ground Truth = \n',' '.join('%10s' % trainData1Loader.dataset.classes[x] for x in label.numpy()))

# Forward => loss => backward => update_weights
def train_model(model,criterion,optimizer,scheduler,dictionary,num_epochs=12):
    correct = 0
    total = 0
    totalLoss = []
    prediction = []
    temp = []
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}\n'.format(epoch,num_epochs-1))
        scheduler.step()   #to step or to update weights
        model.train()
            
        for batch_id,(image,label) in enumerate(trainData1Loader):
            optimizer.zero_grad()
                
            image = image.to(device)
            label = label.to(device)
                
            outputs = model(image)
            _, predictionIndex = torch.max(outputs,1)
            loss = criterion(outputs,label)
            prediction.append(predictionIndex)
            
            #printing loss =
            print("Loss = {0:.5f}".format(loss.item()),end="\r")
            correct += (predictionIndex == label).sum().item()
            total +=label.size(0)
            
            loss.backward()
            optimizer.step()        
            
            del image, label    #important
            
        totalLoss.append(loss)
#         prediction.append(temp)
        torch.cuda.empty_cache()      #important
        
    dictionary['totalLoss'] = totalLoss
    dictionary['correct'] = correct
    dictionary['totalSize'] = total
    dictionary['prediction'] = prediction
    
    #ALWAYS return the model object
    return model
model_ft = models.vgg16(pretrained=True)

for child in model_ft.features.children():
    for param in child.parameters():
        param.requires_grad = False
        
    
    
model_ft.classifier[6].out_features = 12
print(model_ft)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss().cuda()

optimizer_ft = torch.optim.SGD(params=model_ft.classifier.parameters(), lr=0.001, momentum=0.9)

exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

dictModel = {}
model_ft = train_model(model_ft,criterion,optimizer_ft,exp_lr_scheduler,dictionary=dictModel)
dictModel
#loss vs iteration graph:
plt.plot(dictModel['totalLoss'])
plt.xlabel('epochs')
plt.ylabel('loss')
print("Train Accuracy = ",100*(dictModel['correct']/dictModel['totalSize']))
#Validation set:
model_ft_val = train_model(model_ft,criterion,optimizer_ft,exp_lr_scheduler,dictionary=dictModel)
print("Val Accuracy = ",100*(dictModel['correct']/dictModel['totalSize']))
# Validation set
plt.plot(dictModel['totalLoss'])
plt.xlabel('epochs')
plt.ylabel('loss')
model_ft_val = model_ft_val.to(device)
model_ft_val.eval()

result = []

for batch_id,image in enumerate(testDataLoader):
    img = image.to(device)
    ip = torch.autograd.Variable(img)
    testOutput = model_ft(ip)
    _, testPredictionIndex = torch.max(testOutput,1)
    result.append(testPredictionIndex)


result
temp = []
for x in result:
    for y in x.cpu().numpy():
        temp.append(y)
len(temp)
dfDict = {
    'file':os.listdir(PATH+'test'),
    'species':[trainData.classes[m] for m in temp]
}
df = pd.DataFrame(dfDict)
df.to_csv(path_or_buf='submission.csv',index=False)
print(df)
