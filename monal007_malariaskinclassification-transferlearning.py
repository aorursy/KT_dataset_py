## import libraries

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets , models , transforms
from torchvision.utils import make_grid
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from tqdm import tqdm
from sklearn.metrics import classification_report , confusion_matrix
from PIL import Image
%matplotlib inline

## Path , join

path = r"../input/imagedata/malaria"
train = os.path.join(path , "Train")
test = os.path.join(path ,"Test")
## Transformation

trans = transforms.Compose([
    transforms.Resize(112) , transforms.CenterCrop(112)  , transforms.RandomHorizontalFlip(p= 0.4) ,
    transforms.RandomVerticalFlip(p=0.5)  ,transforms.ToTensor()
])
## Joining the datasets

train_data = datasets.ImageFolder(train , transform = trans)
test_data = datasets.ImageFolder(test , transform = trans)

print(len(train_data) , len(test_data))
## Split the train_data into train and val

def split(n , p , random):
    s = int(n*p)
    np.random.seed(random)
    idx = np.random.permutation(n)
    
    return idx[s:] , idx[:s]

train_idx , val_idx = split(len(train_data) , 0.2 , 100)
## print
print(len(train_idx) , len(val_idx))
## make loader
train_load = SubsetRandomSampler(train_idx)
val_load = SubsetRandomSampler(val_idx)

train_loader = DataLoader(train_data , batch_size = 4  , sampler=train_load )
val_loader = DataLoader(train_data , batch_size = 4 , sampler=val_load )
test_loader = DataLoader(test_data , batch_size = 1  , shuffle = True)

## check images size
print(train_data.classes)
print(len(train_loader))
print(len(val_loader))
print(len(test_loader))

img , lab = train_data[0]
print(img.shape)

## plot image --- train_data
img , labels = train_data[0]
plt.imshow(img.permute(1,2,0))
print(train_data.classes[labels] , labels)
plt.show()
## plot image --- test data

img , labels = test_data[0]
plt.imshow(img.permute(1,2,0))
print(test_data.classes[labels] , labels)
plt.show()
## plot image -- train_loader
for img , lab in train_loader:
    print(img.shape)
    plt.imshow(img[0].permute(1,2,0))
    print("Label is " , lab[0])
    break
## plot image -- val_loader
for img , lab in val_loader:
    print(img.shape)
    plt.imshow(img[0].permute(1,2,0))
    print("Label is " , lab[0])
    break
## plot image -- test_loader
for img , lab in test_loader:
    print(img.shape)
    plt.imshow(img[0].permute(1,2,0))
    print("Label is " , lab[0])
    break
## Plot batch images (train_Loader)
def grid(dl):
    for img , lab in dl:
        fig , ax = plt.subplots(figsize = (10 ,10))
        ax.set_xticks([]) ; ax.set_yticks([])
        ax.imshow(make_grid(img , 8).permute(1,2,0))
        break
        
grid(train_loader)
        
    
## val loader
grid(val_loader)
grid(test_loader)  #batch size =1
# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if  train_on_gpu:
    print('CUDA is  available.  Training on GPU ...')
    device = "cuda"
else:
    print('CUDA is not available!  Training on CPU ...')
    device = "cpu"
## Use transfer learning (model = vgg16)
model = models.resnet152(pretrained = True)
model.to(device)
## freeze params
for param in model.fc.parameters():
    param.required_grad = False
## change the output layer
num_ftrs = model.fc.in_features
out = 2     #( pred class)
model.fc = nn.Linear(num_ftrs, out)

model = model.to(device)

model
## optimizer and loss_func
optimizer = optim.Adam(model.fc.parameters() , lr = 0.001)
loss_func = nn.CrossEntropyLoss()
### training on train and test on val

## training 

epoch = 10
val_loss_min = np.Inf

for i in (range(epoch)):
    train_plot_acc , train_plot_loss , val_plot_acc , val_plot_loss = [],[],[],[]
    train_loss = 0
    val_loss = 0
    train_acc = 0
    val_acc = 0
    
    # train_loader
    for img , lab in tqdm(train_loader):
        img , lab = img.to("cuda") , lab.to("cuda")
        
        optimizer.zero_grad()
        output = model(img)
        loss = loss_func(output , lab)
        
        loss.backward()
        optimizer.step()
        
        acc = ((output.argmax(dim=1) == lab).float().mean()).item()
        train_acc += acc/ len(train_loader)
        train_loss += loss.item() / len(train_loader)
        
        train_plot_loss.append(train_loss)
        train_plot_acc.append(train_acc)
        
    # Val_loader    
    for img , lab in tqdm(val_loader):
        img , lab = img.to("cuda") , lab.to("cuda")
        
        optimizer.zero_grad()
        output = model(img)
        loss = loss_func(output , lab)
        
        
        v_acc = ((output.argmax(dim=1) == lab).float().mean()).item()
        val_acc += v_acc/ len(val_loader)
        val_loss += loss.item() / len(val_loader)
        
        val_plot_loss.append(val_loss)
        val_plot_acc.append(val_acc)
        
    print("Epoch {} , Train_loss = {:.4f} , Val_loss = {:.4f} , Train_acc = {:.4f} , Val_acc = {:.4f}".format(
                    i+1 , train_loss , val_loss , train_acc , val_acc))
        
    if val_loss <= val_loss_min:
        print('Validation loss decreased ({:.4f} --> {:.4f}).  Saving model ...'.format(val_loss_min,val_loss))
        torch.save(model.state_dict(), 'malaria_resnet152.pt')
        val_loss_min = val_loss
plt.plot(train_plot_acc  , color = "blue")
plt.plot(train_plot_loss , color = "red")
plt.plot(val_plot_loss , color = "red")
plt.plot(val_plot_acc , color = "blue")
#testing
#test with test_loader


model.load_state_dict(torch.load("malaria_resnet152.pt"))
model.cuda()
model.eval()

with torch.no_grad():
    test_accuracy=0
    test_loss =0
    op_l = []
    lab = []
    for data, label in test_loader:
        data = data.to("cuda")
        label = label.to("cuda")

        output = model(data)
        loss = loss_func(output,label)


        acc = ((output.argmax(dim=1) == label).float().mean()).item()
        op_l.append((output.argmax(dim =1)).item())
        lab.append(label.item())
        test_accuracy += acc/ len(test_loader)
        
        test_loss += loss/ len(test_loader)
        

    print('test_accuracy : {}, test_loss : {}'.format(test_accuracy,test_loss))

print("predicted output"  ,op_l[:10])
print("Original label" , lab[:10])
#metrics

print(confusion_matrix(op_l , lab))
print(classification_report(op_l , lab))
## test with your own image

model.eval()
img_name = r"../input/imagedata/malaria/infect.jpg" # change this to the name of your image file.
def predict_image(image_path, model):
    image = Image.open(image_path)
    image_tensor = trans(image)
    image_tensor = image_tensor.unsqueeze(0)
    plt.imshow(image_tensor[0].permute(1,2,0))
    image_tensor = image_tensor.to(device)
    print(image_tensor.shape)
    
    output = model(image_tensor)
    index = output.argmax().item()
    if index == 0:
        return "Non-Parasitic"
    elif index == 1:
        return "Parasitic"
    else:
        return


predict_image(img_name,model)
################## If you like this Notebook ,please upvote ,this helps me to move forward ###################################################
