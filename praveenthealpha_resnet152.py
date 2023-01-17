!pip install jupyternotify
%load_ext jupyternotify
!pip3 install tensorboardX
# Run tensorboard on port 6006

LOG_DIR = './log'

run_num = 0

get_ipython().system_raw(

    'tensorboard --logdir {} --host 0.0.0.0 --port 6006 &'

    .format(LOG_DIR)

)
!cd ~

!curl -sL https://deb.nodesource.com/setup_8.x -o nodesource_setup.sh

!bash nodesource_setup.sh
!apt-get -y --allow-unauthenticated install nodejs
!npm i -g ngrok --unsafe-perm=true --allow-root
# Install localtunnel

!npm install -g localtunnel
# Tunnel port 6006 for tensorboard

get_ipython().system_raw('lt --port 6006 >> tensorboard.txt 2>&1 &')
from tensorboardX import SummaryWriter

import torchvision.utils as vutils

!pip install Pillow==4.0.0

!pip install PIL

!pip install image

from PIL import Image

def register_extension(id, extension): Image.EXTENSION[extension.lower()] = id.upper()

Image.register_extension = register_extension

def register_extensions(id, extensions): 

    for extension in extensions: register_extension(id, extension)

Image.register_extensions = register_extensions
import torch
cpu_device = torch.device("cpu")

cpu_device
# Initialize the Tensorboard writer

run_num += 1

writer = SummaryWriter(log_dir=LOG_DIR+"/run_{}".format(run_num))
import numpy

import torch.nn.functional as F

import torch.nn.init as init

import time

import copy
import os

from PIL import Image

import matplotlib.pyplot as plt



import torch

import torchvision

from torch.utils.data import DataLoader, Dataset, random_split

import torchvision.transforms as transforms



#For converting the dataset to torchvision dataset format

class VowelConsonantDataset(Dataset):

    def __init__(self, file_path,train=True,transform=None):

        self.transform = transform

        self.file_path=file_path

        self.train=train

        self.file_names=[file for _,_,files in os.walk(self.file_path) for file in files]

        self.len = len(self.file_names)

        if self.train:

            self.classes_mapping=self.get_classes()

    def __len__(self):

        return len(self.file_names)

    

    def __getitem__(self, index):

        file_name=self.file_names[index]

        image_data=self.pil_loader(self.file_path+"/"+file_name)

        if self.transform:

            image_data = self.transform(image_data)

        if self.train:

            file_name_splitted=file_name.split("_")

            Y1 = self.classes_mapping[file_name_splitted[0]]

            Y2 = self.classes_mapping[file_name_splitted[1]]

            z1,z2=torch.zeros(10),torch.zeros(10)

            z1[Y1-10],z2[Y2]=1,1

            label=torch.stack([z1,z2])

            return image_data, label



        else:

            return image_data, file_name

          

    def pil_loader(self,path):

        with open(path, 'rb') as f:

            img = Image.open(f)

            return img.convert('RGB')



      

    def get_classes(self):

        classes=[]

        for name in self.file_names:

            name_splitted=name.split("_")

            classes.extend([name_splitted[0],name_splitted[1]])

        classes=list(set(classes))

        classes_mapping={}

        for i,cl in enumerate(sorted(classes)):

            classes_mapping[cl]=i

        return classes_mapping

    
import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim



import torchvision

import matplotlib.pyplot as plt

from torchvision import datasets



import torchvision.transforms as transforms



import numpy as np

import pandas as pd



train_on_gpu = torch.cuda.is_available()
torch.cuda.current_device()

device = torch.device("cuda:0")
transform = transforms.Compose([

    transforms.Resize(224),

    transforms.ToTensor(),

    transforms.Normalize((0.50643307, 0.46160743, 0.42074028), (0.323405, 0.31323427, 0.33387515))

    ])
full_data = VowelConsonantDataset("../input/train/train",train=True,transform=transform)

train_size = int(0.9 * len(full_data))

test_size = len(full_data) - train_size



train_data, validation_data = random_split(full_data, [train_size, test_size])

dataloader = {}



dataloader["train"] = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)

dataloader["val"] = torch.utils.data.DataLoader(validation_data, batch_size=128, shuffle=True)
len(full_data)
test_data = VowelConsonantDataset("../input/test/test",train=False,transform=transform)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=16,shuffle=False)
def imshow(img,titles):

    img = img.numpy()

    plt.figure(figsize=(6,3))

    plt.axis("off")

    img = np.transpose(img,(1,2,0))

    plt.imshow(img)

    label = str(titles)

    plt.title("{}".format(label))

    plt.show()  
display_loader = torch.utils.data.DataLoader(train_data,batch_size=4,shuffle=True)
def show_images(dataloader):

    data,labels = next(iter(dataloader))

    print(labels.shape)

    img = torchvision.utils.make_grid(data)

    lst= [[torch.max(label[0],-1)[1].item(),torch.max(label[1],-1)[1].item()] for label in labels]

    imshow(img,titles=lst)
show_images(display_loader)
import torchvision.models as models
!pip install torchvision
resnet152 = models.resnet152(pretrained=True)
resnet152
class Model(nn.Module):

    def __init__(self,model,in_features=1000,hidden_features=500,out_features=10):

        super().__init__()

        self.model = model

        self.bn = nn.BatchNorm1d(in_features)

        self.classifier1 = nn.Sequential(

        nn.Linear(in_features,hidden_features),

        nn.BatchNorm1d(hidden_features),

        nn.LeakyReLU(),

        nn.Linear(hidden_features,out_features)   

        )

        

        self.classifier2 = nn.Sequential(

        nn.Linear(in_features,hidden_features),

        nn.BatchNorm1d(hidden_features),

        nn.LeakyReLU(),

        nn.Linear(hidden_features,out_features)       

        )



    def forward(self,x):

        out = self.model(x)

        out = self.bn(out)

        out = F.leaky_relu(out)

        

        out1 = self.classifier1(out)



        out2 = self.classifier2(out)

        

        return out1,out2
model_161 = Model(resnet152)
print(model_161)  # model_161.bn , model_161.classifier1 , model_161.classifier2
for params in model_161.model.parameters():

    params.requires_grad = False
for params in model_161.model.layer3.parameters():

    params.requires_grad = True
for params in model_161.model.layer4.parameters():

    params.requires_grad = True
for params in model_161.model.fc.parameters():

    params.requires_grad = True
for params in model_161.bn.parameters():

    params.requires_grad = True
for params in model_161.classifier1.parameters():

    params.requires_grad = True
for params in model_161.classifier2.parameters():

    params.requires_grad = True
for name, param in model_161.named_parameters():

    if param.requires_grad == True:

        print(name)

def init_weights(m):

    if isinstance(m, nn.Linear):

        xavier(m.weight.data)

    elif isinstance(m, nn.BatchNorm1d):

        m.weight.data.fill_(1)

        m.bias.data.zero_()

    else:

        pass



def xavier(parameters):

    init.kaiming_normal_(parameters,nonlinearity='leaky_relu')



model_161.apply(init_weights)
model_161 = model_161.to(device)
loss_fn1 = nn.CrossEntropyLoss()

loss_fn2 = nn.CrossEntropyLoss()

opt = optim.Adam([

                {'params': model_161.model.layer3.parameters(),"lr":0.0001},

                {'params': model_161.model.layer4.parameters(),"lr":0.0001},

                {'params': model_161.model.fc.parameters(),"lr":0.0001},

                {'params': model_161.bn.parameters(),"lr":0.0001},

                {'params': model_161.classifier1.parameters(),"lr":0.01},

                {'params': model_161.classifier2.parameters(),"lr":0.01}    

            ], lr=1e-2,weight_decay=0.4)



len(opt.param_groups)
print ("Go to this link below to see the Tensorboard:")

!cat tensorboard.txt

print ("Click on SCALARS to see metrics and DISTRIBUTIONS to see weights.")
def write_weights(writer, model, epoch_num):

    for name, param in model.named_parameters():

        if param.requires_grad == True:

            # Weights

            param_cpu  =  param.cpu()

            writer.add_scalar(name+"/mean", param_cpu.data.numpy().mean(), epoch_num)

            writer.add_scalar(name+"/std", param_cpu.data.numpy().std(), epoch_num)



            # Gradients

            writer.add_scalar(name+"/grad_mean", torch.mean(param.grad), epoch_num)

            writer.add_scalar(name+"/grad_std", torch.std(param.grad), epoch_num)



            # Weights histogram (dim over 1024 cause an error)

            if len(param.size()) > 1 and param.size()[-1] <= 1024: 

                writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch_num)

            del param,param_cpu
from torch.optim.lr_scheduler import ReduceLROnPlateau
scheduler = ReduceLROnPlateau(opt, 'min',patience=3,verbose=True)
def train_model(model, dataloaders,loss_fn1,loss_fn2,scheduler,optimizer, num_epochs=100):

    since = time.time()



    val_acc_history = []



    best_model_wts = copy.deepcopy(model.state_dict())

    best_acc = 0.0



    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        print('-' * 10)



        # Each epoch has a training and validation phase

        for phase in ['train', 'val']:

            if phase == 'train':

                model.train()  # Set model to training mode

            else:

                model.eval()   # Set model to evaluate mode

            

            print("{} Mode ............".format(phase.upper()))

            running_loss = 0.0

            running_acc = 0.0

            acc1 = 0.0

            acc2 = 0.0

            main_acc = 0.0 



            # Iterate over data.

            for inputs, labels in dataloaders[phase]:

                inputs = inputs.to(device)

                labels = labels.long().to(device)

                _,label1 = torch.max(labels[:,0,:],1)

                _,label2 = torch.max(labels[:,1,:],1)

                # zero the parameter gradients

                optimizer.zero_grad()



                # forward

                # track history if only in train

                with torch.set_grad_enabled(phase == 'train'):

                    # Get model outputs and calculate loss

                    # Special case for inception because in training it has an auxiliary output. In train

                    #   mode we calculate the loss by summing the final output and the auxiliary output

                    #   but in testing we only consider the final output.

        

                    output = model(inputs)

                    loss1 = loss_fn1(output[0], label1)

                    loss2 = loss_fn2(output[1], label2)

                    loss = (1.5*loss1)+loss2

                    _, pred1 = torch.max(output[0], 1)

                    _, pred2 = torch.max(output[1], 1)



                    # backward + optimize only if in training phase

                    if phase == 'train':

                        loss.backward()

                        optimizer.step()



                # statistics

                running_loss += loss.item()

                acc1 += torch.sum(pred1 == label1).item()/inputs.size(0)

                acc2 += torch.sum(pred2 == label2).item()/inputs.size(0)

                main_acc += torch.sum((pred2 == label2) & (pred1 == label1)).item()/inputs.size(0)

                

                del inputs, labels, output

                torch.cuda.empty_cache()



            epoch_loss = running_loss / len(dataloaders[phase])

            epoch_acc1 = acc1/ len(dataloaders[phase])

            epoch_acc2 = acc2/len(dataloaders[phase])

            epoch_acc = (epoch_acc1 + epoch_acc2)/2

            main_epoch_acc = main_acc/len(dataloaders[phase])

            if phase == "val":

                scheduler.step(epoch_loss)



            print('{} Loss: {:.4f} Acc: {:.4f}  Acc1: {:.4f}  Acc2: {:.4f} MainAcc: {:.4f}'.format(phase, epoch_loss, epoch_acc,epoch_acc1,epoch_acc2,main_epoch_acc))



            

            if phase == "train":

                writer.add_scalar('metrics/train_loss', epoch_loss, epoch)

                writer.add_scalar('metrics/train_acc', main_epoch_acc, epoch)

            else:

                writer.add_scalar('metrics/val_acc', epoch_acc, epoch)

                writer.add_scalar('metrics/val_loss', main_epoch_acc, epoch)

            

            if phase == "train":

                for i in range(6):

                    writer.add_scalar('metrics/lr'+str(i), optimizer.param_groups[i]['lr'], epoch)



                write_weights(writer=writer, model=model, epoch_num=epoch)

           

            # deep copy the model

            if phase == 'val' and epoch_acc > best_acc:

                best_acc = epoch_acc

                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'val':

                val_acc_history.append(epoch_acc)



        print()



    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val Acc: {:4f}'.format(best_acc))



    # load best model weights

    model.load_state_dict(best_model_wts)

    return model, val_acc_history
best_model, best_model_val_acc_history = train_model(model_161,dataloader,loss_fn1,loss_fn2,scheduler,opt,num_epochs=1)
%%notify

time.sleep(3)
# Add this to console to visualize !ngrok http 6006 
# print(best_model)
# loss_fn1 = nn.CrossEntropyLoss()

# loss_fn2 = nn.CrossEntropyLoss()

# opt2 = optim.SGD([

#                 {'params': best_model.model.layer3.parameters(),"lr":0.0001},

#                 {'params': best_model.model.layer4.parameters(),"lr":0.0001},

#                 {'params': best_model.model.fc.parameters(),"lr":0.0001},

#                 {'params': best_model.bn.parameters(),"lr":0.0001},

#                 {'params': best_model.classifier1.parameters(),"lr":0.01},

#                 {'params': best_model.classifier2.parameters(),"lr":0.01}  

#             ], nesterov=True,momentum = 0.9,lr=1e-2,weight_decay=0.4)

scheduler2 = ReduceLROnPlateau(opt2, 'min',patience=3,verbose=True)
# loss_fn1 = nn.CrossEntropyLoss()

# loss_fn2 = nn.CrossEntropyLoss()

# opt2 = optim.Adam([

#                 {'params': best_model.model.features.denseblock3.parameters(),"lr":0.0001},

#                 {'params': best_model.model.features.transition3.parameters(),"lr":0.0001},

#                 {'params': best_model.model.features.denseblock4.parameters(),"lr":0.0001},

#                 {'params': best_model.model.features.norm5.parameters(),"lr":0.0001},

#                 {'params': best_model.model.classifier.parameters(),"lr":0.0001},

#                 {'params': best_model.classifier1.parameters(),"lr":0.001},

#                 {'params': best_model.classifier2.parameters(),"lr":0.001}

#             ], lr=1e-2,weight_decay=0.1)





# best_model, best_model_val_acc_history = train_model(scheduler=scheduler2,num_epochs=10,model=best_model,dataloaders=dataloader,loss_fn1=loss_fn1,loss_fn2=loss_fn2,optimizer=opt2)
# def test(best_model):

#     best_model.eval()

#     confusion_matrix1 = np.zeros([10, 10])

#     confusion_matrix2 = np.zeros([10, 10])

    

#     with torch.no_grad():

        

#         for inputs, labels in dataloader["val"]:

            

#             inputs = inputs

#             labels = labels.long()

            

#             output = best_model(inputs)

            

#             _,pred1 =  torch.max(output[0], 1)

#             _,pred2 =  torch.max(output[1], 1)

#             _,label1 = torch.max(labels[:,0,:],1)

#             _,label2 = torch.max(labels[:,1,:],1)

            

#             del inputs, labels, output

            

#             for x, y in zip(pred1.numpy(), label1.numpy()):

#                 confusion_matrix1[x][y] += 1

            

#             for x, y in zip(pred2.numpy(), label2.numpy()):

#                 confusion_matrix2[x][y] += 1

    

#     return confusion_matrix1,confusion_matrix2
# def plot_confusion_matrix(confusion_matrix,name):

#     classes = np.arange(10)

#     fig, ax = plt.subplots()

#     im = ax.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)

#     ax.figure.colorbar(im, ax=ax)

#     ax.set(xticks=np.arange(confusion_matrix.shape[1]),

#                 yticks=np.arange(confusion_matrix.shape[0]),

#                 xticklabels=classes, yticklabels=classes,

#                 ylabel='True label',

#                 xlabel='Predicted label',

#                 title = name)

#     thresh = confusion_matrix.max() / 2.

#     for i in range(confusion_matrix.shape[0]):

#         for j in range(confusion_matrix.shape[1]):

#             ax.text(j, i, int(confusion_matrix[i, j]),

#                     ha="center", va="center",

#                     color="white" if confusion_matrix[i, j] > thresh else "black")

              

#     fig.tight_layout()
# confusion_matrix1,confusion_matrix2 = test(best_model)
# %%notify

# time.sleep(3)
# plot_confusion_matrix(confusion_matrix1,"Vowel")
# plot_confusion_matrix(confusion_matrix2,"Consonants")
# del confusion_matrix1,confusion_matrix2
# a = torch.randn([4,3,224,224]).to(device)

# test = best_model(a)

# test[0].shape,test[1].shape
next(best_model.parameters()).is_cuda
from collections import OrderedDict
def predict_test(model,test_loader):

    od = OrderedDict()

    for img,file_names in test_loader:

        img = img.to(device)

        pred_1,pred_2 = model(img)

        for pred1,pred2,file_name in zip(pred_1,pred_2,file_names):

            od[file_name] = "V"+str(int(torch.max(pred1,-1)[1].item()))+"_"+"C"+str(int(torch.max(pred2,-1)[1].item()))

        del img,file_names,pred_1,pred_2

        torch.cuda.empty_cache()

    return od
test_label_dict = predict_test(best_model,test_loader)
import pandas as pd
submission = pd.DataFrame({'ImageId':list(test_label_dict.keys()),'Class':list(test_label_dict.values())})

submission.tail()
submission.head()
submission.to_csv("submisision.csv", index=False)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



# create a random sample dataframe



# create a link to download the dataframe

create_download_link(submission)



# ↓ ↓ ↓  Yay, download link! ↓ ↓ ↓ 