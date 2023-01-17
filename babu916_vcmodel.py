!pip install mlflow --quiet
!pip install pyngrok --quiet

# import mlflow

# with mlflow.start_run(run_name="MLflow on Colab"):
#   mlflow.log_metric("m1", 2.0)
#   mlflow.log_param("p1", "mlflow-colab")



# run tracking UI in the background
get_ipython().system_raw("mlflow ui --port 5000 &") # run tracking UI in the background


# create remote tunnel using ngrok.com to allow local port access
# borrowed from https://colab.research.google.com/github/alfozan/MLflow-GBRT-demo/blob/master/MLflow-GBRT-demo.ipynb#scrollTo=4h3bKHMYUIG6

from pyngrok import ngrok

# Terminate open tunnels if exist
ngrok.kill()

# Setting the authtoken (optional)
# Get your authtoken from https://dashboard.ngrok.com/auth
NGROK_AUTH_TOKEN = ""
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# Open an HTTPs tunnel on port 5000 for http://localhost:5000
public_url = ngrok.connect(port="5000", proto="http", options={"bind_tls": True})
print("MLflow Tracking UI:", public_url)
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
    
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets

import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import mlflow


import copy
import numpy as np
import pandas as pd

train_on_gpu = torch.cuda.is_available()
transform_train = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5]),
    ])
full_data = VowelConsonantDataset("../input/padhai-tamil-vow-cons-classification/train/train",train=True,transform=transform_train)
train_size = int(0.9 * len(full_data))
test_size = len(full_data) - train_size

train_data, validation_data = random_split(full_data, [train_size, test_size])

batch_sz = 60

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_sz, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=batch_sz, shuffle=True)
def get_labelnum(x):
    return torch.max(x,-1)[1].item()
test_data = VowelConsonantDataset("../input/padhai-tamil-vow-cons-classification/test/test",train=False,transform=transform_train)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_sz,shuffle=False)
def imshow(img, title):
    npimg = img.numpy()
    plt.figure(figsize=(16, 3))
    plt.axis('off')
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()

def show_batch_images(dataloader):
    images, labels = next(iter(dataloader))
    print(labels.shape)
    print(labels[:,0,:])
    
    img = torchvision.utils.make_grid(images)
    lst= [[get_labelnum(label[0]),get_labelnum(label[1])] for label in labels]
    
    imshow(img, title=lst)
display_loader = torch.utils.data.DataLoader(train_data,batch_size=1,shuffle=True)

show_batch_images(display_loader)


class Params(object):
    def __init__(self, batch_size, epochs, seed, log_interval):
        self.batch_size = batch_size
        self.epochs = epochs
        self.seed = seed
        self.log_interval = log_interval       
   
from torchvision import models

    
class VCModel(nn.Module):
    
    def __init__(self,in_features=1000,hidden_features=500,out_features=10):
        super().__init__()
        self.restnet_model = models.resnet18(pretrained=True)


        self.vowel_classifier = nn.Sequential(
          nn.Linear(in_features, hidden_features),
          nn.BatchNorm1d(hidden_features),
          nn.Dropout(0.3),
          nn.ReLU(),
          nn.Linear(hidden_features, out_features))


        self.cons_classifier = nn.Sequential(
          nn.Linear(in_features, hidden_features),
          nn.BatchNorm1d(hidden_features),
          nn.Dropout(0.3),              
          nn.ReLU(),
          nn.Linear(hidden_features, out_features))

    def forward(self, x):
        out = self.restnet_model(x)
        vout = self.vowel_classifier(out)
        cout = self.cons_classifier(out)
        return vout,cout
            

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)



def model_train(model, opt, loss_fn, train_loader, hparam):

    min_loss = 1000    
    n_iters = np.ceil(50000/hparam.batch_size)    
    
    for batchid, data in enumerate(train_loader, 0):

        model.train()
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        opt.zero_grad()

        labels = labels.long().to(device)
        _,l1 = torch.max(labels[:,0,:],1)
        _,l2 = torch.max(labels[:,1,:],1)


        outputs = model(inputs)
        loss1 = loss_fn(outputs[0], l1)
        loss2 = loss_fn(outputs[1], l2)
        
        loss = loss1+loss2 

        loss.backward()
        opt.step()

        if min_loss > loss.item():
            min_loss = loss.item()
            best_model = copy.deepcopy(model.state_dict())
            print('Min loss %0.2f' % min_loss)

        if batchid % args.log_interval == 0:
            mlflow.log_metric('train_loss', loss.data.item()/len(inputs)*1000)
            print('Iteration: %d/%d, Loss: %0.2f' % (batchid,n_iters, loss.item()))

        del inputs, labels, outputs
        torch.cuda.empty_cache()

    return (loss.item(), best_model)

def model_evaluate(dataloader, model, loss_fn):
    total, correct = 0, 0
    loss = 0
    model.eval()
    
    with torch.no_grad():
        
        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            _,l1 = torch.max(labels[:,0,:],1)
            _,l2 = torch.max(labels[:,1,:],1)

            _, p1 = torch.max(outputs[0].data, 1)
            _, p2 = torch.max(outputs[1].data, 1)


            loss1 = loss_fn(outputs[0], l1)
            loss2 = loss_fn(outputs[1], l2)

            loss += (loss1+loss2).item() 

            ll1, ll2 = (p1==l1), (p2==l2)        

            total += labels.size(0)
            correct += (ll1 == ll2).sum().item()  
            #print ('total {} correct {}'.format(total,correct))
        
    loss /= len(dataloader.dataset)        
    acc = 100.0 * correct / total
    
    return loss, acc
def run_epochs(model,a_lossfn, a_opt, hparams):
    
    per_epoch_loss = []
    for epoch in range(1, hparams.epochs + 1):
        loss, best_model = model_train(model, a_opt, a_lossfn, train_loader,hparams)
        
        model.load_state_dict(best_model)

        eval_loss, eval_accu = model_evaluate(validation_loader, model, a_lossfn)        
        print("Epoch: {} Test [loss {}, acc {}]" .format(epoch,eval_loss,eval_accu))
                
        mlflow.log_metric('test_loss', eval_loss*1000)
        mlflow.log_metric('test_accuracy', eval_accu)
           
        mlflow.pytorch.log_model(model, "models")        
        per_epoch_loss.append(loss)

    plt.plot(per_epoch_loss)
    plt.show()


# for lr in [0.01, 0.02, 0.05, 0.1]:
#     for mtm in [0.9, 0.95, 0.99]:
        
#         vcmodel = VCModel()
#         vcmodel.to(device)
#         loss_fn = nn.CrossEntropyLoss()
#         opt = optim.SGD(vcmodel.parameters(), lr=lr, momentum = mtm)
        
#         args = Params(batch_size=batch_sz, epochs=4, seed=0, log_interval=20)
#         mlflow.set_experiment("SGD LR{} MOMENTUM{}".format(lr,mtm))
#         with mlflow.start_run() as run:        
#             run_epochs(vcmodel,loss_fn, opt, args)
            
#         del vcmodel, loss_fn, opt

        
        
#0.01, 0.95 --> 

sgd_lr = 0.01
sgd_mtm = 0.95

vcmodel = VCModel()
vcmodel.to(device)
loss_fn = nn.CrossEntropyLoss()
opt = optim.SGD(vcmodel.parameters(), lr=sgd_lr, momentum = sgd_mtm)

args = Params(batch_size=batch_sz, epochs=4, seed=0, log_interval=20)
mlflow.set_experiment("SGD LR{} MOMENTUM{}".format(lr,mtm))
with mlflow.start_run() as run:        
    run_epochs(vcmodel,loss_fn, opt, args)

adam_lr = 0.01
vcmodel1 = VCModel()
vcmodel1.to(device)
loss_fn = nn.CrossEntropyLoss()
opt = optim.Adam(vcmodel1.parameters(), lr=adam_lr)

args = Params(batch_size=batch_sz, epochs=4, seed=0, log_interval=20)
mlflow.set_experiment("ADAM LR{}".format(adam_lr))
with mlflow.start_run() as run:        
    run_epochs(vcmodel1,loss_fn, opt, args)

import seaborn as sns
model_load = mlflow.pytorch.load_model('file:///kaggle/working/mlruns/12/2fa3bd5c156240bdbf9614c054a25110/artifacts/models')
print(model_load)
model_load.to('cpu')
weight_layer0 = list(model_load.parameters())[0].data.numpy()
sns.distplot(weight_layer0.ravel())
def get_prediction_results(model, dataloader):
    
    op = {}
    #set to eval
    model.eval()
    for data in dataloader:
        inputs, filename = data
        inputs = inputs.to(device)
        
        outputs = model.forward(inputs)
        
        _, p1 = torch.max(outputs[0].data, 1)
        _, p2 = torch.max(outputs[1].data, 1)
        
        for x,y,z in zip(p1.tolist(),p2.tolist(),filename):
            st = "V{}_C{}".format(int(x), int(y))
            print (z,x,y,st)
            op[z] = st
        
        del inputs,filename,outputs
        torch.cuda.empty_cache()

    return op       
model_load.to(device)

res_dict = get_prediction_results(model_load, test_loader)
import pandas as pd

res_df = pd.DataFrame({'ImageId':list(res_dict.keys()),'Class':list(res_dict.values())})
res_df.tail()

res_df.to_csv('submission.csv', index=False)


!ls