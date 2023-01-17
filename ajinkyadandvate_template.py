# All the required imports



import pandas as pd

import numpy as np

import os

import torch

import torchvision

from torchvision import transforms,datasets,models

from torch.utils.data import Dataset

import matplotlib.pyplot as plt

from torch import nn

import torch.nn.functional as F

from torch import optim

from skimage import io, transform

from torch.autograd import Variable



from PIL import Image



%matplotlib inline 
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

# Exploring train.csv file

df = pd.read_csv('../input/train.csv')

df.head()
#Dataset class



class ImageDataset(Dataset):

    



    def __init__(self, csv_file, root_dir, transform=None):

        """

        Args:

            csv_file (string): Path to the csv file with labels.

            root_dir (string): Directory with all the images.

            transform (callable, optional): Optional transform to be applied

                on a sample.

        """

        self.data_frame = pd.read_csv(csv_file)

        self.root_dir = root_dir

        self.transform = transform



    def __len__(self):

        return len(self.data_frame)



    def __getitem__(self, idx):

        img_name = os.path.join(self.root_dir, self.data_frame['Id'][idx])         # getting path of image

        image = Image.open(img_name).convert('RGB')                                # reading image and converting to rgb if it is grayscale

        label = np.array(self.data_frame['Category'][idx])                         # reading label of the image

        

        if self.transform:            

            image = self.transform(image)                                          # applying transforms, if any

        

        sample = (image, label)        

        return sample
# Transforms to be applied to each image (you can add more transforms), resizing every image to 3 x 224 x 224 size and converting to Tensor

transform = transforms.Compose([transforms.RandomResizedCrop(224),

                                transforms.RandomHorizontalFlip(p=0.3),

                                transforms.RandomRotation(degrees=(10.40), resample=False, expand=False, center=None),

                                transforms.ToTensor()                               

                                ])



trainset = ImageDataset(csv_file = '../input/train.csv', root_dir = '../input/data/data/', transform=transform)     #Training Dataset

trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=0)                     #Train loader, can change the batch_size to your own choice
# check if CUDA / GPU is available, if unavaiable then turn it on from the right side panel under SETTINGS, also turn on the Internet

train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:

    print('CUDA is not available.  Training on CPU ...')

else:

    print('CUDA is available!  Training on GPU ...')
# Reading sample_submission file to get the test image names

submission = pd.read_csv('../input/sample_sub.csv')

submission.head()
#Loading test data to make predictions



testset = ImageDataset(csv_file = '../input/sample_sub.csv', root_dir = '../input/data/data/', transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=0)
print("PyTorch Version: ",torch.__version__)

print("Torchvision Version: ",torchvision.__version__)
def featmodel():

    Model=models.resnet101(pretrained=True)

    #count=0

    #for child in Model.children():

        #count+=1

        #if count<8:

    for param in Model.parameters():

                param.require_grad=False



    num_feat=Model.fc.in_features

    Model.fc=nn.Sequential(nn.Linear(num_feat,512),nn.Dropout(0.5),nn.Linear(512,67))

    return Model



Model=featmodel()

Model.cuda()

param_to_update=[]



for name,param in Model.named_parameters():

    if param.requires_grad==True:

        param_to_update.append(param)



optimizer=optim.SGD(param_to_update,0.01,0.9)

criterion=nn.CrossEntropyLoss()
# Training Loop (You can write your own loop from scratch)

n_epochs = 7   #number of epochs, change this accordingly



for epoch in range(1, n_epochs+1):

    for i,(images,labels) in enumerate(trainloader):

        images=Variable(images.cuda())

        labels=Variable(labels.cuda())

        

        optimizer.zero_grad()

        

        outputs = Model(images)

        

        loss = criterion(outputs,labels)

        

        loss.backward()

        

        optimizer.step()

        

    print('loss{}'.format(loss.data))

        
optimizer2=optim.SGD(param_to_update,0.0001,0.9)
# Training Loop (You can write your own loop from scratch)

n_epochs = 21  #number of epochs, change this accordingly



for epoch in range(1, n_epochs+1):

    for i,(images,labels) in enumerate(trainloader):

        images=Variable(images.cuda())

        labels=Variable(labels.cuda())

        

        optimizer2.zero_grad()

        

        outputs = Model(images)

        

        loss = criterion(outputs,labels)

        

        loss.backward()

        

        optimizer2.step()

        

    print('loss{}'.format(loss.data))

        
optimizer3=optim.SGD(param_to_update,0.000001,0.9)
n_epochs = 5  #number of epochs, change this accordingly



for epoch in range(1, n_epochs+1):

    for i,(images,labels) in enumerate(trainloader):

        images=Variable(images.cuda())

        labels=Variable(labels.cuda())

        

        optimizer3.zero_grad()

        

        outputs = Model(images)

        

        loss = criterion(outputs,labels)

        

        loss.backward()

        

        optimizer3.step()

        

    print('loss{}'.format(loss.data))

        
#Exit training mode and set model to evaluation mode

Model.eval() # eval mode



total=0

correct=0



for images,labels in trainloader:

  #images=images.view(images.shape[0],-1)

  images=Variable(images.cuda())

  

  outputs=Model(images)

  _,predicted=torch.max(outputs.data,1)

  #print(labels)

  total+=labels.size(0)

  correct+=(predicted.cpu()==labels.cpu()).sum()

  

  

print("accuracy ")

print(float(100*correct/total))

# iterate over test data to make predictions



predictions=[]

for data, target in testloader:

    # move tensors to GPU if CUDA is available

    

   # if train_on_gpu:

    data, target = data.cuda(), target.cuda()

    # forward pass: compute predicted outputs by passing inputs to the model

    output = Model(data)

    _, pred = torch.max(output, 1)

    for i in range(len(pred)):

        predictions.append(int(pred[i]))

        



        



submission['Category'] = predictions             #Attaching predictions to submission file
predictions
print("The state dict keys: \n\n", Model.state_dict().keys())
checkpoint = {'model': featmodel(),

              'state_dict': Model.state_dict(),

              'optimizer' : optimizer.state_dict()}



torch.save(checkpoint, 'checkpoint.pth')


#saving submission file

submission.to_csv('submission.csv', index=False, encoding='utf-8')
# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = submission.to_csv()

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



# create a random sample dataframe

#df = pd.DataFrame(np.random.randn(50, 4), columns=list('ABCD'))



# create a link to download the dataframe

create_download_link(df)



# ↓ ↓ ↓  Yay, download link! ↓ ↓ ↓ 
def load_checkpoint(filepath):

    checkpoint = torch.load(filepath)

    model = checkpoint['model']

    model.load_state_dict(checkpoint['state_dict'])

    for parameter in model.parameters():

        parameter.requires_grad = False

    

    model.eval()

    

    return model
model = load_checkpoint('checkpoint.pth')

model.cuda()

print(model)