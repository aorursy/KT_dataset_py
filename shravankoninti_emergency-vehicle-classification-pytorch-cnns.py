# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# libraries

import numpy as np

import pandas as pd

import seaborn as sns

import os

import cv2

import matplotlib.pyplot as plt

%matplotlib inline



import torch

import torchvision

import numpy as np

import matplotlib.pyplot as plt

import torch.nn as nn

import torch.nn.functional as F

from torchvision.datasets import CIFAR10

from torchvision.transforms import ToTensor

from torchvision.utils import make_grid

from torch.utils.data.dataloader import DataLoader

from torch.utils.data import random_split

%matplotlib inline



from pathlib import Path

import os

import cv2

import glob

import torchvision

import torchvision.transforms as transforms

from torch.utils.data.sampler import SubsetRandomSampler

from torch.utils.data import Dataset

from PIL import Image

import torchvision.transforms as transforms

import torch

import torchvision.transforms.functional as F

import torch.nn.functional as F

import torch.optim as optim

from tqdm.notebook import tqdm
!pip install albumentations > /dev/null 2>&1
!pip install pretrainedmodels > /dev/null 2>&1
import albumentations

import pretrainedmodels
## read the csv data files

train_df = pd.read_csv('../input/janatahack-av-computervision/train_SOaYf6m/train.csv')

test_df = pd.read_csv('../input/janatahack-av-computervision/test_vc2kHdQ.csv')

submit = pd.read_csv('../input/janatahack-av-computervision/sample_submission_yxjOnvz.csv')
train_df.shape, test_df.shape
train_df.groupby('emergency_or_not').count()
sns.countplot(x='emergency_or_not' , data=train_df)
## set the data folder

data_folder = Path("../input/janatahack-av-computervision")

data_path = "../input/janatahack-av-computervision/train_SOaYf6m/images/"



path = os.path.join(data_path , "*jpg")
data_path
files = glob.glob(path)

data=[]

for file in files:

    image = cv2.imread(file)

    data.append(image)
train_images = data[:1646]

test_images= data[1646:]
#Look at the shape of the image

print(train_images[0].shape), print(train_images[100].shape)
def get_images_class(cat):

    list_of_images = []

    fetch = train_df.loc[train_df['emergency_or_not']== cat][:3].reset_index()

    for i in range(0,len(fetch['image_names'])):

        list_of_images.append(fetch['image_names'][i])

    return list_of_images 
get_images_class(0)
get_images_class(1)
import matplotlib.pyplot as plt

import matplotlib.image as mpimg

fig = plt.figure(figsize=(20,15))

for i, image_name in enumerate(get_images_class(0)):

    plt.subplot(1,3 ,i+1)

    img=mpimg.imread('../input/janatahack-av-computervision/train_SOaYf6m/images/'+image_name)

    imgplot = plt.imshow(img)

    plt.xlabel(str("Non-Emergency Vehicle") + " (Index:" +str(i+1)+")" )

plt.show()
import matplotlib.pyplot as plt

import matplotlib.image as mpimg

fig = plt.figure(figsize=(20,15))

for i, image_name in enumerate(get_images_class(1)):

    plt.subplot(1,3 ,i+1)

    img=mpimg.imread('../input/janatahack-av-computervision/train_SOaYf6m/images/'+image_name)

    imgplot = plt.imshow(img)

    plt.xlabel(str("Emergency Vehicle") + " (Index:" +str(i)+")" )

plt.show()
class EmergencyDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):

        self.df = pd.read_csv(csv_file)

        self.transform = transform

        self.root_dir = root_dir

        

    def __len__(self):

        return len(self.df)    

    

    def __getitem__(self, idx):

        row = self.df.loc[idx]

        img_id, img_label = row['image_names'], row['emergency_or_not']

        img_fname = self.root_dir + str(img_id)

#         + ".jpg"

        img = Image.open(img_fname)

        if self.transform:

            img = self.transform(img)

        return img, img_label
TRAIN_CSV = '../input/janatahack-av-computervision/train_SOaYf6m/train.csv'

transform = transforms.Compose([transforms.ToTensor()])

dataset = EmergencyDataset(TRAIN_CSV, data_path, transform=transform)
torch.manual_seed(10)



val_pct = 0.2

val_size = int(val_pct * len(dataset))

train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

len(train_ds), len(val_ds)
batch_size = 32
transform = transforms.Compose(

    [transforms.ToTensor(),

     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataset = EmergencyDataset(TRAIN_CSV, data_path, transform=transform)
train_loader  = DataLoader(train_ds, batch_size, shuffle=True, num_workers=2, pin_memory=True)

validation_loader = DataLoader(val_ds, batch_size*2, num_workers=2, pin_memory=True)
class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        # 3 input image channel, 16 output channels, 3x3 square convolution kernel

        self.conv1 = nn.Conv2d(3,16,kernel_size=3,stride=2,padding=1)

        self.conv2 = nn.Conv2d(16, 32,kernel_size=3,stride=2, padding=1)

        self.conv3 = nn.Conv2d(32, 64,kernel_size=3,stride=2, padding=1)

        self.conv4 = nn.Conv2d(64, 64,kernel_size=3,stride=2, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.dropout = nn.Dropout2d(0.4)

        self.batchnorm1 = nn.BatchNorm2d(16)

        self.batchnorm2 = nn.BatchNorm2d(32)

        self.batchnorm3 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(64*4*4,512)

        self.fc2 = nn.Linear(512, 256)

        self.fc3 = nn.Linear(256, 2)

        self.sig = nn.Sigmoid()

        

        

    def forward(self, x):



        x = self.batchnorm1(F.relu(self.conv1(x)))

        x = self.batchnorm2(F.relu(self.conv2(x)))

        x = self.dropout(self.batchnorm2(self.pool(x)))

        x = self.batchnorm3(self.pool(F.relu(self.conv3(x))))

        x = self.dropout(self.conv4(x))



        x = x.view(x.size(0), -1)





        x = self.dropout(self.fc1(x))

        x = self.dropout(self.fc2(x))

        x = self.sig(self.fc3(x))

        return x
model = Net() # On CPU

#model = Net().to(device)  # On GPU

print(model)
criterion = nn.CrossEntropyLoss()

# criterion = nn.BCELoss()

optimizer = optim.Adam(model.parameters(), lr=0.0001)
def accuracy(out, labels):

    _,pred = torch.max(out, dim=1)

    return torch.sum(pred==labels).item()
n_epochs = 20

print_every = 10

valid_loss_min = np.Inf

val_loss = []

val_acc = []

train_loss = []

train_acc = []

total_step = len(train_loader)

for epoch in range(1, n_epochs+1):

    running_loss = 0.0

    # scheduler.step(epoch)

    correct = 0

    total=0

    print(f'Epoch {epoch}\n')

    for batch_idx, (data_, target_) in enumerate(train_loader):

        #data_, target_ = data_.to(device), target_.to(device)# on GPU

        # zero the parameter gradients

        optimizer.zero_grad()

        # forward + backward + optimize

        outputs = model(data_)

        loss = criterion(outputs, target_)

        loss.backward()

        optimizer.step()

        # print statistics

        running_loss += loss.item()

        _,pred = torch.max(outputs, dim=1)

        correct += torch.sum(pred==target_).item()

        total += target_.size(0)

        if (batch_idx) % 20 == 0:

            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 

                   .format(epoch, n_epochs, batch_idx, total_step, loss.item()))

    train_acc.append(100 * correct / total)

    train_loss.append(running_loss/total_step)

    print(f'\ntrain loss: {np.mean(train_loss):.4f}, train acc: {(100 * correct / total):.4f}')

    batch_loss = 0

    total_t=0

    correct_t=0

    with torch.no_grad():

        model.eval()

        for data_t, target_t in (validation_loader):

            #data_t, target_t = data_t.to(device), target_t.to(device)# on GPU

            outputs_t = model(data_t)

            loss_t = criterion(outputs_t, target_t)

            batch_loss += loss_t.item()

            _,pred_t = torch.max(outputs_t, dim=1)

            correct_t += torch.sum(pred_t==target_t).item()

            total_t += target_t.size(0)

        val_acc.append(100 * correct_t / total_t)

        val_loss.append(batch_loss/len(validation_loader))

        network_learned = batch_loss < valid_loss_min

        print(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t / total_t):.4f}\n')

        # Saving the best weight 

        if network_learned:

            valid_loss_min = batch_loss

            torch.save(model.state_dict(), 'model_classification.pt')

            print('Detected network improvement, saving current model')

    model.train()
fig = plt.figure(figsize=(20,10))

plt.title("Train - Validation Loss")

plt.plot( train_loss, label='train')

plt.plot( val_loss, label='validation')

plt.xlabel('num_epochs', fontsize=12)

plt.ylabel('loss', fontsize=12)

plt.legend(loc='best')
fig = plt.figure(figsize=(20,10))

plt.title("Train - Validation Accuracy")

plt.plot(train_acc, label='train')

plt.plot(val_acc, label='validation')

plt.xlabel('num_epochs', fontsize=12)

plt.ylabel('accuracy', fontsize=12)

plt.legend(loc='best')
# Importing trained Network with better loss of validation

model.load_state_dict(torch.load('model_classification.pt'))
def img_display(img):

    img = img / 2 + 0.5     # unnormalize

    npimg = img.numpy()

    npimg = np.transpose(npimg, (1, 2, 0))

    return npimg
dataiter = iter(validation_loader)

images, labels = dataiter.next()

vehicle_types = {0: 'Non-Emergency-Vehicle', 1: 'Emergency-Vehicle'}

# Viewing data examples used for training

fig, axis = plt.subplots(3, 5, figsize=(20, 15))

with torch.no_grad():

    model.eval()

    for ax, image, label in zip(axis.flat,images, labels):

        ax.imshow(img_display(image)) # add image

        image_tensor = image.unsqueeze_(0)

        output_ = model(image_tensor)

        output_ = output_.argmax()

        k = output_.item()==label.item()

        ax.set_title(str(vehicle_types[label.item()])+":" +str(k)) # add label

TEST_CSV = '../input/janatahack-av-computervision/sample_submission_yxjOnvz.csv'

test_dataset = EmergencyDataset(TEST_CSV, data_path, transform=transform)
len(test_dataset)
test_dl = DataLoader(test_dataset, batch_size, num_workers=2, pin_memory=True)
test_dl
@torch.no_grad()

def predict_dl(dl, model):    

    torch.cuda.empty_cache()

    batch_probs = []

    for xb, _ in tqdm(dl):

        probs = model(xb)        

        batch_probs.append(probs.cpu().detach())

    batch_probs = torch.cat(batch_probs)



    return [x.numpy() for x in batch_probs]

 
submission_df = pd.read_csv(TEST_CSV)

test_preds = predict_dl(test_dl, model)

submission_df.emergency_or_not = np.argmax(test_preds, axis = 1)

submission_df.head()
submission_df.tail()
submission_df['emergency_or_not'].value_counts()
submission_df.to_csv('submission.csv', index=False)
model.eval()



preds = []

for batch_i, (data, target) in enumerate(test_dl):

#     data, target = data.cuda(), target.cuda()

    output = model(data)



    pr = output[:,1].detach().cpu().numpy()

    for i in pr:

        preds.append(i)

submission_df['emergency_or_not'] = submission_df['emergency_or_not'].apply(lambda x: 1 if x >= 0.5 else 0)

submission_df.to_csv('submission_output.csv', index=False)
submission_df.head()
submission_df['emergency_or_not'].value_counts()
print(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t / total_t):.4f}\n')

        # Saving the best weight 
## function to create download link

from IPython.display import HTML

def create_download_link(title = "Download CSV file", filename = "submission_output.csv"):  

    html = '<a href={filename}>{title}</a>'

    html = html.format(title=title,filename=filename)

    return HTML(html)
create_download_link(filename = 'submission_output.csv')
arch = (512,256,2)
lrs = [0.0001]
epochs = [20]
validation_loss =  np.mean(val_loss)

validation_acc =  (100 * correct_t / total_t)

print(validation_loss, validation_acc)
# # Project name used for jovian.commit

# project_name = '01-Emergency_Vehicle_Detection'
# !pip install jovian --upgrade --quiet
# import jovian
# # Clear previously recorded hyperparams & metrics

# jovian.reset()

# jovian.log_hyperparams(arch=arch, 

#                        lrs=lrs, 

#                        epochs=epochs)

# jovian.log_metrics(test_loss=validation_loss, test_acc=validation_acc)
# torch.save(model.state_dict(), '01-emergency-vehicle-detector.pth')

# jovian.commit(project=project_name, environment=None, outputs=['01-emergency-vehicle-detector.pth'])