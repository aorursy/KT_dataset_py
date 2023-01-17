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
!pip install albumentations > /dev/null 2>&1
!pip install pretrainedmodels > /dev/null 2>&1
import albumentations

import pretrainedmodels

from tqdm.notebook import tqdm
model_resnet18 = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)

model_resnet34 = torch.hub.load('pytorch/vision', 'resnet34', pretrained=True)
for name, param in model_resnet18.named_parameters():

    if("bn" not in name):

        param.requires_grad = False

        

for name, param in model_resnet34.named_parameters():

    if("bn" not in name):

        param.requires_grad = False
num_classes = 2



model_resnet18.fc = nn.Sequential(nn.Linear(model_resnet18.fc.in_features,512),

                                  nn.ReLU(),

                                  nn.Dropout(),

                                  nn.Linear(512, num_classes))



model_resnet34.fc = nn.Sequential(nn.Linear(model_resnet34.fc.in_features,512),

                                  nn.ReLU(),

                                  nn.Dropout(),

                                  nn.Linear(512, num_classes))
def train(model, optimizer, loss_fn, train_loader, val_loader, epochs=5, device="cpu"):

    for epoch in range(epochs):

        training_loss = 0.0

        valid_loss = 0.0

        model.train()

        for batch in train_loader:

            optimizer.zero_grad()

            inputs, targets = batch

            inputs = inputs.to(device)

            targets = targets.to(device)

            output = model(inputs)

            loss = loss_fn(output, targets)

            loss.backward()

            optimizer.step()

            training_loss += loss.data.item() * inputs.size(0)

        training_loss /= len(train_loader.dataset)

        

        model.eval()

        num_correct = 0 

        num_examples = 0

        for batch in val_loader:

            inputs, targets = batch

            inputs = inputs.to(device)

            output = model(inputs)

            targets = targets.to(device)

            loss = loss_fn(output,targets) 

            valid_loss += loss.data.item() * inputs.size(0)

                        

            correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], targets).view(-1)

            num_correct += torch.sum(correct).item()

            num_examples += correct.shape[0]

        valid_loss /= len(val_loader.dataset)



        print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}, accuracy = {:.4f}'.format(epoch, training_loss,

        valid_loss, num_correct / num_examples))

## read the csv data files

train_df = pd.read_csv('../input/janatahack-av-computervision/train_SOaYf6m/train.csv')

test_df = pd.read_csv('../input/janatahack-av-computervision/test_vc2kHdQ.csv')

submit = pd.read_csv('../input/janatahack-av-computervision/sample_submission_yxjOnvz.csv')
train_df.shape, test_df.shape
## set the data folder

data_folder = Path("../input/janatahack-av-computervision")

data_path = "../input/janatahack-av-computervision/train_SOaYf6m/images/"



path = os.path.join(data_path , "*jpg")
files = glob.glob(path)

data=[]

for file in files:

    image = cv2.imread(file)

    data.append(image)
data_path
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
batch_size=32

img_dimensions = 224



# Normalize to the ImageNet mean and standard deviation

# Could calculate it for the cats/dogs data set, but the ImageNet

# values give acceptable results here.

img_transforms = transforms.Compose([

    transforms.Resize((img_dimensions, img_dimensions)),

    transforms.ToTensor(),

    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225] )

    ])



img_test_transforms = transforms.Compose([

    transforms.Resize((img_dimensions,img_dimensions)),

    transforms.ToTensor(),

    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225] )

    ])



def check_image(path):

    try:

        im = Image.open(path)

        return True

    except:

        return False

TRAIN_CSV = '../input/janatahack-av-computervision/train_SOaYf6m/train.csv'

transform = transforms.Compose([

    transforms.Resize((img_dimensions, img_dimensions)),

    transforms.ToTensor(),

    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225] )

    ])

dataset = EmergencyDataset(TRAIN_CSV, data_path, transform=transform)
torch.manual_seed(10)



val_pct = 0.2

val_size = int(val_pct * len(dataset))

train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

len(train_ds), len(val_ds)
batch_size = 32
def show_batch(dl, invert=True):

    for images, labels in dl:

        fig, ax = plt.subplots(figsize=(16, 8))

        ax.set_xticks([]); ax.set_yticks([])

        data = 1-images if invert else images

        ax.imshow(make_grid(data, nrow=16).permute(1, 2, 0))

        break
train_data_loader  = DataLoader(train_ds, batch_size, shuffle=True, num_workers=2, pin_memory=True)

validation_data_loader = DataLoader(val_ds, batch_size*2, num_workers=2, pin_memory=True)





TEST_CSV = '../input/janatahack-av-computervision/sample_submission_yxjOnvz.csv'

test_dataset = EmergencyDataset(TEST_CSV, data_path, transform=transform)

test_data_loader = DataLoader(test_dataset, batch_size, num_workers=2, pin_memory=True)
if torch.cuda.is_available():

    device = torch.device("cuda") 

else:

    device = torch.device("cpu")
print(device)
print(f'Num training images: {len(train_data_loader.dataset)}')

print(f'Num validation images: {len(validation_data_loader.dataset)}')

print(f'Num test images: {len(test_data_loader.dataset)}')
def valid_model(model):

    correct = 0

    total = 0

    with torch.no_grad():

        for data in validation_data_loader:

            images, labels = data[0].to(device), data[1].to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    print('correct: {:d}  total: {:d}'.format(correct, total))

    print('accuracy = {:f}'.format(correct / total))
model_resnet18.to(device)

optimizer = optim.Adam(model_resnet18.parameters(), lr=0.001)

train(model_resnet18, optimizer, torch.nn.CrossEntropyLoss(), train_data_loader, validation_data_loader, epochs=10, device=device)
valid_model(model_resnet18)
@torch.no_grad()

def predict_dl(dl, model):    

#     torch.cuda.empty_cache()

    batch_probs = []

    for xb, _ in tqdm(dl):

        probs = model(xb)        

        batch_probs.append(probs.cpu().detach())

    batch_probs = torch.cat(batch_probs)



    return [x.numpy() for x in batch_probs]

 
submission_df = pd.read_csv(TEST_CSV)

test_preds = predict_dl(test_data_loader, model_resnet18)

submission_df.emergency_or_not = np.argmax(test_preds, axis = 1)

submission_df.head()
submission_df['emergency_or_not'].value_counts()
submission_df.to_csv('submission.csv', index=False)
submission_df.head()
# import the modules we'll need

from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv()

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)





# create a link to download the dataframe

create_download_link(submission_df)
model_resnet34.to(device)

optimizer = optim.Adam(model_resnet34.parameters(), lr=0.001)

train(model_resnet34, optimizer, torch.nn.CrossEntropyLoss(), train_data_loader, validation_data_loader, epochs=10, device=device)
valid_model(model_resnet34)
data_path
import os

def find_classes(dir):

    classes = os.listdir(dir)

    classes.sort()

    class_to_idx = {classes[i]: i for i in range(len(classes))}

    return classes, class_to_idx



def make_prediction(model, filename):

    labels, _ = find_classes(data_path)

    img = Image.open(filename)

    img = img_test_transforms(img)

    img = img.unsqueeze(0)

    prediction = model(img.to(device))

    prediction = prediction.argmax()

    print(labels[prediction])

    

make_prediction(model_resnet34, '/kaggle/input/janatahack-av-computervision/train_SOaYf6m/images/2345.jpg')

make_prediction(model_resnet34, '/kaggle/input/janatahack-av-computervision/train_SOaYf6m/images/438.jpg')
torch.save(model_resnet18.state_dict(), "./model_resnet18.pth")

torch.save(model_resnet34.state_dict(), "./model_resnet34.pth")



# Remember that you must call model.eval() to set dropout and batch normalization layers to

# evaluation mode before running inference. Failing to do this will yield inconsistent inference results.



resnet18 = torch.hub.load('pytorch/vision', 'resnet18')

resnet18.fc = nn.Sequential(nn.Linear(resnet18.fc.in_features,512),nn.ReLU(), nn.Dropout(), nn.Linear(512, num_classes))

resnet18.load_state_dict(torch.load('./model_resnet18.pth'))

resnet18.eval()



resnet34 = torch.hub.load('pytorch/vision', 'resnet34')

resnet34.fc = nn.Sequential(nn.Linear(resnet34.fc.in_features,512),nn.ReLU(), nn.Dropout(), nn.Linear(512, num_classes))

resnet34.load_state_dict(torch.load('./model_resnet34.pth'))

resnet34.eval()
# Test against the average of each prediction from the two models

models_ensemble = [resnet18.to(device), resnet34.to(device)]

correct = 0

total = 0

with torch.no_grad():

    for data in validation_data_loader:

        images, labels = data[0].to(device), data[1].to(device)

        predictions = [i(images).data for i in models_ensemble]

        avg_predictions = torch.mean(torch.stack(predictions), dim=0)

        _, predicted = torch.max(avg_predictions, 1)



        total += labels.size(0)

        correct += (predicted == labels).sum().item()

        

print('accuracy = {:f}'.format(correct / total))

print('correct: {:d}  total: {:d}'.format(correct, total))
# Test against the average of each prediction from the two models

models_ensemble = [resnet18.to(device), resnet34.to(device)]

correct = 0

total = 0

with torch.no_grad():

    for data in test_data_loader:

        images = data[0].to(device)

        predictions = [i(images).data for i in models_ensemble]

        avg_predictions = torch.mean(torch.stack(predictions), dim=0)

        _, predicted = torch.max(avg_predictions, 1)
predicted
# Project name used for jovian.commit

# project_name = '01-Emergency_Vehicle_Detection_using_Pretrained_Models'
# !pip install jovian --upgrade --quiet
# import jovian
# Clear previously recorded hyperparams & metrics

# jovian.reset()
# torch.save(model_resnet18.state_dict(), "model_resnet18.pth")

# torch.save(model_resnet34.state_dict(), "model_resnet34.pth")

# jovian.commit(project=project_name, environment=None, 

#               outputs=['model_resnet18.pth', 'model_resnet34.pth'])