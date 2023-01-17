# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))    



# Any results you write to the current directory are saved as output.
import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import torch

import torchvision

from PIL import Image

from torch.utils.data import Dataset, DataLoader, random_split

from torchvision import transforms
submission = pd.read_csv('../input/padhai-hindi-vowel-consonant-classification/sample_submission.csv')

submission.head()
submission.shape
from PIL import Image

import matplotlib.pyplot as plt

with open('/kaggle/input/padhai-hindi-vowel-consonant-classification/test/test/1609.png', 'rb') as file:

    img=Image.open(file)

    plt.axis('off')

    plt.imshow(img)

    print(img.size)



print(img.format)
files = os.listdir("/kaggle/input/padhai-hindi-vowel-consonant-classification/train/train")
label = files[0].split('_')

print(label)

label = [int(label[0][1]), int(label[1][1])]

print(label)
class ImageDataset(Dataset):

    def __init__(self, file, transform = None, train = True):

        self.file = file

        self.files = os.listdir(file)

        self.transform = transform

        self.train = train

        

    def __len__(self):

        return len(os.listdir(self.file))

    

    def __getitem__(self, indx):

        img_name = self.file + self.files[indx]

        img = Image.open(img_name).convert('RGB')

                

        if self.transform:

            img = self.transform(img)

            

        if self.train == True:

            label = self.files[indx].split('_')

            label = [int(label[0][1]), int(label[1][1])]

            label = np.array(label)

            return img, label

        else:

            return img, self.files[indx] 

            print(self.files[indx])
transform = transforms.Compose([transforms.Resize((224, 224)),

                                transforms.ToTensor(), 

                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])    
train_data = ImageDataset("/kaggle/input/padhai-hindi-vowel-consonant-classification/train/train/", transform = transform, train = True)

test_data = ImageDataset("/kaggle/input/padhai-hindi-vowel-consonant-classification/test/test/", transform = transform, train = False)
len(train_data), len(test_data)

train_data[0]
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)
for data in train_loader:

    img, label = data

    print(img.shape)

    image = np.transpose(img[1].numpy(), (1,2,0))

    print(image.shape)

    plt.imshow(image)

    break
import torch.nn as nn

import torch.optim as optim

import torchvision.models as models

from tqdm import tqdm_notebook
model_V = models.resnet18(pretrained=True)

model_C = models.resnet18(pretrained=True)
for param in model_V.parameters():

    param.requires_grad = False

for param in model_C.parameters():

    param.requires_grad = False
model_V.fc = nn.Linear(512, 10)

model_C.fc = nn.Linear(512, 10)
device = torch.device("cuda:0" if torch.cuda.is_available else 'cpu')
model_V.to(device)

model_C.to(device)

loss_fn = nn.CrossEntropyLoss()

opt_V = optim.Adam(model_V.parameters())

opt_C = optim.Adam(model_C.parameters())
for i in tqdm_notebook(range(8)):

    for data in tqdm_notebook(train_loader):

        image, label = data

        image, label = image.to(device), label.to(device)

        

        opt_V.zero_grad()

        opt_C.zero_grad()

        out_V = model_V(image)

        out_C = model_C(image)

        loss_V = loss_fn(out_V, label[:,0])

        loss_C = loss_fn(out_C, label[:,1])

        loss_V.backward()

        loss_C.backward()

        opt_V.step()

        opt_C.step()

torch.save(model_V.state_dict(), "best_model_V.pth")

torch.save(model_C.state_dict(), "best_model_C.pth")

del image, label, out_V, out_C

torch.cuda.empty_cache()
model_V.load_state_dict(torch.load("best_model_V.pth"))

model_V.to(device)

model_V.eval()



model_C.load_state_dict(torch.load("best_model_C.pth"))

model_C.to(device)

model_C.eval()
outputs = []

for test_img, test_filename in tqdm_notebook(test_loader):

        test_img = test_img.to(device)

        output_V = model_V(test_img)

        output_C = model_C(test_img)

        num, ind_V = torch.max(output_V, 1)

        num, ind_C = torch.max(output_C, 1)

        output_V =  ind_V.cpu().tolist()

        output_C =  ind_C.cpu().tolist()

        for i,j in zip(output_V, output_C):

         outputs.append("V" + str(i) + '_'+ "C" + str(j))

        

        
len(outputs)
submission = pd.DataFrame({"ImageId": test_filename, "Class": outputs})

submission.head()
submission.to_csv( 'submission.csv', index = False)