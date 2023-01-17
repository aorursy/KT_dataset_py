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
import torch.nn as nn

import torchvision



class Task5Model(nn.Module):



    def __init__(self, num_classes):



        super().__init__()



        self.bw2col = nn.Sequential(

            nn.BatchNorm2d(3),

            nn.Conv2d(3, 10, 3, padding=0), nn.ReLU(),

            nn.Conv2d(10, 3, 3, padding=0), nn.ReLU())



        self.mv2 = torchvision.models.mobilenet_v2(pretrained=True)



        self.final = nn.Sequential(

            nn.Linear(1280, 512), nn.ReLU(), nn.BatchNorm1d(512),

            nn.Linear(512, num_classes))



    def forward(self, x):

        x = self.bw2col(x)

        x = self.mv2.features(x)

        x = x.max(dim=-1)[0].max(dim=-1)[0]

        x = self.final(x)

        return x



m = Task5Model(264)
import torch

m.load_state_dict(torch.load('../input/cornell-mobnet/best_model.pth',map_location=torch.device('cpu')))
torch.save(m,'MobileNet.pth')
import pickle

with open('mobpickle.pkl','wb') as f:

    pickle.dump(m,f)
with open('mobpickle.pkl','rb') as f:

    m2 = pickle.load(f)
m2
import torchvision

mobclass = torchvision.models.mobilenet_v2(pretrained=True)
import pickle

with open('mobnet_torch.pkl','wb') as f:

    pickle.dump(mobclass,f)

mobclass