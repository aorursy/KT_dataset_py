import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # for drawing

import seaborn as sns # for drawing as well

from scipy.optimize import minimize # will use for training

from sklearn.metrics import accuracy_score, f1_score # some metrics

from functools import partial # better google what it is

from tqdm import tqdm as tqdm # nice trackbar

import cv2



import torch # this is the best library for neural networks (IMHO)

import torch.nn as nn # high-level API, useful for deep learning



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

os.listdir('/kaggle/input/digit-recognizer')
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')



train.head()
target = train['label']



train = train.drop('label', axis=1).values.reshape([-1, 28, 28]) / 255

test = test.values.reshape([-1, 28, 28]) / 255
for i in range(4):

    plt.title(target[i])

    plt.imshow(train[i], cmap='gray')

    plt.show()
!pip install efficientnet-pytorch
from efficientnet_pytorch import EfficientNet
effnet = EfficientNet.from_pretrained('efficientnet-b0')
img = train[0]

img = cv2.resize(img, (224, 224)) # min 64x64, max inf

img = np.tile(img[None], [3, 1, 1])

img = img[None] # Add batch dimension

img = torch.tensor(img, dtype=torch.float32)

img.shape
feats = effnet.extract_features(img)
feats.shape
def extract_features(img):

    '''Input: grayscale image'''

    

    # Minimum size required by EfficientNet

    img = cv2.resize(img, (64, 64))

    # Convert grayscale to RGB

    img = np.tile(img[None], [3, 1, 1])

    # Add batch dimension

    img = img[None]

    # Convert to tensor

    img = torch.tensor(img, dtype=torch.float32, device=device)

    

    # Pass image through the neural network

    feats = effnet.extract_features(img)

    

    # Get rid of spatial dimensions

    feats = feats[0].mean(1).mean(1)

    # Convert to numpy

    feats = feats.data.cpu().numpy()

    # float16 takes less memory than float32 (this is optional)

    feats = feats.astype('float16')

    

    return feats
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(device)



effnet = effnet.to(device)
feats_train = np.array([extract_features(img) for img in tqdm(train)])

feats_test = np.array([extract_features(img) for img in tqdm(test)])
print('train', feats_train.shape)

print('test', feats_test.shape)
from sklearn.linear_model import LogisticRegression



lr = LogisticRegression()

lr.fit(feats_train, target)



predictions = lr.predict(feats_test)
submission = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')

submission.Label = predictions



# Save it as csv file

submission.to_csv('submission_10n.csv', index=False)



submission.head()