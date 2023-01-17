# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm



import warnings

warnings.filterwarnings('always')

warnings.filterwarnings('ignore')



# data visualisation and manipulation

import matplotlib.pyplot as plt

from matplotlib import style



import cv2



import random as rn



import os



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
def make_train_data(label,DIR):

    for img in tqdm(os.listdir(DIR)):

        try:

            path = os.path.join(DIR,img)

            img = cv2.imread(path,cv2.IMREAD_COLOR)

            img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))



            X.append(np.array(img))

            Z.append(str(label))

        except:

            pass

        

def assign_label(img,label):

    return label
X=[]

Z=[]

IMG_SIZE=250

normal='../input/pothole-detection-dataset/normal/'

potholes='../input/pothole-detection-dataset/potholes/'



make_train_data('NORMAL',normal)

make_train_data('POTHOLES',potholes)
fig,ax=plt.subplots(2,5)

plt.subplots_adjust(bottom=0.3, top=0.7, hspace=0)

fig.set_size_inches(15,15)



for i in range(2):

    for j in range (5):

        l=rn.randint(0,len(Z))

        ax[i,j].imshow(X[l][:,:,::-1])

        ax[i,j].set_title(Z[l])

        ax[i,j].set_aspect('equal')
from sklearn.preprocessing import LabelEncoder

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split



le=LabelEncoder()

Y=le.fit_transform(Z)

Y=to_categorical(Y,2)

print(Y)

X=np.array(X)

#X=X/255



x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=0)