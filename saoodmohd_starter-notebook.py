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
os.listdir('../input/handwritten-digit-recognition-pitc')
X_train = np.load('../input/handwritten-digit-recognition-pitc/X_train.npy')

X_test = np.load('../input/handwritten-digit-recognition-pitc/X_test.npy')

y_train = np.load('../input/handwritten-digit-recognition-pitc/y_train.npy')



print(X_train.shape)

print(X_test.shape)

print(y_train.shape)



for i in range(X_test.shape[0]):

    if(X_test[i][:14,:14].mean() < 255/2):

        X_test[i][:14,:14] = 255 - X_test[i][:14,:14]



    if(X_test[i][14:,:14].mean() < 255/2):

        X_test[i][14:,:14] = 255 - X_test[i][14:,:14]



    if(X_test[i][:14,14:].mean() < 255/2):

        X_test[i][:14,14:] = 255 - X_test[i][:14,14:]



    if(X_test[i][14:,14:].mean() < 255/2):

        X_test[i][14:,14:] = 255 - X_test[i][14:,14:]

import matplotlib.pyplot as plt    

plt.imshow(X_test[56],cmap='gray')
pred = np.zeros((X_test.shape[0]),dtype=np.int)



df = pd.DataFrame(columns = ["Id","Predictions"])

df["Predictions"] = pred

df["Id"] = np.arange(len(pred))



df.to_csv('out.csv',index=None)