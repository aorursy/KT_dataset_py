# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
pd.read_csv('../input/train.csv')
label=train['label']
label
train.drop('label',axis=1,inplace=True)
import sklearn as sk
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(train,label,test_size=0.2)
from sklearn.linear_model import LogisticRegression
gln=LogisticRegression(n_jobs=-1,solver='lbfgs')
gln.fit(x_train,y_train)
import matplotlib.pyplot as plt
pred=x_train.iloc[0,:]
pred=np.array(pred)
gln.predict(pred.reshape(1,-1))
plt.imshow(pred.reshape(28,28))