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
filePath1='../input/tensor-matrix/Tensor.npy'

Tensor=np.load(filePath1)
print(Tensor.shape)
filePath2='../input/news-content-matrix/newsContent.csv'

newsContent=pd.read_csv(filePath2,encoding='utf-8',header=-1)
print(newsContent.shape)
!pip install tensorly
import tensorly
T=tensorly.unfold(Tensor,0)
print(T.shape)
X=np.concatenate((T,newsContent),axis=1)
Y=np.ones(182)

Y[91:182]=0
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.30,random_state=0,shuffle=True)
from xgboost import XGBClassifier

classifier = XGBClassifier()

classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, Y_pred)

print(cm)