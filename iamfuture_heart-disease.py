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
data = pd.read_csv('../input/heart.csv')

data.head()
import matplotlib.pyplot as plt

import seaborn as sn

label_target = ['Yes','No']

li_target = data.target.value_counts()



sn.barplot(label_target,li_target)

plt.show()
data.columns
label_s = ['Male','Female']

li_s = data.sex.value_counts()



sn.barplot(label_s,li_s)

plt.show()
plt.figure(figsize=(10,18))

sn.heatmap(data.corr(),annot=True,cmap='RdYlBu')

plt.show()
features = data.iloc[:,:12]

label = data.iloc[:,-1]
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split



X_train,X_test,Y_train,Y_test = train_test_split(features,label,test_size=0.15,random_state=4)

clf = LogisticRegression(solver='lbfgs')

clf.fit(X_train,Y_train)

y_pred = clf.predict(X_test)

print('Accuracy Score:',np.mean(y_pred == Y_test)*100)
from sklearn.metrics import confusion_matrix

sn.heatmap(confusion_matrix(Y_test,y_pred),annot=True,cmap='RdYlBu')

plt.show()