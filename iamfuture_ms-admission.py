# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sn



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/Admission_Predict_Ver1.1.csv')

data.head()

#data['University Rating'].value_counts()
data = data.drop('Serial No.',axis=1)
sn.heatmap(data.corr(),annot=True,cmap='RdYlGn')

plt.figure(figsize=(10,10))

plt.show()
gre_above_315 = data.loc[data['GRE Score']>=315]

gre_above_315.head()
sn.countplot(data['University Rating'])

plt.title('University Ratings')

plt.show()
plt.title('Research for all students')

sn.countplot(data['Research'])

plt.show()

plt.title('Research for students above 315')

sn.countplot(gre_above_315['Research'])

plt.show()
plt.title('SOP for all students')

sn.countplot(data['SOP'])

plt.show()

plt.title('SOP for students above 315')

sn.countplot(gre_above_315['SOP'])

plt.show()
# LOR is written with a space, please type data.columns to see the names of columns written properly

plt.title('LOR for all students')

sn.countplot(data['LOR '])

plt.show()

plt.title('LOR for students above 315')

sn.countplot(gre_above_315['LOR '])

plt.show()
# kaggle kernel has no use of writing figsize. The output figure size remains the same in the output. 

plt.title('TOEFL for all students')

sn.countplot(data['TOEFL Score'])

plt.figure(figsize=(14,10))

plt.show()

plt.title('TOEFL for students above 315')

sn.countplot(gre_above_315['TOEFL Score'])

plt.figure(figsize=(14,10))

plt.show()
plt.title('CGPA for all students')

sn.countplot(data['CGPA'])

plt.figure(figsize=(14,10))

plt.show()

plt.title('CGPA for students above 315')

sn.countplot(gre_above_315['CGPA'])

plt.figure(figsize=(14,10))

plt.show()
plt.title('University applied by all students')

sn.countplot(data['University Rating'])

plt.figure(figsize=(14,10))

plt.show()

plt.title('University applied by students above 315')

sn.countplot(gre_above_315['University Rating'])

plt.figure(figsize=(14,10))

plt.show()
data.columns
from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn import metrics



features = data.iloc[:,:7]

labels = data.iloc[:,-1]
# CONSIDERING THE MOST HIGHLY OUTCOMES SO THAT MAXIMUM ARE SELECTED FOR MS

labels[labels > 0.6] = 1

labels[labels < 0.6] = 0

labels.head()
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score, confusion_matrix,classification_report



X_train, X_test, Y_train, Y_test = train_test_split(features,labels,test_size=0.35,random_state=0)
print(X_train.shape)

print(Y_train.shape)

print(X_test.shape)

print(Y_test.shape)
clf = LogisticRegression(solver='lbfgs')

clf.fit(X_train,Y_train)

y_pred = clf.predict(X_test)

print('Training accuracy Score:',np.mean(Y_test == y_pred)*100)

data_verify = pd.read_csv('../input/Admission_Predict.csv')

data_verify.head()
X_t = data_verify.iloc[:,:7]

Y_t = data_verify.iloc[:,-1]

Y_t[Y_t > 0.6] = 1

Y_t[Y_t < 0.6] = 0

y_test_pred = clf.predict(X_t)

print('Test Accuracy:',np.mean(Y_t == y_test_pred)*100)