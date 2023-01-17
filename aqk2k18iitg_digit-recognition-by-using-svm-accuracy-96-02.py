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
#importing library which is giong to call

import matplotlib.pyplot as pt#for plotting

import seaborn as sns

from sklearn.svm import SVC #suport vector mechines for classifing data with maximum margin

from sklearn.model_selection import train_test_split #for splitting data in to train and test set

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score



# importing train data

data=pd.read_csv("../input/digit-recognizer/train.csv")

#importing test data

data_test=pd.read_csv("../input/digit-recognizer/test.csv")

data.head()
Data=data.to_numpy()

X=Data[:,1:]

y=d=Data[:,0]

np.unique(np.isnan(X))
#splitting data for trainig and cross validation

xtrain, xtest, ytrain, ytest= train_test_split(X, y, test_size=0.25, random_state=1)

sns.countplot(ytrain) #ploting classes counts

#data normalization

sc_X = StandardScaler()

xtrain = sc_X.fit_transform(xtrain)

xtest = sc_X.transform(xtest)

data_test = sc_X.transform(data_test)
#Training our model by using xtrain and ytrain

clf=SVC(C=1, kernel='rbf', gamma='auto')

clf.fit(xtrain,ytrain)

#predetecting the output for xtest

pre=clf.predict(xtest)

confusion_matrix(ytest, pre)
acuraccy=accuracy_score(ytest,pre)

print(acuraccy)

submission = pd.Series(pre,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),submission],axis = 1)

submission.to_csv("final_submission_v1.csv",index=False)

submission.head()