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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
dataset = pd.read_csv('../input/Social_Network_Ads.csv')
dataset.info()
dataset.describe()
dataset.describe(include = ['O'])
dataset['Gender'] = dataset['Gender'].map({'Male':1,'Female':0})
x = dataset.iloc[:,1:4]

y = dataset.iloc[:,4:]
def normalise(x):

    return ((x - np.min(x))/(np.max(x)-np.min(x)))



x.iloc[:,2:] = x.iloc[:,2:].apply(normalise)
from sklearn.model_selection import train_test_split

x_train,x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=100)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(x_train,y_train)

y_pred_lr = lr.predict(x_test)
c = [i for i in range(120)]

plt.figure()

plt.scatter(c,y_test,color='green',label='original')

plt.scatter(c,y_pred_lr)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred_lr)

cm
from sklearn.neighbors import KNeighborsClassifier

lr_knn = KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)

lr_knn.fit(x_train,y_train)

y_pred_knn = lr_knn.predict(x_test)

cm = confusion_matrix(y_test,y_pred_knn)

cm
from sklearn.svm import SVC

lr_svc = SVC(kernel='linear',random_state=0)

lr_svc.fit(x_train,y_train)

y_pred_svc = lr_svc.predict(x_test)

cm = confusion_matrix(y_test,y_pred_svc)

cm
from sklearn.svm import SVC

lr_ksvc = SVC(kernel = 'rbf')

lr_ksvc.fit(x_train,y_train)

y_pred_ksvc = lr_ksvc.predict(x_test)

cm = confusion_matrix(y_test, y_pred_ksvc)

cm
from sklearn.naive_bayes import GaussianNB

lr_nb = GaussianNB()

lr_nb.fit(x_train,y_train)

y_pred_nb = lr_nb.predict(x_test)

cm = confusion_matrix(y_test,y_pred_nb)

cm
from sklearn.tree import DecisionTreeClassifier

lr_dtc = DecisionTreeClassifier()

lr_dtc.fit(x_train,y_train)

y_pred_dtc = lr_dtc.predict(x_test)

cm = confusion_matrix(y_test,y_pred_dtc)

cm
from sklearn.ensemble import RandomForestClassifier

lr_rfc = RandomForestClassifier()

lr_rfc.fit(x_train,y_train)

y_pred_rfc = lr_rfc.predict(x_test)

cm = confusion_matrix(y_test,y_pred_rfc)

cm