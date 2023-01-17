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
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

from datetime import datetime
filename = "/kaggle/input/batdemirtest/file.csv"

dataFrame = pd.read_csv(filename)

dataFrame.head()
dataFrame['takım a'] = LabelEncoder().fit_transform(dataFrame['takım a'])

dataFrame['takım b'] = LabelEncoder().fit_transform(dataFrame['takım b'])



dataFrame.head()
dataFrame.info()
data1 = dataFrame[dataFrame['sonuc\n']==2]

print("Takım A kazananlar-data1:"+ str(data1.shape))



data2 = dataFrame[dataFrame['sonuc\n']==1]

print("Takım B kazananlar-data2:"+ str(data2.shape))



data3 = dataFrame[dataFrame['sonuc\n']==0]

print("Beraber olanlar-data3:"+ str(data3.shape))
X = dataFrame[['takım a', 'takım b', 'oran']]

Y = dataFrame['sonuc\n']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

X_train = X_train.astype('int64')

X_test = X_test.astype('int64')

#ölçeklendirme

scaler = preprocessing.MinMaxScaler((-1,1))

scaler.fit(X)

XX_train = scaler.transform(X_train.values)

XX_test  = scaler.transform(X_test.values)

YY_train = Y_train.values 

YY_test  = Y_test.values



print(XX_train,XX_test,YY_train,YY_test)

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
models = []

models.append(('Logistic Regression', LogisticRegression()))

models.append(('Naive Bayes', GaussianNB()))

models.append(('Decision Tree (CART)',DecisionTreeClassifier())) 

models.append(('K-NN', KNeighborsClassifier()))

models.append(('SVM', SVC()))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('AdaBoostClassifier', AdaBoostClassifier()))

models.append(('BaggingClassifier', BaggingClassifier()))

models.append(('RandomForestClassifier', RandomForestClassifier()))
for name, model in models:

    model = model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)

    from sklearn import metrics

    print("Model -> %s -> ACC: %%%.2f" % (name,metrics.accuracy_score(Y_test, Y_pred)*100))