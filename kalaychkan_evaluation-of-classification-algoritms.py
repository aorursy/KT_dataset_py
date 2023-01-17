# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



#Libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import preprocessing

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings("ignore")



#Machine Learning

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier



#Visualization

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/voicegender/voice.csv")
data.head()
data.info()
data.label = [1 if each == "male" else 0 for each in data.label]
data.info()
data.describe()
y = data.label.values

x_data = data.drop(["label"],axis=1)

min_max_scaler = preprocessing.MinMaxScaler()

x = min_max_scaler.fit_transform(x_data)

data_normalized = pd.DataFrame(x)

data_normalized.describe()
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.5,random_state=42)

x_train = x_train.T

x_test = x_test.T

y_train = y_train.T

y_test = y_test.T



print("x_train: ",x_train.shape)

print("x_test: ",x_test.shape)

print("y_train: ",y_train.shape)

print("y_test: ",y_test.shape)
models = []

models.append(('Logistic Regression', LogisticRegression()))

models.append(('Naive Bayes', GaussianNB()))

models.append(('Decision Tree (CART)',DecisionTreeClassifier())) 

models.append(('K-NN', KNeighborsClassifier()))

models.append(('SVM', SVC()))

models.append(('RandomForestClassifier', RandomForestClassifier()))



for name, model in models:

    model = model.fit(x_train.T,y_train.T)

    ACC = model.score(x_test.T,y_test.T)

    print("{} -> ACC: {} ".format(name,ACC))   

    y_pred = model.predict(x_test.T)

    y_true = y_test.T

    cm = confusion_matrix(y_true,y_pred)

    f, ax = plt.subplots(figsize =(5,5))

    sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

    plt.xlabel("y_pred")

    plt.ylabel("y_true")

    plt.show()