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
df = pd.read_csv("../input/Admission_Predict_Ver1.1.csv")
df.head()
df.info()
x = df.iloc[:, 1:8] 
y = df.iloc[:, 8]  
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y)
from sklearn import svm
clf = svm.SVR(gamma='auto')
clf.fit(x_train, y_train)
label=[]
accuracy=[]
label.append('SVR')
accuracy.append(clf.score(x_test, y_test))
print(clf.score(x_test, y_test))
from sklearn import linear_model
clf = linear_model.Ridge(alpha=.5)
clf.fit(x_train, y_train)
label.append('Ridge')
accuracy.append(clf.score(x_test, y_test))
print(clf.score(x_test, y_test))
clf = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0], cv=3)
clf.fit(x_train, y_train)
label.append('RidgeCV')
accuracy.append(clf.score(x_test, y_test))
print(clf.score(x_test, y_test))
clf = linear_model.Lasso(alpha=0.1)
clf.fit(x_train, y_train)
label.append('Lasso')
accuracy.append(clf.score(x_test, y_test))
print(clf.score(x_test, y_test))
clf = linear_model.BayesianRidge()
clf.fit(x_train, y_train)
label.append('BayesianRidge')
accuracy.append(clf.score(x_test, y_test))
print(clf.score(x_test, y_test))
clf = linear_model.ARDRegression()
clf.fit(x_train, y_train)
label.append('ARDRegression')
accuracy.append(clf.score(x_test, y_test))
print(clf.score(x_test, y_test))
clf = linear_model.TheilSenRegressor()
clf.fit(x_train, y_train)
label.append('TheilSenRegressor')
accuracy.append(clf.score(x_test, y_test))
print(clf.score(x_test, y_test))
clf.predict(x_test[10:20])
y_test[15:25]
import matplotlib.pyplot as plt
import numpy as np
index = np.arange(len(label))
def plot_bar_x():
    # this is for plotting purpose
    index = np.arange(len(label))
    plt.bar(index, accuracy)
    plt.xlabel('Model', fontsize=10)
    plt.ylabel('Accuracy', fontsize=10)
    plt.xticks(index, label, fontsize=10, rotation=90)
    plt.title('Accuracy of different models')
    plt.savefig("model_accuracy.png")
    plt.show()
plot_bar_x()
