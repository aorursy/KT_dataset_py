# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
file_path = "../input/lung-cancer-dataset/lung_cancer_examples.csv"

data = pd.read_csv(file_path, index_col="Name")
data.shape
data.describe
data.tail()
num_result_0 = data.loc[data.Result==0].count()

num_result_0
num_result_1 = data[data.Result==1].count()

num_result_1
x = ["0", "1"]

y = [31, 28]

plt.title("Class distribution")

plt.ylabel("Number of records")

plt.bar(x,y)
sns.swarmplot(x=data['Result'],

              y=data['Age'])
sns.swarmplot(x=data['Result'],

              y=data['Smokes'])
sns.swarmplot(x=data['Result'],

              y=data['AreaQ'])
sns.swarmplot(x=data['Result'],

              y=data['Alkhol'])
from sklearn.model_selection import cross_val_score, KFold, train_test_split

X = data.drop(['Surname','Result'], axis=1)

y = data['Result'].copy()
X.shape
y.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
X_train.shape
X_test.shape
results_dict = {}
from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeRegressor

from sklearn.naive_bayes import GaussianNB

kf = KFold(n_splits=5, random_state=0, shuffle=True)
lr = LogisticRegression(C=0.5, random_state=1).fit(X_train, y_train)

mean_acc_lr = cross_val_score(lr, X_train, y_train, n_jobs=-1, cv=kf, scoring='accuracy').mean()

results_dict['Logistic Regression'] = mean_acc_lr

results_dict
knn = KNeighborsClassifier(n_neighbors=5)

mean_acc_knn = cross_val_score(knn, X_train, y_train, n_jobs=-1, cv=kf, scoring='accuracy').mean()

results_dict['KNN'] = mean_acc_knn

results_dict
dt = DecisionTreeRegressor()

mean_acc_dt = cross_val_score(dt, X_train, y_train, n_jobs=-1, cv=kf, scoring='accuracy').mean()

results_dict['Decision Tree'] = mean_acc_dt

results_dict
nb = GaussianNB()

mean_acc_nb = cross_val_score(nb, X_train, y_train, n_jobs=-1, cv=kf, scoring='accuracy').mean()

results_dict['NB'] = mean_acc_nb

results_dict
x = ['Logistic Regression', 'KNN', 'Decision Tree', 'NB']

y = [results_dict['Logistic Regression'], results_dict['KNN'], results_dict['Decision Tree'], results_dict['NB']]

plt.title("Accuracy comparison")

plt.ylabel("Accuracy")

plt.bar(x,y)
from sklearn.metrics import accuracy_score

nb = GaussianNB().fit(X_train, y_train)

predicted = nb.predict(X_test)

accuracy_score(y_test, predicted)
from sklearn.metrics import plot_confusion_matrix



disp = plot_confusion_matrix(nb, X_test, y_test,

                                 display_labels=data['Result'],

                                 cmap=plt.cm.Blues)



disp.ax_.set_title("Confusion Matrix")

disp.confusion_matrix

plt.show()
from sklearn.metrics import confusion_matrix

confusion = confusion_matrix(y_test, predicted)

confusion
TP = confusion[1, 1]

TN = confusion[0, 0]

FP = confusion[0, 1]

FN = confusion[1, 0]
sensitivity = TP/(TP+FN)

specificity = TN/(TN+FP)
"Sensitivity: {} | Specifictity: {}".format(sensitivity, specificity)