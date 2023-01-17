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
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix

from sklearn.datasets import load_iris

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn import metrics



from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')  

import warnings

warnings.filterwarnings('ignore')  #this will ignore the warnings.it wont display warnings in notebook
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.metrics import classification_report

from sklearn.metrics import recall_score

def printreport(exp, pred):

    print(classification_report(exp, pred))

    print("recall score")

    print(recall_score(exp,pred,average = 'macro'))
gr = pd.read_csv('../input/greedata/featureO.csv')
for column in gr.columns[0:-1]:

    for spec in gr["label"].unique():

        selected_spec = gr[gr["label"] == spec]

        selected_column = selected_spec[column]

        

        std = selected_column.std()

        avg = selected_column.mean()

        

        three_sigma_plus = avg + (3 * std)

        three_sigma_minus =  avg - (3 * std)

        

        outliers = selected_column[((selected_spec[column] > three_sigma_plus) | (selected_spec[column] < three_sigma_minus))].index

        gr.drop(outliers, inplace=True)

        print(column, spec, outliers)
x = gr.iloc[:,0:3].values 

y = gr.label.values

print(x)

print(y)

from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.3, random_state=0, stratify=y)
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)

x_test = scaler.fit_transform(x_test)
xtrain = x_train

ytrain = y_train

xtest = x_test

ytest = y_test
k_range = list(range(1,11))

scores = []

for k in k_range:

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(xtrain, ytrain)

    y_pred = knn.predict(xtest)

    scores.append(metrics.accuracy_score(ytest, y_pred))

    

plt.plot(k_range, scores)

plt.xlabel('Value of k for KNN')

plt.ylabel('Accuracy Score')

plt.title('Accuracy Scores for Values of k of k-Nearest-Neighbors')

plt.show()

p1 = pd.DataFrame(scores , k_range)

s1 = p1.loc[:,0]

s1_argmax = s1[s1 == s1.max()].index.values
k = s1_argmax[0]
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors=k, metric='minkowski')

knn.fit(x_train,y_train)



y_pred = knn.predict(x_test)



print('KNN')

cm = confusion_matrix(y_test,y_pred)

s = accuracy_score(y_test, y_pred)

print('accury')

print(s)

printreport(y_test, y_pred)



#print(cm)

import seaborn as sns

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()
x = gr.iloc[:,0:3].values 

y = gr.label.values

print(x)

print(y)





from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0, stratify=y)



scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)

x_test = scaler.fit_transform(x_test)
xtrain = x_train

ytrain = y_train

xtest = x_test

ytest = y_test
k_range = list(range(1,11))

scores = []

for k in k_range:

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(xtrain, ytrain)

    y_pred = knn.predict(xtest)

    scores.append(metrics.accuracy_score(ytest, y_pred))

    

plt.plot(k_range, scores)

plt.xlabel('Value of k for KNN')

plt.ylabel('Accuracy Score')

plt.title('Accuracy Scores for Values of k of k-Nearest-Neighbors')

plt.show()

p1 = pd.DataFrame(scores , k_range)

# print(p1)

# print(p1.loc[:,0].max())



s1 = p1.loc[:,0]

s1_argmax = s1[s1 == s1.max()].index.values



k = s1_argmax[0]
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors=k, metric='minkowski')

knn.fit(x_train,y_train)



y_pred = knn.predict(x_test)



print('KNN')

cm = confusion_matrix(y_test,y_pred)

s = accuracy_score(y_test, y_pred)

print('accury')

print(s)

printreport(y_test, y_pred)



#print(cm)

import seaborn as sns

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()
kfold = KFold(n_splits=5, shuffle=True)

results = cross_val_score(knn, x, y, cv=kfold)

print(results)

print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
import xgboost as xgb

from sklearn.metrics import accuracy_score, confusion_matrix

from xgboost import plot_importance



xgb_cls = xgb.XGBClassifier(objective="multi:softmax", num_class=5)

xgb_cls.fit(x_train, y_train)

y_pred = xgb_cls.predict(x_test)



print('XGboost')

s = accuracy_score(y_test, y_pred)

print('accury')

print(s)

cm = confusion_matrix(y_test, y_pred)

printreport(y_test, y_pred)







#print(cm)

import seaborn as sns

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()



plot_importance(xgb_cls)