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

import seaborn as sn

from sklearn.feature_selection import SelectKBest,chi2

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import StratifiedKFold

from sklearn import model_selection

from sklearn.model_selection import train_test_split
train_data = pd.read_csv("/kaggle/input/mobile-price-classification/train.csv")
train_data.head()
train_data.tail()
train_data.isnull().any()
train_data['price_range'].value_counts()
x = train_data.iloc[:,0:20]

y = train_data.iloc[:,20:]
x
print(x.shape)

print(y.shape)
best_feat = SelectKBest(chi2,k=10)

fit = best_feat.fit(x,y)
dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(x.columns)
# concat both dataframes for comparing best scores



score_list = pd.concat([dfcolumns,dfscores],axis=1)

score_list.columns = ["features","scores"]
score_list
print(score_list.nlargest(10,"scores"))
train_data.drop(train_data.columns[[1,2,3,5,7,9,10,17,18,19]],axis=1,inplace=True)
train_data.head()
train_data.tail()
x = train_data.iloc[:,0:10].values

y = train_data.iloc[:,10:].values
x
y
# Random Forest Classifier



skfold = StratifiedKFold(n_splits=3,random_state=0)

model_c = RandomForestClassifier(criterion='entropy',n_estimators=10,random_state=0,n_jobs=-1)

results_skfold = model_selection.cross_val_score(model_c, x, y.ravel(), cv=skfold)

print("Accuracy: %.2f%%" % (results_skfold.mean()*100.0))
# KNN Classifier



skfold = StratifiedKFold(n_splits=3,random_state=0)

model_knn = KNeighborsClassifier(n_neighbors=10)

results_skfold = model_selection.cross_val_score(model_c, x, y.ravel(), cv=skfold)

print("Accuracy: %.2f%%" % (results_skfold.mean()*100.0))
# Decision Tree Classifier



skfold = StratifiedKFold(n_splits=3,random_state=0)

model_c = DecisionTreeClassifier()

results_skfold = model_selection.cross_val_score(model_c, x, y.ravel(), cv=skfold)

print("Accuracy: %.2f%%" % (results_skfold.mean()*100.0))
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
print(x_train.shape)

print(x_test.shape)

print(y_train.shape)

print(y_test.shape)
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(x_train,y_train.ravel())
test_data = pd.read_csv("/kaggle/input/mobile-price-classification/test.csv")
test_data.head()
test_data_1 = test_data
test_data_1 = test_data_1.drop(test_data_1.columns[[0,2,3,4,6,8,10,11,18,19,20]],axis=1)
predicted_val = knn.predict(test_data_1)
predicted_val
test_data['price_range']=predicted_val
test_data