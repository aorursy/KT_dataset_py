# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from sklearn.datasets import load_iris

from sklearn import tree

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score,confusion_matrix

from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dataset=pd.read_csv("../input/breast-cancer-wisconsin-data/data.csv")

dataset.head()
sns.scatterplot(x=dataset.radius_mean, y=dataset.smoothness_mean, hue=dataset.diagnosis )


y = dataset.diagnosis                          # M or B 

list = ['Unnamed: 32','id','diagnosis']

x = dataset.drop(list,axis = 1 )

x.head()

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.3)

model=tree.DecisionTreeClassifier().fit(X_train,y_train) 
ac = accuracy_score(y_test,model.predict(X_test))

print('Accuracy is: ',ac)

cm = confusion_matrix(y_test,model.predict(X_test))

sns.heatmap(cm,annot=True,fmt="d")
tree.plot_tree(model);
forest=RandomForestClassifier(n_estimators=45,max_depth=10).fit(X_train,y_train)
ac = accuracy_score(y_test,forest.predict(X_test))

print('Accuracy is: ',ac)
xgboostmodel=XGBClassifier(n_estimators=1000,learning_rate=0.05).fit(X_train,y_train)

ac = accuracy_score(y_test,xgboostmodel.predict(X_test))

print('Accuracy is: ',ac)