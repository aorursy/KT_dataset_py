# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import tree

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv("/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv")

data.head()
data.isnull().sum()
import matplotlib.pyplot as plt

import seaborn as sns

data["DEATH_EVENT"].value_counts()
data.dtypes
data.describe()
print(set(data["smoking"]))

print(set(data["DEATH_EVENT"]))
data[["smoking","DEATH_EVENT"]].corr()
data[["smoking","DEATH_EVENT"]]
data[["high_blood_pressure","DEATH_EVENT"]].corr()
data[["diabetes","DEATH_EVENT"]].corr()
data[["age","DEATH_EVENT"]].corr()
for column in data.select_dtypes([np.number]).columns:

    try:

        print(column,": ",data[[column,"DEATH_EVENT"]].corr()[column][1])

    except:

        print("Problem in: ",column)
x_with_corr=data[["age","ejection_fraction","serum_creatinine","serum_sodium","time"]]

y=data["DEATH_EVENT"]
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_with_corr, y, test_size=0.33, random_state=1071)
from sklearn.tree import DecisionTreeClassifier

dtree= DecisionTreeClassifier(max_depth=3)

dtree.fit(x_train,y_train)

ypred=dtree.predict(x_test)

import sklearn.metrics as metrik

print(metrik.accuracy_score(y_pred=ypred,y_true=y_test))

print(metrik.confusion_matrix(y_pred=ypred,y_true=y_test))
import graphviz

dot_data = tree.export_graphviz(dtree, out_file=None, 

                     feature_names=x_train.columns,class_names=["zero","one"],

                         filled=True, rounded=True,special_characters=True)  

graph = graphviz.Source(dot_data)  

graph 
dot_data = tree.export_graphviz(dtree, out_file=None, 

                     feature_names=x_train.columns,class_names=["one","zero"],

                         filled=True, rounded=True,special_characters=True)  

graph = graphviz.Source(dot_data)  

graph 
from  xgboost import XGBClassifier

xgbo = XGBClassifier(learning_rate=0.01)





xgbo.fit(x_train,y_train)

ypred=xgbo.predict(x_test)





print(metrik.accuracy_score(y_pred=ypred,y_true=y_test))

print(metrik.confusion_matrix(y_pred=ypred,y_true=y_test))
from  xgboost import XGBClassifier

xgbo = XGBClassifier(learning_rate=0.1)





xgbo.fit(x_train,y_train)

ypred=xgbo.predict(x_test)





print(metrik.accuracy_score(y_pred=ypred,y_true=y_test))

print(metrik.confusion_matrix(y_pred=ypred,y_true=y_test))
from  xgboost import XGBClassifier

xgbo = XGBClassifier(learning_rate=0.1,n_estimators=100,gamma=4,subsample=0.8,)





xgbo.fit(x_train,y_train)

ypred=xgbo.predict(x_test)





print(metrik.accuracy_score(y_pred=ypred,y_true=y_test))

print(metrik.confusion_matrix(y_pred=ypred,y_true=y_test))
import xgboost

xgboost.plot_importance(xgbo, importance_type="cover")
xgboost.plot_tree(xgbo,num_trees=4)
mybooster = xgbo.get_booster()    

model_bytearray = mybooster.save_raw()[4:]

def myfun(self=None):

    return model_bytearray

mybooster.save_raw = myfun

#https://stackoverflow.com/questions/61928198/getting-unicodedecodeerror-when-using-shap-on-xgboost solution in this page is used

import shap

explainer = shap.TreeExplainer(mybooster)

shap_values = explainer.shap_values(x_train)

shap.initjs()

shap.force_plot(explainer.expected_value, shap_values[0,:], x_train.iloc[0,:])
shap.force_plot(explainer.expected_value, shap_values[0,:], x_train.iloc[1,:])
shap.force_plot(explainer.expected_value, shap_values[0,:], x_train.iloc[25,:])
shap.force_plot(explainer.expected_value, shap_values[0,:], x_train.iloc[40,:])
X=data.iloc[:,0:12]

Y=data.iloc[:,12:]



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=1071)
from  xgboost import XGBClassifier

xgbo = XGBClassifier()





xgbo.fit(X_train,y_train)

ypred=xgbo.predict(X_test)





print(metrik.accuracy_score(y_pred=ypred,y_true=y_test))

print(metrik.confusion_matrix(y_pred=ypred,y_true=y_test))
from  xgboost import XGBClassifier

xgbo = XGBClassifier(learning_rate=0.1,n_estimators=100,gamma=4,subsample=0.8,)





xgbo.fit(X_train,y_train)

ypred=xgbo.predict(X_test)





print(metrik.accuracy_score(y_pred=ypred,y_true=y_test))

print(metrik.confusion_matrix(y_pred=ypred,y_true=y_test))
from  xgboost import XGBClassifier

xgbo = XGBClassifier(learning_rate=0.1,n_estimators=1000,gamma=5,subsample=0.8,)





xgbo.fit(X_train,y_train)

ypred=xgbo.predict(X_test)





print(metrik.accuracy_score(y_pred=ypred,y_true=y_test))

print(metrik.confusion_matrix(y_pred=ypred,y_true=y_test))
xgboost.plot_importance(xgbo, importance_type="cover")
shap.initjs()

shap.force_plot(explainer.expected_value, shap_values[0,:], x_train.iloc[1,:])