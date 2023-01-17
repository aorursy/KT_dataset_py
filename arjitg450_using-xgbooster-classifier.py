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
#importing modules and functions

import pandas as pd

import numpy as np

import graphviz

from sklearn import tree

from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier

from sklearn.model_selection import train_test_split # Import train_test_split function

from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
dff=pd.read_csv("../input/adult-census-income/adult.csv")
dff=dff.drop(columns=['fnlwgt', 'capital.gain','capital.loss'])

dff["workclass"]=dff["workclass"].replace("?", "Private")

dff.head(4)

dff["workclass"].value_counts()
dff["occupation"]=dff["occupation"].replace("?","Not specified")

dff.head(4)

dff.income[dff.occupation=="Not Specifeid"]
a=dff.income[dff.education=="Some-college"]

a.value_counts()
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

dff['age']=le.fit_transform(dff["age"])

dff['workclass']=le.fit_transform(dff["workclass"])

dff['education']=le.fit_transform(dff["education"])

dff['education.num']=le.fit_transform(dff["education.num"])



dff['marital.status']=le.fit_transform(dff["marital.status"])

dff['occupation']=le.fit_transform(dff["occupation"])

dff['relationship']=le.fit_transform(dff["relationship"])

dff['race']=le.fit_transform(dff["race"])

dff['sex']=le.fit_transform(dff["sex"])

dff['hours.per.week']=le.fit_transform(dff["hours.per.week"])

dff['native.country']=le.fit_transform(dff["native.country"])

dff['income']=le.fit_transform(dff["income"])
dff.head()
feature_cols=["age","workclass","education","education.num","martial.status","occupation","relationship","race","sex","hours.per.week","native.country"]

target=["0","1"]

X_train,X_test,y_train, y_test=train_test_split(dff[dff.columns[0:11]], dff[dff.columns[11]], test_size=0.33, random_state=42)
# Create Decision Tree classifer object

clf = DecisionTreeClassifier(criterion="entropy",max_depth=6)



# Train Decision Tree Classifer

clf = clf.fit(X_train,y_train)



#Predict the response for test dataset

y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
import graphviz

dot_data=tree.export_graphviz(clf,out_file=None)

graph = graphviz.Source(dot_data)

graph.render("dff")
dot_data = tree.export_graphviz(clf, out_file=None, 

feature_names=feature_cols,  

class_names=target,  

filled=True, rounded=True,  

special_characters=True)  

graph = graphviz.Source(dot_data)  

graph 

 
from xgboost import XGBClassifier

import xgboost as xgb
model = XGBClassifier(max_depth=3,learning_rate=0.1,n_estimators=300,booster="gbtree",reg_lambda=0.5,reg_alpha=0.5)

model.fit(X_train, y_train)
y_pred_xgb = model.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_xgb))


#plotting technique - 1 (varying the size)

from xgboost import plot_tree

import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (80,12)

plot_tree(model)

plt.show()
#plotting technique 2 (more clear,more dpi)

xgb.to_graphviz(model, num_trees=2)