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
#Import Random Forest Model

from sklearn.ensemble import RandomForestClassifier



#Create a Gaussian Classifier

clf=RandomForestClassifier(n_estimators=300,max_depth=5,min_samples_split=3)



#Train the model using the training sets y_pred=clf.predict(X_test)

clf.fit(X_train,y_train)



y_pred=clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

