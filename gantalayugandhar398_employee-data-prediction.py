# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
### Load the required libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.tree import DecisionTreeClassifier, export_graphviz,DecisionTreeRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,mean_absolute_error

from sklearn import tree,ensemble

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler
employee_data=pd.read_csv("/kaggle/input/employee_data.csv")
employee_data.head()
employee_data.shape
employee_data.describe()
#checking the datatypes and levels of employee_data

def inspect_data(data):

    return pd.DataFrame({"Data Type":data.dtypes,"No of Levels":data.apply(lambda x: x.nunique(),axis=0), "Levels":data.apply(lambda x: str(x.unique()),axis=0)})

inspect_data(employee_data)
#function for missing values in columns in data

def missing_coldata(df):

    missin_col = pd.DataFrame(round(df.isnull().sum().sort_values(ascending=False)/len(df.index)*100,1),

                              columns=['% of missing value'])

    missin_col['Count of Missing Values'] = df.isnull().sum()

    return missin_col



missing_coldata(employee_data)
#drop the columns because %of missing values are high

employee_data=employee_data.drop(["filed_complaint","recently_promoted"],axis=1)
#dividing numerical

cat_cols=["department","salary","n_projects","status"]

num_cols=["last_evaluation","satisfaction","avg_monthly_hrs","tenure"]
employee_data[cat_cols] = employee_data[cat_cols].apply(lambda x: x.astype('category'))

employee_data[num_cols] = employee_data[num_cols].apply(lambda x: x.astype('float'))

employee_data.dtypes
employee_data['department']=employee_data['department'].replace(np.NaN,'sales')
## Convert Categorical Columns to Dummies

cat_cols=["department","salary","n_projects","status"]

employee_data = pd.get_dummies(employee_data,columns=cat_cols,drop_first=True,)
employee_data.head()
## Split the data into X and y

x = employee_data.copy().drop("status_Left",axis=1)

y = employee_data["status_Left"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)
## Print the shape of x_train, x_test, y_train, y_test

print(x_train.shape)

print(x_test.shape)

print(y_train.shape)

print(y_test.shape)
x_train['satisfaction'].fillna(x_train['satisfaction'].mean(),inplace=True)

x_train['last_evaluation'].fillna(x_train['last_evaluation'].mean(),inplace=True)

x_train['tenure'].fillna(x_train['tenure'].mean(),inplace=True)
x_test['satisfaction'].fillna(x_train['satisfaction'].mean(),inplace=True)

x_test['last_evaluation'].fillna(x_train['last_evaluation'].mean(),inplace=True)

x_test['tenure'].fillna(x_train['tenure'].mean(),inplace=True)
x_train.isnull().sum()
x_test.isnull().sum()
x_train.head()
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

scaler.fit(x_train.iloc[:,:4])



x_train.iloc[:,:4]=scaler.transform(x_train.iloc[:,:4])

x_test.iloc[:,:4]=scaler.transform(x_test.iloc[:,:4])
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

ABclassifier = ensemble.AdaBoostClassifier()

from xgboost import  XGBClassifier



from sklearn.metrics import classification_report,confusion_matrix

from sklearn.metrics import precision_recall_fscore_support as sco

from sklearn.metrics import confusion_matrix

##logistic regression

lrc = LogisticRegression()



lrc.fit(x_train,y_train)



y_pred_train_lrc = lrc.predict(x_train)

y_pred_test_lrc = lrc.predict(x_test)



print(classification_report(y_train,y_pred_train_lrc))

print(classification_report(y_test,y_pred_test_lrc))



#status_Left=1

#status_Employed=0
confusion_matrix(y_test,y_pred_test_lrc)
##decision tree

dtc = DecisionTreeClassifier()



dtc.fit(x_train,y_train)



y_pred_train_dtc = dtc.predict(x_train)

y_pred_test_dtc = dtc.predict(x_test)



print(classification_report(y_train,y_pred_train_dtc))

print(classification_report(y_test,y_pred_test_dtc))

##svc

svc = SVC()



svc.fit(x_train,y_train)



y_pred_train_svc = svc.predict(x_train)

y_pred_test_svc = svc.predict(x_test)



print(classification_report(y_train,y_pred_train_svc))

print(classification_report(y_test,y_pred_test_svc))



##random forest

rfc = RandomForestClassifier()

rfc.fit(X = x_train,y = y_train)



train_predictions = rfc.predict(x_train)

test_predictions = rfc.predict(x_test)



print(classification_report(y_train,train_predictions))

print(classification_report(y_test,test_predictions))

##knn classifier

knn_classifier = KNeighborsClassifier(n_neighbors=3,weights="distance",algorithm="brute")

knn_classifier.fit(x_train, y_train)



pred_train = knn_classifier.predict(x_train) 

pred_test = knn_classifier.predict(x_test)



print(classification_report(y_train,pred_train))

print(classification_report(y_test,pred_test))
##adaboost

ABclassifier.fit(x_train,y_train)

AB_train_preds=ABclassifier.predict(x_train)

AB_test_preds = ABclassifier.predict(x_test)



print("Classification Report")

print(classification_report(y_test,AB_test_preds))

print("Classification Report")

print(classification_report(y_train,AB_train_preds))
##xgboost

x_classifier = XGBClassifier()

x_classifier.fit(x_train,y_train)



xgboost_train_preds=x_classifier.predict(x_train)

xgboost_test_preds=x_classifier.predict(x_test) 



print(classification_report(y_test,xgboost_test_preds))

print(classification_report(y_train,xgboost_train_preds))