# import the important libraries.

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns



import os

print(os.listdir("../input"))



import warnings

warnings.filterwarnings('ignore')

import pandas_profiling as pp


data=pd.read_csv('../input/bank-additional-full.csv', sep = ';')

data.sample(5)
print("Shape of the data:",data.shape)

print("Columns Names are:\n",data.columns)
print("Data Types for all the columns of the data: \n",data.dtypes)
numeric_data = data.select_dtypes(include = np.number)

numeric_data.head()
numeric_data.columns
categorical_data = data.select_dtypes(exclude = np.number)

categorical_data.head()
categorical_data.columns
pp.ProfileReport(data)
print("Is there any null values in the data ? \n",data.isnull().values.any())
print("Total Null Values in the data = ",data.isnull().sum().sum())
total= data.isnull().sum()

percent_missing = data.isnull().sum()/data.isnull().count()

print(percent_missing)
data[data.duplicated(keep='first')]
data.drop_duplicates(keep='first',inplace=True)
print("Information about the dataframe : \n ")

data.info()
# Which columns have the most missing values?

def missing_data(df):

    total = df.isnull().sum()

    percent = total/df.isnull().count()*100

    missing_values = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    types = []

    for col in df.columns:

        dtype = str(df[col].dtype)

        types.append(dtype)

    missing_values['Types'] = types

    missing_values.sort_values('Total',ascending=False,inplace=True)

    return(np.transpose(missing_values))

missing_data(data)
print('Discrption of Numeric Data : ')

data.describe()
print('Discrption of Object Data : ')

data.describe(include='object')
class_values = (data['y'].value_counts()/data['y'].value_counts().sum())*100

class_values
print("Histogram for the numerical features :\n")

data.hist(figsize=(15,15),edgecolor='k',color='skyblue')

plt.tight_layout()

plt.show()
cols = categorical_data.columns

for column in cols:

    plt.figure(figsize=(15,6))

    plt.subplot(121)

    data[column].value_counts().plot(kind='bar')

    plt.title(column)

    plt.tight_layout()
print("Target values counts:\n",data['y'].value_counts())

data['y'].value_counts().plot.bar()

plt.show()

data.plot(kind='box',subplots=True,layout=(6,2),figsize=(15,15))

plt.tight_layout()
data.groupby(["contact"]).mean()
data.pivot_table(values="age",index="month",columns=["marital","contact"])
data.groupby("education").mean()
from sklearn.preprocessing import LabelEncoder

LE=LabelEncoder()

cat_var=['age', 'job', 'marital', 'education', 'default', 'housing', 'loan',

       'contact', 'month', 'day_of_week','poutcome','y']

for i in cat_var:

    data[i]=LE.fit_transform(data[i])

    

data.head()
X=data.iloc[:,0:7]

y=data.iloc[:,-1:]
#Now with single statement, you will be able to see all the variables created globally across the notebook, data type and data/information

%whos
from sklearn.model_selection import train_test_split,cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import r2_score,confusion_matrix

from sklearn.preprocessing import StandardScaler

import xgboost as xgb

from sklearn.linear_model import ElasticNetCV

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.preprocessing import RobustScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error,accuracy_score

from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split

import math

import sklearn.model_selection as ms

import sklearn.metrics as sklm
sc=StandardScaler()

sc.fit_transform(X)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=2)
lr=LogisticRegression(penalty = 'l1',solver = 'liblinear')

lr.fit(X_train,y_train)

pred_lr=lr.predict(X_test)

confusion_matrix(y_test,pred_lr)

score_lr= accuracy_score(y_test,pred_lr)

print("Accuracy Score is: ", score_lr)

knn=KNeighborsClassifier()

knn.fit(X_train,y_train)

pred_knn=knn.predict(X_test)

confusion_matrix(y_test,pred_knn)
score_knn = cross_val_score(knn,y_test,pred_knn,cv=5)

print(score_knn)

print("Mean of the cross validation scores:",score_knn.mean())
dt=DecisionTreeClassifier()

dt.fit(X_train,y_train)

pred_dt=dt.predict(X_test)

confusion_matrix(y_test,pred_dt)
score_dt=cross_val_score(dt,y_test,pred_dt,cv=5)

print(score_dt)

print("Mean of the cross validation scores:",score_dt.mean())
rf=RandomForestClassifier()

rf.fit(X_train,y_train)

pred_rf=rf.predict(X_test)

confusion_matrix(y_test,pred_rf)
score_rf=cross_val_score(rf,y_test,pred_dt,cv=5)

print(score_rf)

print("Mean of the cross validation scores:",score_rf.mean())
xgb_clf= xgb.XGBClassifier()

xgb_clf.fit(X_train,y_train)

pred_xgb=xgb_clf.predict(X_test)

confusion_matrix(y_test,pred_xgb)
score_xgb = cross_val_score(xgb_clf,y_test,pred_xgb,cv=5)

print(score_xgb)

print("Mean of the cross validation scores:",score_xgb.mean())
print('Feature importances:\n{}'.format(repr(xgb_clf.feature_importances_)))
print("Accuracy Score of Logistic Regression",score_lr)

print("Accuracy Score of KNN",score_knn.mean())

print("Accuracy Score of Decision Tree",score_dt.mean())

print("Accuracy Score of Random Forest",score_rf.mean())

print("Accuracy Score of XGB",score_xgb.mean())
plt.bar(x=["LR","KNN","DT","RF","XGB"],height=[score_lr,score_knn.mean(),score_dt.mean(),score_rf.mean(),score_xgb.mean()])

plt.ylim(0.88,1)

plt.show()