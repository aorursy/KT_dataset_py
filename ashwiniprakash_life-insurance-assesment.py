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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline 

data_train = pd.read_csv('../input/train.csv')
data_test = pd.read_csv('../input/test.csv')

#Lets see data sample
data_train.sample(10)


# Lets check Df shape
data_train.shape

# there are 128 features.
data_test.shape
data_train.dtypes
data_train.dtypes.unique()
#No string data type - all are numerical values which is good.
data_train.isnull().sum()[data_train.isnull().sum() !=0]
#Below listed columns have missing values in the combined (Train+test) dataset. 
# Lets draw a bar graph to visualize percentage of missing features in train set
missing= data_train.isnull().sum()[data_train.isnull().sum() !=0]
missing=pd.DataFrame(missing.reset_index())
missing.rename(columns={'index':'features',0:'missing_count'},inplace=True)
missing['missing_count_percentage']=((missing['missing_count'])/59381)*100
plt.figure(figsize=(20,8))
sns.barplot(y=missing['features'],x=missing['missing_count_percentage'])

#Looking at below bar grah- 
#Medical_Hist_32/24/15/10 , Family_hist_5 are top five features with huge amount of missing data ( imputaion to these might not be fruitful - I will drop these features)

# Lets see spread of data before we impute missing values
plt.plot(figsize=(15,10))
sns.boxplot(data_train['Employment_Info_1'])
# Employment_Info_1 seems to have lots of outliers - Median should be right to impute missing values
data_train['Employment_Info_1'].isna().sum()
data_train['Employment_Info_1'].fillna(data_train['Employment_Info_1'].median(),inplace=True) 
# imputing with Meadian , as there are lots of Outliers 
data_test['Employment_Info_1'].fillna(data_test['Employment_Info_1'].median(),inplace=True) 
data_train['Employment_Info_1'].isna().sum()
#Outlier Treatment -
data_train['Employment_Info_1'].describe()

sns.boxplot(data_train['Employment_Info_4'])
# ['Employment_Info_4'] is has most of the values centered close to zero , also huge presence of outliers 

data_train['Employment_Info_4'].fillna(data_train['Employment_Info_4'].median(),inplace=True)
data_test['Employment_Info_4'].fillna(data_test['Employment_Info_4'].median(),inplace=True)
sns.boxplot(data_train['Employment_Info_6'])
#No outlieers - mean should be rigth candidate to impute missing values
data_train['Employment_Info_6'].fillna(data_train['Employment_Info_6'].mean(),inplace=True)
data_test['Employment_Info_6'].fillna(data_test['Employment_Info_6'].mean(),inplace=True)
sns.boxplot(y=data_train['Medical_History_1'])
data_train['Medical_History_1'].fillna(data_train['Medical_History_1'].median(),inplace=True)
data_test['Medical_History_1'].fillna(data_test['Medical_History_1'].median(),inplace=True)
#lets drop features with high number of missing values 
data_train.drop(['Medical_History_10','Medical_History_15','Medical_History_24','Medical_History_32','Family_Hist_3','Family_Hist_5','Family_Hist_2','Family_Hist_4'],axis=1,inplace=True)


data_test.drop(['Medical_History_10','Medical_History_15','Medical_History_24','Medical_History_32','Family_Hist_3','Family_Hist_5','Family_Hist_2','Family_Hist_4'],axis=1,inplace=True)
data_train.isnull().sum()[data_train.isnull().sum()!=0]
#imputing with median 
data_train['Insurance_History_5'].fillna(data_train['Insurance_History_5'].median(),inplace=True)
data_test['Insurance_History_5'].fillna(data_test['Insurance_History_5'].median(),inplace=True)

data_train.isnull().sum()
#All missing NA values has been treated

data_train.head()
#Product_info_2 seems to be the only feature where we should map string values with numeric categorical values
data_train['Product_Info_2'].unique()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data_train['Product_Info_2']=le.fit_transform(data_train['Product_Info_2'])
data_test['Product_Info_2']=le.transform(data_test['Product_Info_2'])

#data_train.dtypes
#Employment_Info_1-4-6  Insurance_History_5
# I faced an error stating dta types of train columns are not float/numeric ill apply encoder on all column and see what happens

data_train.head()
# feature meatrix and response vector seperation
X_train=data_train.iloc[:,0:-1]
y_train=data_train['Response']
X_train.drop('Id',axis=1,inplace=True)
X_train.head()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_train,y_train)
y_train.unique()
#there are 8 labels/class in dataset

from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.multiclass import OneVsRestClassifier

# Using a Decision Tree classifier 
from sklearn.tree import DecisionTreeClassifier
param_grid={'max_depth':range(1,20,2)}
DT=DecisionTreeClassifier()
clf_DT=GridSearchCV(DT,param_grid,cv=10,scoring='accuracy',n_jobs=-1).fit(X_train,y_train)
y_pred=clf_DT.predict(X_test)
print(accuracy_score(y_test,y_pred))


#Using a Random Forest tree classifier
from sklearn.ensemble import RandomForestClassifier
param_grid={'max_depth':range(1,20,2)}
RF=RandomForestClassifier()
clf_rf=GridSearchCV(RF,param_grid,cv=10,scoring='accuracy',n_jobs=-1).fit(X_train,y_train)
y_pred=clf_rf.predict(X_test)
accuracy_score(y_test,y_pred)
ids = data_test['Id']
predictions = clf_DT.predict(data_test.drop('Id', axis=1))


output = pd.DataFrame({ 'Id' : ids, 'Response': predictions })
output.to_csv('/Users/adityaprakash/Downloads/predictions.csv', index = False)
output.head()
