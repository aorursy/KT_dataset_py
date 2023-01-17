# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")
sample_submission=pd.read_csv("../input/sample_submission.csv")
sample_submission.head()
print(train.shape)
print(test.shape)
train.head(2)
train.describe()
test.head(2)
train.Cover_Type.value_counts()
#there are total 7 forest cover type
#let's check for the nulls
train.isnull().sum()
#no null values in training data
test.isnull().sum()
#no null values in test data
train.head()
#let's drop Id column from both
y=train['Cover_Type']
x=train.drop(columns=['Id','Cover_Type'],axis=1)
x.head(5)
x.Elevation.describe()

sns.distplot(np.log1p(train['Elevation']))
#tri modal train.elevation

sns.distplot(np.log1p(train['Aspect']))
#left skewed and bimodal

sns.distplot(train['Slope'])
#tri modal train.elevation
#right skewed data  
Horizontal_Distance_To_Fire_Pointssns.distplot(train['Horizontal_Distance_To_Fire_Points'])
#tri modal train.elevation
sns.distplot(train['Vertical_Distance_To_Hydrology'])
#tri modal train.elevation
sns.distplot(train['Horizontal_Distance_To_Roadways'])
#tri modal train.elevation
sns.distplot(train['Hillshade_9am'])
#tri modal train.elevation
#Hillshade_9am	Hillshade_Noon	Hillshade_3pm	Horizontal_Distance_To_Fire_Points

sns.distplot(train['Hillshade_Noon'])
#tri modal train.elevation
sns.distplot(train['Hillshade_3pm'])
#tri modal train.elevation
sns.distplot(train['Horizontal_Distance_To_Fire_Points'])
sns.countplot(train.Soil_Type3)
# it seems as w have to transform some feature
#let' extract a features Distance from hydrology
x['Distance_From_Hydrology']=np.sqrt((x['Vertical_Distance_To_Hydrology']**2)+(x['Horizontal_Distance_To_Hydrology']**2))
test['Distance_From_Hydrology']=np.sqrt((test['Vertical_Distance_To_Hydrology']**2)+(test['Horizontal_Distance_To_Hydrology']**2))
test.tail(2)
test=test.drop(columns='Id',axis=1)
test.head(2)
test.shape
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.1)
#Random Forest
from sklearn.tree import DecisionTreeClassifier
classifir_dt=DecisionTreeClassifier()
classifir_dt.fit(x_train,y_train)
y_dt=classifir_dt.predict(x_test)
#checking the Accuracy
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,y_dt))
print(confusion_matrix(y_test,y_dt))
#So the decision tree accuracy is 78%
#let's check with Random Forest
from sklearn.ensemble import RandomForestClassifier
clasifier_rf=RandomForestClassifier()
clasifier_rf.fit(x_train,y_train)
y_rf=clasifier_rf.predict(x_test)
#checking the accuracy of random forest
print(classification_report(y_test,y_rf))
print(confusion_matrix(y_test,y_rf))
#Random foresr gives us improvement of 3%
#leta's move to Gradient boost
from sklearn.ensemble import GradientBoostingClassifier
classifir_gb=GradientBoostingClassifier(learning_rate=1,max_depth=5)
classifir_gb.fit(x_train,y_train)
#y_gb=classifir_ab.predict(x_test)
#checking the accuracy of random forest
print(classification_report(y_test,y_gb))
print(confusion_matrix(y_test,y_gb))
from xgboost import XGBClassifier
classifir_xgb=XGBClassifier(learning_rate=1,max_depth=10)
classifir_xgb.fit(x_train,y_train)
y_xgb=classifir_xgb.predict(x_test)
#checking the accuracy of random forest
print(classification_report(y_test,y_xgb))
print(confusion_matrix(y_test,y_xgb))
#let's submit our solution
#Let,s Train with all the data we have
classifir_xgb.fit(x,y)

x.head()
test.head()
print(x.shape,test.shape)
#lets train the full with XGBOost
classifir_xgb.fit(x,y)
y_pred=classifir_xgb.predict(test)
out = pd.DataFrame()
out['Id'] = sample_submission['Id']
out['Cover_Type'] = y_pred
out.to_csv('my_submission.csv', index=False)
out.head(5)
#maximum F1score I am getting is 0.86
from sklearn.svm import LinearSVC
classifier_svm=LinearSVC(C=0.1)
classifier_svm.fit(x_train,y_train)
y_svm=classifier_svm.predict(x_test)
#checking the accuracy of SVM
print(classification_report(y_test,y_svm))
print(confusion_matrix(y_test,y_svm))
# Now lets start to transform the data one by one
sns.distplot(np.log1p(x.Elevation))
#Will not get any advantage after transforing the column
sns.distplot(np.log1p(test.Elevation))
