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
## Initialize the Libraries



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import os

from scipy.stats import zscore

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics

from sklearn.metrics import confusion_matrix

import statsmodels.api as sm

from statsmodels.sandbox.regression.predstd import wls_prediction_std

from sklearn import model_selection

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import AdaBoostClassifier
## Import data



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
## creating dataframe



titanic_train= pd.read_csv('/kaggle/input/titanic/train.csv')

titanic_test= pd.read_csv('/kaggle/input/titanic/test.csv')

titanic_submission= pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
titanic_test.head()




## rearranging column in titanic train





titanic_train_arrange=titanic_train[['PassengerId','Name','Age','Sex','SibSp','Parch','Ticket','Fare','Cabin','Embarked','Pclass','Survived']]





## rearranging column in titanic test



titanic_test_arrange=titanic_test[['PassengerId','Name','Age','Sex','SibSp','Parch','Ticket','Fare','Cabin','Embarked','Pclass']]



## Looking into train & test rearrange dataset



print(titanic_train_arrange.head())



print('\n')

print(titanic_test_arrange.head())



## info about the train dataset



titanic_train_arrange.info()
## summary statistics



titanic_train_arrange.describe()
## Missing values



print(titanic_train_arrange.isnull().sum())





print('\n')



print(titanic_test_arrange.isnull().sum())




# titanic_train_arrange.describe()
# ## age



# titanic_train_arrange['Age']= titanic_train_arrange.fillna(titanic_train_arrange['Age'].median())

# titanic_test_arrange['Age']= titanic_test_arrange.fillna(titanic_test_arrange['Age'].median())



## replacing nan values of age with ffill method



titanic_train_arrange["Age"].fillna( method ='ffill',inplace=True)

titanic_test_arrange["Age"].fillna( method ='ffill',inplace=True)



# ## Fare



# titanic_train_arrange['Fare']= titanic_train_arrange.fillna(titanic_train_arrange['Fare'].median())

# titanic_test_arrange['Fare']= titanic_test_arrange.fillna(titanic_test_arrange['Fare'].median())



## replacing nan values of fare with ffill method



titanic_train_arrange["Fare"].fillna( method ='ffill',inplace=True)



titanic_test_arrange["Fare"].fillna( method ='ffill',inplace=True)



# ##Cabin



titanic_train_arrange['Cabin']= titanic_train_arrange.fillna(titanic_train_arrange['Cabin'].mode())

titanic_test_arrange['Cabin']= titanic_test_arrange.fillna(titanic_test_arrange['Cabin'].mode())



## Embarked

titanic_train_arrange['Embarked']= titanic_train_arrange.fillna(titanic_train_arrange['Embarked'].mode())



print(titanic_train_arrange.info())



print(titanic_test_arrange.info())
# # ## checking for datatypes



print(titanic_train_arrange.describe())



print(titanic_test_arrange.describe())
# ### coverting fractional age valuees to 1



titanic_train_arrange.loc[titanic_train_arrange.Age <1,'Age']=1

titanic_test_arrange.loc[titanic_test_arrange.Age <1,'Age']=1

titanic_train_arrange.info()

titanic_train_arrange.describe()

titanic_train_arrange.shape







titanic_test_arrange.info()

print('\n')

titanic_test_arrange.describe()
# # 1. Age round off



titanic_train_arrange['Age']= round(titanic_train_arrange['Age'])

titanic_test_arrange['Age']= round(titanic_test_arrange['Age'])

# # 2. Fare round off upto 2 decimal points



titanic_train_arrange['Fare']= round(titanic_train_arrange['Fare'],2)

titanic_test_arrange['Fare']= round(titanic_test_arrange['Fare'],2)





print(titanic_train_arrange.describe())



print('\n')



print(titanic_test_arrange.describe())
# # 2. Dummy variable for sex column 



titanic_train_dummy= pd.get_dummies(titanic_train_arrange,columns=['Sex','Embarked','Pclass'],drop_first=True)

titanic_test_dummy= pd.get_dummies(titanic_test_arrange,columns=['Sex','Embarked','Pclass'],drop_first=True)



titanic_train_dummy.head()
titanic_test_dummy.head()
##removal of unnecessary columns



titanic_train_ml= titanic_train_dummy.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)

titanic_test_ml= titanic_train_dummy.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)
## train test split



# create target and independent vaiable



X=titanic_train_ml.drop(['Survived'],axis=1)

y=titanic_train_ml.filter(['Survived'])





X_train,X_test,y_train,y_test= train_test_split(X,y,test_size= 0.2,random_state=1)


# create model

Linear_model= LinearRegression()

# fit the model

Linear_model= Linear_model.fit(X_train,y_train)

# Predict the model

Linear_y_pred= Linear_model.predict(X_test)



print(Linear_y_pred)

#Model Fitted summary



Linearmodel_ols=sm.OLS(y_train,X_train)



result=Linearmodel_ols.fit()



print(result.summary())


print("mean_absolute_error : ",metrics.mean_absolute_error(y_test,Linear_y_pred))

print("mean_squared_error : ",metrics.mean_squared_error(y_test,Linear_y_pred))

print("mean_squared_root_error :",np.sqrt(metrics.mean_squared_error(y_test,Linear_y_pred)))



# create model

Logistic_model= LogisticRegression()

# fit the model

Logistic_model= Logistic_model.fit(X_train,y_train)

# Predict the model

Logistic_y_pred= Logistic_model.predict(X_test)



print(Logistic_y_pred)
print("accuracy metrics :",metrics.accuracy_score(y_test,Logistic_y_pred))

print('\n')

print("Confusion metrics: ",metrics.confusion_matrix(y_test,Logistic_y_pred))

print('\n')

print('\n')



print('classification report: ',metrics.classification_report(y_test,Logistic_y_pred))
# create model

decision_tree_model= DecisionTreeClassifier()

# fit the model

decision_tree_model= decision_tree_model.fit(X_train,y_train)

# Predict the model

decision_tree_y_pred= decision_tree_model.predict(X_test)



print(decision_tree_y_pred)
print("accuracy metrics :",metrics.accuracy_score(y_test,decision_tree_y_pred))

print('\n')

print("Confusion metrics: ",metrics.confusion_matrix(y_test,decision_tree_y_pred))

print('\n')

print('\n')



print('classification report: ',metrics.classification_report(y_test,decision_tree_y_pred))
kfold= model_selection.KFold(n_splits=10,random_state=7)

y=np.array(y)

cart= DecisionTreeClassifier()

num_trees=100

model= BaggingClassifier(base_estimator=cart,n_estimators=num_trees,random_state=7)

results= model_selection.cross_val_score(model,X,y,cv=kfold)

print(results.mean())
print(results)
kfold= model_selection.KFold(n_splits=10,random_state=10)

cart= DecisionTreeClassifier()

num_trees=100

model= AdaBoostClassifier(base_estimator=cart,n_estimators=num_trees,random_state=10)

results= model_selection.cross_val_score(model,X,y,cv=kfold)

print(results.mean())