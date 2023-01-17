import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set()




import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
t_train = pd.read_csv("../input/train.csv")
t_test = pd.read_csv("../input/test.csv")
### Checking which columns have NA values
#t_train.isnull().any()  returns Boolean list of columns with NA or not


t_train.columns[t_train.isnull().any()]
## Number of NA values in columns with NAs


t_train[['Age','Cabin','Embarked']].isnull().sum()
# Nullify Cabin, Drop  Embarked NA rows, Give median to Age NA values


t_train.drop('Cabin', axis=1, inplace=True)
#Drop Embarked NA rows
t_train = t_train[t_train['Embarked'].notnull()]
#Check if  Embarked has na

t_train['Embarked'].isnull().sum()
## Give Age na values median value

med = t_train['Age'].median()


t_train['Age'] = t_train['Age'].fillna(value=med)



#Family unit or not

def nuking (row):
    if row['SibSp'] >= 1 and row['Parch']>=1:
      return "FM"
    else:
        return "Not"
    


t_train['NUKEFM_or_Not'] = t_train.apply (lambda row: nuking (row),axis=1)
########## create a variable that designates if they are a Miss or Not
# Turn Age column into float 

#def f(x):
#    try:
#        return np.float(x)
#    except:
#        return np.nan

#t_train['Age'] = t_train['Age'].apply(f)


def missing (row):
    if row['Age'] >= 18 and row['Name'].find("Miss") != -1:
      return "Miss"
    else:
      return "Not"
    

    #if row['Age'] >= 18 and row['Name'].find("Miss") != -1:
     
#_train['Age']= pd.to_numeric(t_train['Age'], errors='coerce').fillna(0)


t_train['Miss_or_Not'] = t_train.apply (lambda row: missing (row),axis=1)
#t_train.apply (lambda row: missing (row),axis=1)




### # Create new column  family_size
def sizing(row):
    len = row['SibSp'] + row['Parch'] + 1
    return len
    
t_train['family_size']=t_train.apply (lambda row: sizing (row),axis=1)
## turn Survived into Categorical Variable
def surving (row):
    if row['Survived'] == 1 :
      return "Yes"
    else:
        return "No"
    


t_train['Survived'] = t_train.apply (lambda row: surving (row),axis=1)
## turn PClass into Categorical Variable
def classing (row):
    if row['Pclass'] == 1 :
      return "First"
    elif row['Pclass'] == 2:
        return "Second"
    else:
        return"Third"
    


t_train['Pclass'] = t_train.apply (lambda row: classing (row),axis=1)
t_train['Pclass'], uniclass = pd.factorize(t_train['Pclass'], sort=True)
uniclass    

##
t_train['Survived'], uniques = pd.factorize(t_train['Survived']) 

t_train['Sex'], unisex = pd.factorize(t_train['Sex'])
t_train['Embarked'], uniembar = pd.factorize(t_train['Embarked'])
t_train['Miss_or_Not'], unimiss = pd.factorize(t_train['Miss_or_Not'])
t_train['NUKEFM_or_Not'], uninuke = pd.factorize(t_train['NUKEFM_or_Not'])
t_train.info()
uniclass


t_train.head(10)
# First remove Name and Ticket collumns
t_train.drop('Name', axis=1, inplace=True)
t_train.drop('Ticket', axis=1, inplace=True)
t_train.drop('PassengerId', axis=1, inplace=True)
import xgboost as xgb
#Create training and test datasets
X_train = t_train.drop("Survived", axis = 1)
y_train = t_train['Survived'].values
#y_train = y_train.reshape(-1, 1)
#X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.2, random_state=42)
X_test = t_test


y_train.shape

###
#X_train.shape
#Convert the training and testing sets into DMatrixes
#DMatrix is the recommended class in xgboost.

DM_train = xgb.DMatrix(data = X_train, 
                       label = y_train)  
#DM_test =  xgb.DMatrix(t_test)

#Parameters for grid search

gbm_param_grid = {
     'colsample_bytree': np.linspace(0.5, 1, 5),
     'n_estimators':[100, 200],
     'max_depth': [5, 8, 10, 11, 15],      
}

#Instantiate the classifier
gbm = xgb.XGBClassifier()


#grid_mse = GridSearchCV(estimator = gbm, param_grid = gbm_param_grid, scoring = 'roc_auc',cv = 5, verbose = 1)
grid_mse2 = GridSearchCV(estimator = gbm, param_grid = gbm_param_grid, scoring = 'roc_auc',cv = 5, verbose = 1)
#grid_mse.fit(X_train, y_train)
#print("Best parameters found: ",grid_mse.best_params_)
#print("Lowest RMSE found: ", np.sqrt(np.abs(grid_mse.best_score_)))
#grid_mse2.fit(X_train, y_train)
#print("Best parameters found: ",grid_mse2.best_params_)
#print("Lowest RMSE found: ", np.sqrt(np.abs(grid_mse2.best_score_)))
### Checking which columns have NA values
#t_train.isnull().any()  returns Boolean list of columns with NA or not


#t_test.columns[t_test.isnull().sum()]
t_test.isnull().sum()
# Drop Cabin
t_test.drop('Cabin', axis=1, inplace=True)

#Drop Fare NA rows
med = t_test['Fare'].median()


t_test['Fare'] = t_test['Fare'].fillna(value=med)

## Median Age

med = t_test['Age'].median()


t_test['Age'] = t_test['Age'].fillna(value=med)




t_test['NUKEFM_or_Not'] = t_test.apply (lambda row: nuking (row),axis=1)

t_test['Miss_or_Not'] = t_test.apply (lambda row: missing (row),axis=1)

t_test['family_size']=t_test.apply (lambda row: sizing (row),axis=1)




t_test['Pclass'] = t_test.apply (lambda row: classing (row),axis=1)

p_Id = t_test['PassengerId']
#  remove Name, Ticket & PassengerID collumns
t_test.drop('Name', axis=1, inplace=True)
t_test.drop('Ticket', axis=1, inplace=True)
t_test.drop('PassengerId', axis=1, inplace=True)


t_test['Pclass'], uniclass = pd.factorize(t_test['Pclass'], sort=True)




t_test['Sex'], unisex = pd.factorize(t_test['Sex'])
t_test['Embarked'], uniembar = pd.factorize(t_test['Embarked'])
t_test['Miss_or_Not'], unimiss = pd.factorize(t_test['Miss_or_Not'])
t_test['NUKEFM_or_Not'], uninuke = pd.factorize(t_test['NUKEFM_or_Not'])


t_test.head()


#pred = grid_mse2.predict(t_test)



#t_test['PassengerId'] = p_Id

#t_test['Survived'] = pred


#Kaggle_submit= t_test[['PassengerId', 'Survived']]


#Kaggle_submit.shape
#Kaggle_submit.to_csv('PythonKagglesub9', index =False)







