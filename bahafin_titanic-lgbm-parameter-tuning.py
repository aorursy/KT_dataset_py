import pandas as pd

#read training set into a pandas dataframe
train = pd.read_csv('../input/train.csv')
train.info()
train.head()
train.describe(include = 'all')
median_age = train['Age'].median()
median_fare = train['Fare'].median()
#create a function to do above preparations for a given data set, this will be used to prepare test dataset
#these features are adopted from various kernels
def prep_data(df):
    to_be_dropped = ['Name', 'Cabin', 'Ticket']
    
    #handle missing values
    #fill missing values of embarked with most repeated value
    df['Embarked'] = df['Embarked'].fillna('S')
    #fill missing values of age with mean
    df['Age'] = df['Age'].fillna(median_age)
    #fill missing values for age and create bands
    df['Fare'] = df['Fare'].fillna(median_fare)
    
    #create bands from companion size
    df['Companions'] = df['Parch'] + df['SibSp']

    #add parch, sibsp to the list of columns to be dropped
    to_be_dropped.extend(['Parch', 'SibSp'])
    
    #create bands for age
    df.loc[ df['Age'] <= 16, 'Age'] = 0
    df.loc[ (df['Age'] > 16) & (df['Age'] <= 32), 'Age'] = 1
    df.loc[ (df['Age'] > 32) & (df['Age'] <= 48), 'Age'] = 2
    df.loc[ (df['Age'] > 48) & (df['Age'] <= 64), 'Age'] = 3
    df.loc[ df['Age'] > 64, 'Age'] = 4
    df['Age'] = df['Age'].astype(int)
    
    #create bands for fare
    df.loc[ df['Fare'] <= 7.91, 'Fare'] = 0
    df.loc[ (df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare'] = 1
    df.loc[ (df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare'] = 2
    df.loc[ df['Fare'] > 31, 'Fare'] = 3
    df['Fare'] = df['Fare'].astype(int)
    
    #find titles within name field
    df['Title'] = df['Name'].str.extract('([A-Za-z]+)\.', expand=False)
    #assign a value for missing titles
    df['Title'] = df['Title'].fillna('NoTitle')
    #Unify titles
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
                       
    #mapping values
    df['Embarked'] = df['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    df['Sex'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    df['Title'] = df['Title'].map({'NoTitle': 0, 'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}).astype(int)
    df['CabinCode'] = df['Cabin'].astype(str).str[0]
    df['CabinCode'] = df['CabinCode'].map({'A': 1, 'B': 1, 'C': 1, 'D': 1, 'E':1, 'F':1, 'G': 1, 'T': 0, 'n' : 0}).astype(int)
    
    df = df.set_index('PassengerId')
    df = df.drop(to_be_dropped, axis = 1)
             
    return df
train = prep_data(train)
train.info()
train.head()
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

#split dataframe into X & Y
y = train['Survived']
X = train.drop('Survived', axis = 1)

#create traning & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=7)
#switch off depreciation & user warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 

import lightgbm as lgb 

lgbm = lgb.LGBMClassifier(nthread = 4, boosting_type = 'dart')

#grid search for finding optimum learning rate
param_grid  = {'learning_rate': [0.08, 0.09, 0.1]}
grid_search = GridSearchCV(lgbm, param_grid, scoring='roc_auc', cv=10)
grid_result = grid_search.fit(X, y)

#update model parameters with new learning rate
lgbm.set_params(learning_rate = grid_result.best_params_['learning_rate'])

#using cv function of xgboost to find the optimum number of trees
lgbm_param = lgbm.get_params()
lgbm_train = lgb.Dataset(X_train, y_train,
                         categorical_feature = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title', 'CabinCode'])
cvresult = lgb.cv(lgbm_param, lgbm_train, num_boost_round=1000, metrics=['auc'], early_stopping_rounds=50)

#update model parameters with the resulting number of trees
lgbm.set_params(n_estimators=len(cvresult['auc-mean']))

lgbm.fit(X_train, y_train)
print (lgbm)
# evaluate predictions
accuracy = accuracy_score(y_test, lgbm.predict(X_test))
print("Accuracy: %.2f%%" % (accuracy * 100.0))
#read test set into a pandas dataframe
test = pd.read_csv('../input/test.csv')
test.info()
test.head()
test = prep_data(test)
test.info()
test.head()
#make predictions
predictions = lgbm.predict(test)
#create a submission df from predictions
submission = pd.DataFrame({
        "PassengerId": test.index,
        "Survived": predictions
    })
#output submissions df
submission.to_csv('submission.csv', index=False)