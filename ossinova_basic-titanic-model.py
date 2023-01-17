#This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# importing pckgs

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

import re

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory





for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



# loading tran & test data

train = pd.read_csv("../input/titanic/train.csv") #training set

test = pd.read_csv("../input/titanic/test.csv") #test set



# Store our passenger ID for easy access

PassengerId = test['PassengerId']
# Exploring train set

train.info()

train.describe(include='all')
# Exploring test set

test.info()

test.describe(include='all')
train.isna().sum()
# Taken from user Pegasus - https://www.kaggle.com/ritesh1993/titanic-basic

# Gives the length of the name

train['Name_length'] = train['Name'].apply(len)

test['Name_length'] = test['Name'].apply(len)

# Feature that tells whether a passenger had a cabin on the Titanic

train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)



#train.head(5)

#test.head(5)



full_data = [train, test] # combine dataset (for cleaning/feature engineering on both datasets)
# Feature engineering steps taken from Sina - https://www.kaggle.com/sinakhorami/titanic-best-working-classifier

# Create new feature FamilySize as a combination of SibSp and Parch

for dataset in full_data:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
# Create new feature IsAlone from FamilySize

for dataset in full_data:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

    
# Remove all NULLS in the Embarked column

for dataset in full_data:

    dataset['Embarked'] = dataset['Embarked'].fillna('S')

    
# Remove all NULLS in the Fare column and create a new feature CategoricalFare

for dataset in full_data:

    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())

train['CategoricalFare'] = pd.qcut(train['Fare'], 4)
# Create a New feature CategoricalAge

for dataset in full_data:

    age_avg = dataset['Age'].mean()

    age_std = dataset['Age'].std()

    age_null_count = dataset['Age'].isnull().sum()

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list

    dataset['Age'] = dataset['Age'].astype(int)

train['CategoricalAge'] = pd.cut(train['Age'], 5)
# Define function to extract titles from passenger names

def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    # If the title exists, extract and return it.

    if title_search:

        return title_search.group(1)

    return ""
# Create a new feature Title, containing the titles of passenger names

for dataset in full_data:

    dataset['Title'] = dataset['Name'].apply(get_title)
# Group all non-common titles into one single grouping "Rare"

for dataset in full_data:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

# data cleaning

for dataset in full_data:

    # Mapping Sex

    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    

    # Mapping titles

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)

    

    # Mapping Embarked

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    

    # Mapping Fare

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 						        = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							        = 3

    dataset['Fare'] = dataset['Fare'].astype(int)

    

    # Mapping Age

    dataset.loc[ dataset['Age'] <= 16, 'Age'] 					       = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4 ;
# Feature selection

drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp'] # elements to drop



train = train.drop(drop_elements, axis = 1)

train = train.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)

test  = test.drop(drop_elements, axis = 1)



# Create Numpy arrays of train, test and target ( Survived) dataframes to feed into our models

y = train['Survived'].ravel()

train = train.drop(['Survived'], axis=1)

X = train.values # Creates an array of the train data

test = test.values



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3,random_state=15)





from sklearn.metrics import accuracy_score

y_train.shape, X_test.shape, X_train.shape, y_test.shape
# Random Forest

from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=200)

random_forest.fit(X_train,y_train)

rf_predictions = random_forest.predict(test)



# Evaluate acc

y_pred = random_forest.predict(X_test)

acc_rf = round(accuracy_score(y_pred, y_test) * 100, 2)

print(acc_rf)
# Logistic Regression

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train,y_train)

lg_pred = model.predict(test)



# Evaluate acc

y_pred = model.predict(X_test)

acc_lr = round(accuracy_score(y_pred, y_test) * 100, 2)

print(acc_lr)
# Decision Tree

from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train,y_train)



dt_pred = decision_tree.predict(test)



# Evaluate acc

y_pred = decision_tree.predict(X_test)

acc_dt = round(accuracy_score(y_pred, y_test) * 100, 2)

print(acc_dt)
# import xgboost as xgb



from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score

clf = XGBClassifier()



xgbc_model = XGBClassifier(silent=False, 

                      scale_pos_weight=1,

                      learning_rate=0.01,  

                      colsample_bytree = 0.4,

                      subsample = 0.8,

                      objective='binary:logistic', 

                      n_estimators=1000, 

                      reg_alpha = 0.3,

                      max_depth=4, 

                      gamma=10)

xgbc_model.fit(X_train,y_train,verbose=True)



xgbc_pred = xgbc_model.predict(test)



# Evaluate acc

y_pred = xgbc_model.predict(X_test)

acc_xgbc = round(accuracy_score(y_pred, y_test) * 100, 2)

print(acc_xgbc)
# Gaussian Naive Bayes

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score



gaussian = GaussianNB()

gaussian.fit(X_train, y_train)

gaussian_pred = gaussian.predict(test)



# Evaluate acc

y_pred = gaussian.predict(X_test)

acc_gaussian = round(accuracy_score(y_pred, y_test) * 100, 2)

print(acc_gaussian)
# Gradient Boosting Classifier

from sklearn.ensemble import GradientBoostingClassifier



gbk = GradientBoostingClassifier()

gbk.fit(X_train, y_train)

gbk_pred = gbk.predict(test)



# Evaluate acc

y_pred = gbk.predict(X_test)

acc_gbk = round(accuracy_score(y_pred, y_test) * 100, 2)

print(acc_gbk)
# Generate Submission File

def submit(x,y):

    submission = pd.DataFrame({ 'PassengerId': PassengerId,

                            'Survived': x })

    

    submission.to_csv(f"Submission_{y}.csv", index=False)



    

submit(rf_predictions, 'randomforest') # submit randomforest

submit(lg_pred, 'logReg') # submit logistic regression

submit(dt_pred, 'decisionTree') # submit decision tree

submit(xgbc_pred, 'xgbc') # submit xgbc

submit(gaussian_pred, 'gaussianNB') # submit gaussian naive bayes

submit(gbk_pred, 'gradientBoost') # submit gradient boost classifier