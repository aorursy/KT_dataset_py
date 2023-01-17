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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train_file = '../input/titanic/train.csv'

test_file = '../input/titanic/test.csv'

sub = '../input/titanic/gender_submission.csv'

train_data = pd.read_csv (train_file)

test_data = pd.read_csv (test_file)

sub_data = pd.read_csv (sub)

print (train_data.info ())

print ('\n')

print (train_data.shape, '\t', test_data.shape)
sub_data.head ()
train_data.head ()
train_data.dtypes
sns.countplot (train_data ['Survived'])
print (train_data.shape)

print ('\n')

print (train_data.isnull ().sum ())
train_data ['Age'] = train_data.groupby ('Pclass')['Age'].apply (lambda x : x.fillna (np.mean (x)))

train_data ['Cabin'].fillna (0, inplace = True)

V = train_data ['Embarked'].value_counts ().sort_values (ascending = False).index [0]

train_data ['Embarked'] = train_data ['Embarked'].fillna (V)
print (train_data.isnull ().sum ().sum ())
train_data ['Parch'].value_counts ()
train_data ['SibSp'].value_counts ()
train_data.loc [train_data.Parch != 0, 'Parch'] = 1

train_data.loc [train_data.SibSp != 0, 'SibSp'] = 1

train_data.loc [train_data.Cabin != 0, 'Cabin'] = 1
plt.figure (figsize = (8,12))

plt.subplot (2,1,1)

sns.distplot (train_data ['Age'])

plt.subplot (2,1,2)

sns.distplot (train_data ['Fare'])
print (round (train_data [['Age', 'Fare']].describe (),2))
def combine (df1, df2):

    J = []

    for i in range (df1.shape [0]):

        K = df1[i] +'_' + df2[i]

        J.append (K)

    return pd.Series (J)

train_data ['Age_cat'] = pd.cut (train_data ['Age'], [0,15,35,55,80], labels = ['child', 'young','mature', 'old'])

train_data ['Age_cat_Sex'] = combine (train_data ['Age_cat'], train_data ['Sex'])
def name_length (df):

    J = []

    for i in range (df.shape [0]):

        K = len (df[i])

        J.append (K)

    return pd.Series (J)

train_data ['Name_length'] = name_length (train_data ['Name'])
train_data ['Name_length'].hist ()
def treat_outliers (df):

    K = round (np.mean (df) + np.std (df)*1.8)

    return K



upper_fare = treat_outliers (train_data ['Fare'])

train_data.loc [train_data.Fare > upper_fare, 'Fare'] = upper_fare



upper_name_length = treat_outliers (train_data ['Name_length'])

train_data.loc [train_data.Name_length > upper_name_length, 'Name_length'] = upper_name_length
def ticket_no (df):

    L = []

    for i in range (df.shape [0]):

        P = df.str.split ()[i][-1][0]

        L.append (P)

    return pd.Series (L)

train_data ['ticket_no_first'] = ticket_no (train_data ['Ticket'])
train_data ['ticket_no_first'].unique()
cat_cols = ['Pclass', 'Age_cat_Sex', 'Embarked','Cabin', 'ticket_no_first', 'Parch', 'SibSp']

num_cols = ['Age','Fare', 'Name_length']
plt.figure (figsize = (12,10))

for i, col in enumerate (cat_cols):

    plt.subplot (3,3,i+1)

    sns.countplot (train_data [col], hue = 'Survived', data = train_data)

    plt.xticks (rotation = 90)

    

plt.tight_layout ()
plt.figure (figsize = (12,8))

for i, col in enumerate (num_cols):

    plt.subplot (2,2,i+1)

    sns.boxplot (x = 'Survived', y = col, data = train_data)

train_X = train_data.drop ('Survived', axis = 1)

y = train_data ['Survived']
from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from sklearn.feature_selection import SelectKBest, f_classif

from sklearn import svm

from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV
# Break off validation set from training data

X_train, X_valid, y_train, y_valid = train_test_split(train_X, y, train_size=0.75, test_size=0.25,random_state=0)
# Preprocessing for numerical data

numerical_transformer = Pipeline (steps = [('scaler', MinMaxScaler ())])

# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, num_cols),

        ('cat', categorical_transformer, cat_cols)

    ])

# Define model

#model = svm.SVC ()

#model = XGBClassifier()

model = LogisticRegression ()

#model = RandomForestClassifier()



# Bundle preprocessing and modeling code in a pipeline

clf = Pipeline(steps=[('preprocessor', preprocessor),('model', model)])



#param = {'model__n_estimators': [100, 1000, 10000], 'model__max_depth' : [5,8,10], 'model__random_state' : [0]}

param = {'model__penalty' : ['l1', 'l2'],'model__C' : np.logspace(-4, 4, 20),'model__solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], 'model__random_state' : [0]}

#param = {'model__max_depth' : [5,7,9,11], 'model__learning_rate' : [0.01,0.1,1,10,100], 'model__n_estimators' : [100,1000,10000], 'model__random_state' : [0]}

#param = {'model__C' : [0.1, 1, 10], 'model__gamma' : [0.00001, 0.0001, 0.001, 0.01, 0.1]}



grid = GridSearchCV (clf,param_grid = param, cv = 5, verbose=True, n_jobs=-1)
# Preprocessing of training data, fit model 

grid.fit(X_train, y_train)
grid.best_params_
# Preprocessing for numerical data

numerical_transformer = Pipeline (steps = [('scaler', MinMaxScaler ())])

# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, num_cols),

        ('cat', categorical_transformer, cat_cols)

    ])

# Define model

#model = svm.SVC (C = 100, gamma = 0.1, kernel = 'rbf')

#model = XGBClassifier(max_depth=5, learning_rate=0.01, n_estimators=100, gamma=0, 

#                       min_child_weight=1, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.005)





model = LogisticRegression (C = 1.624, penalty = 'l1', random_state = 0, solver = 'liblinear')

#model = LogisticRegression ()

#param = {'penalty' : ['l1', 'l2'], 'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], 'random_state' : [0]}



#model = RandomForestClassifier(n_estimators = 10000, max_depth = 5, random_state = 0)

#model = XGBRegressor (n_estimators = 1000, learning_rate = 0.01, random_state = 0)

# Bundle preprocessing and modeling code in a pipeline

                      

clf = Pipeline(steps=[('preprocessor', preprocessor),('model', model)])

# Preprocessing of training data, fit model 

clf.fit(X_train, y_train)



# Make validation predictions and calculate mean absolute error

pred_valid = clf.predict(X_valid)
print (confusion_matrix (pred_valid, y_valid))

print (classification_report (pred_valid, y_valid))

print (accuracy_score (pred_valid, y_valid))
test_data.info ()
test_data ['Age'] = test_data.groupby ('Pclass')['Age'].apply (lambda x : x.fillna (np.mean (x)))

test_data ['Cabin'] = test_data ['Cabin'].fillna (0)

test_data ['Fare'] = test_data ['Fare'].fillna (np.mean (test_data ['Fare']))
test_data.loc [test_data.Parch != 0, 'Parch'] = 1

test_data.loc [test_data.SibSp != 0, 'SibSp'] = 1

test_data.loc [test_data.Cabin != 0, 'Cabin'] = 1
test_data ['Age_cat'] = pd.cut (test_data ['Age'], [0,15,35,55,80], labels = ['child', 'young','mature', 'old'])

test_data ['Age_cat_Sex'] = combine (test_data ['Age_cat'], test_data ['Sex'])

test_data ['Name_length'] = name_length (test_data ['Name'])

test_data ['ticket_no_first'] = ticket_no (test_data ['Ticket'])
test_data.loc [test_data.Fare > upper_fare, 'Fare'] = upper_fare

test_data.loc [test_data.Name_length > upper_name_length, 'Name_length'] = upper_name_length
predictions = clf.predict (test_data)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submissiont.csv', index=False)

print("Your submission was successfully saved!")