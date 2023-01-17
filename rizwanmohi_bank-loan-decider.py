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
bank_train = pd.read_csv("../input/Bank_Full_Train_NA.csv")

bank_test = pd.read_csv("../input/Bank_Full_Test_NA.csv")



print("Train ",bank_train.shape)

print("Test ",bank_test.shape)
#viewing few records from the dataset

bank_train.head()
bank_train.education.unique()
print(bank_test.education.unique())

bank_test['education'].replace(np.nan,"quaternary",inplace=True)

bank_test.education.unique()
def inspect_data(data):

    return pd.DataFrame({"Data Type":data.dtypes,"No of Levels":data.apply(lambda x: x.nunique(),axis=0), "Levels":data.apply(lambda x: str(x.unique()),axis=0)})

inspect_data(bank_train)
#Comparing consistancy in both test and train database for levels

def compare_train_test(train_data, test_data):

    train_levels = train_data.apply(lambda x: set(x.unique()),axis=0)

    test_levels = test_data.apply(lambda x: set(x.unique()),axis=0)

    extra = []

    missing = []

    for x1,x2 in zip(train_levels, test_levels):

        missing.append(x1-x2)

        extra.append(x2-x1)

    

    return pd.DataFrame({"Train Data Type":train_data.dtypes, "Test Data Type":test_data.dtypes,

                         "Train #Levels":train_data.apply(lambda x: x.unique().shape[0],axis=0), "Test #Levels":test_data.apply(lambda x: x.unique().shape[0],axis=0),

                         "Test Missing":missing, "Test Extra":extra})

compare_train_test(bank_train, bank_test)
num_cols = ['age','balance','day','duration','campaign','pdays','previous']

cat_cols = bank_train.columns.difference(num_cols).tolist()



print("Numeric Columns ->", num_cols)

print("Categorical Columns ->", cat_cols)
y_train = bank_train['y']

X_train = bank_train.copy().drop('y', axis=1)

y_test = bank_test['y']

X_test = bank_test.copy().drop('y', axis=1)
pd.DataFrame({"Train data null values":X_train.isnull().sum(),"Test data null values":X_test.isnull().sum()})
## Dummifying Categorical variables

X_train = pd.get_dummies(X_train)

X_test = pd.get_dummies(X_test)



print(X_train.shape)

print(X_test.shape)
set(X_train.columns.tolist())-set(X_test.columns.tolist())
X_train_aligned, X_test_aligned = X_train.align(X_test, join='outer', axis=1, fill_value=0)

print(X_train_aligned.shape)

print(X_test_aligned.shape)
from sklearn.preprocessing import Imputer

median_imputer = Imputer(strategy='median')

median_imputer.fit(X_train_aligned[num_cols])



X_train_aligned[num_cols] = median_imputer.transform(X_train_aligned[num_cols])

X_test_aligned[num_cols] = median_imputer.transform(X_test_aligned[num_cols])



pd.DataFrame({"Train":X_train_aligned.isnull().sum(),"Test":X_test_aligned.isnull().sum()})
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train_aligned[num_cols])



X_train_aligned[num_cols] = scaler.transform(X_train_aligned[num_cols])

X_test_aligned[num_cols] = scaler.transform(X_test_aligned[num_cols])
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()

rfc.fit(X = X_train_aligned,y = y_train)
train_predictions = rfc.predict(X_train_aligned)

test_predictions = rfc.predict(X_test_aligned)
from sklearn.metrics import accuracy_score,f1_score

print("\nTrain accuracy",accuracy_score(y_train,train_predictions))

print("\nTrain f1-score for class 'yes'",f1_score(y_train,train_predictions,pos_label="yes"))

print("\nTrain f1-score for class 'no'",f1_score(y_train,train_predictions,pos_label="no"))
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import make_scorer



## n_jobs = -1 uses all cores of processor

## max_features is the maximum number of attributes to select for each tree

rfc_grid = RandomForestClassifier(n_jobs=-1, max_features='sqrt', class_weight='balanced_subsample')

 

# Use a grid over parameters of interest

## n_estimators is the number of trees in the forest

## max_depth is how deep each tree can be

## min_sample_leaf is the minimum samples required in each leaf node for the root node to split

## "A node will only be split if in each of it's leaf nodes there should be min_sample_leaf"



param_grid = {"n_estimators" : [10, 25, 50, 75, 100],

           "max_depth" : [10, 12, 14, 16, 18, 20],

           "min_samples_leaf" : [5, 10, 15, 20],

           "class_weight" : ['balanced','balanced_subsample']}

 

rfc_cv_grid = RandomizedSearchCV(estimator = rfc_grid, param_distributions = param_grid, cv = 3, n_iter=10,

                                scoring = make_scorer(lambda yt,yp: f1_score(yt,yp,pos_label = 'yes')))

rfc_cv_grid.fit(X_train_aligned, y_train)

rfc_cv_grid.best_estimator_
train_predictions = rfc_cv_grid.best_estimator_.predict(X_train_aligned)

test_predictions = rfc_cv_grid.best_estimator_.predict(X_test_aligned)
print("\nTrain accuracy",accuracy_score(y_train,train_predictions))

print("\nTrain f1-score for class 'yes'",f1_score(y_train,train_predictions,pos_label="yes"))

print("\nTrain f1-score for class 'no'",f1_score(y_train,train_predictions,pos_label="no"))
print("\nTest accuracy",accuracy_score(y_test,test_predictions))

print("\nTest f1-score for class 'yes'",f1_score(y_test,test_predictions,pos_label="yes"))

print("\nTest f1-score for class 'no'",f1_score(y_test,test_predictions,pos_label="no"))
rfc_cv_grid.best_estimator_.feature_importances_
## Get important Features

feat_importances = pd.Series(rfc_cv_grid.best_estimator_.feature_importances_, index = X_train_aligned.columns)
## Sort importances  

feat_importances_ordered = feat_importances.nlargest(n=10)

feat_importances_ordered
## Plot Importance

%matplotlib notebook

feat_importances_ordered.plot(kind='bar')