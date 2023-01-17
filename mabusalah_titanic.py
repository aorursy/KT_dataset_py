import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

import pandas as pd,  xgboost

from scipy import stats

from scipy.stats import randint

from sklearn.model_selection import RandomizedSearchCV

from sklearn import metrics

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        #Import Training and Testing Data        

        if filename == 'train.csv':

            train = pd.read_csv(os.path.join(dirname, filename))

            print("Training Set:"% train.columns, train.shape)

        if filename == 'test.csv':        

            test = pd.read_csv(os.path.join(dirname, filename))

            print("Test Set:"% test.columns, test.shape)

# Any results you write to the current directory are saved as output.
#Percentage of Survived/Did not Survive shows that the training Data is not balanced.

print("Survived: ", train.Survived.value_counts()[1]/len(train)*100,"%")

print("Did not Survive: ", train.Survived.value_counts()[0]/len(train)*100,"%")
#Let us have a look at the Anomalies in the Dataset

plt.figure(figsize=(10,4))

plt.xlim(train.Age.min(), train.Age.max()*1.1)

sns.boxplot(x=train.Age)



plt.figure(figsize=(10,4))

plt.xlim(train.Fare.min(), train.Fare.max()*1.1)

sns.boxplot(x=train.Fare)
#We noticed that most of the passengers age is between 20 and 40, Let us see how many passengers are over 65. 

#Also the fare shows that most of people paid less than 100 Pounds per ticket.

train[train['Age']>=65]
test[test['Age']>=65]
test[test['Fare']>=300]
train=train[train['Age']<65]

train=train[train['Fare']<300]
# Obtain target and predictors

y = train.Survived.copy()

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

X = train[features].copy()

X_test = test[features].copy()
# Get list of categorical variables

Obj_Type = (X.dtypes == 'object')

object_cols = list(Obj_Type[Obj_Type].index)



print("Categorical variables:")

print(object_cols)
from sklearn.preprocessing import LabelEncoder



# Apply label encoder to each column with categorical data

label_encoder = LabelEncoder()



for col in object_cols:

    #make NAN as 0 Catgory Variable

    X_test[col] = label_encoder.fit_transform(X_test[col].fillna('0'))

    X[col] = label_encoder.fit_transform(X[col].fillna('0'))
from sklearn.impute import SimpleImputer



# Imputation

my_imputer = SimpleImputer()

imputed_X_test = pd.DataFrame(my_imputer.fit_transform(X_test))

imputed_X = pd.DataFrame(my_imputer.fit_transform(X))

# Imputation removed column names; put them back

imputed_X_test.columns = X_test.columns

imputed_X.columns = X.columns
# Break off validation set from training data

X_train, X_valid, y_train, y_valid = train_test_split(imputed_X, y, train_size=0.8, test_size=0.2,

                                                      random_state=0)
#Return the f1 Score

def train_model(classifier, feature_vector_train, label, feature_vector_valid):

    # fit the training dataset on the classifier

    classifier.fit(feature_vector_train, label)

    

    # predict the labels on validation dataset

    predictions = classifier.predict(feature_vector_valid)    



    return metrics.f1_score(y_valid,predictions)
from sklearn.model_selection import KFold

clf_xgb = xgboost.XGBClassifier(objective = 'binary:logistic')

param_dist = {'n_estimators': stats.randint(140, 1000),

              'learning_rate': stats.uniform(0.01, 0.6),

              'subsample': stats.uniform(0.3, 0.9),

              'max_depth': [3, 4, 5, 6, 7, 8, 9, 12],

              'colsample_bytree': stats.uniform(0.5, 0.9),

              'min_child_weight': [1, 2, 3, 4, 6, 8]

             }



numFolds = 5

kfold_5 = KFold(n_splits = numFolds, shuffle = True)



clf = RandomizedSearchCV(clf_xgb, 

                         param_distributions = param_dist,

                         cv = kfold_5,  

                         n_iter = 5, # you want 5 here not 25 if I understand you correctly 

                         scoring = 'roc_auc', 

                         error_score = 0, 

                         verbose = 3, 

                         n_jobs = -1)

clf.fit(X_train, y_train)
clf.best_params_
clf.best_score_
XGB = train_model(xgboost.XGBClassifier(learning_rate=0.1, n_estimators=169, max_depth=9,

 min_child_weight=6, gamma=0, subsample=0.9, colsample_bytree=0.8),X_train, y_train, X_valid)

print ("Result of XGB: ", XGB)
#Now working with Real challenge Data

pr=xgboost.XGBClassifier(learning_rate=0.1, n_estimators=169, max_depth=9,

 min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8,

 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27)

pr.fit(imputed_X, y)

predictions = pr.predict(imputed_X_test)

d={'PassengerId':test['PassengerId'],'Survived':predictions}

df=pd.DataFrame(data=d)

df.to_csv("test_predictions.csv", index=False)