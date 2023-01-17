import numpy as np  #For numarical calculation
import pandas as pd # This is use dataset related operation
import matplotlib.pyplot as plt  #use for visulaization
import seaborn as sns    # use for 3d visulization
import os
print(os.listdir('../input/titanic'))
# set our Dataframe
train_df = pd.read_csv('../input/titanic/train.csv')
test_df = pd.read_csv('../input/titanic/test.csv')
sub = pd.read_csv('../input/titanic/gender_submission.csv')
train_df.head(2)
train_df.shape
train_df.isnull().sum()
import pandas_profiling as pp
pp.ProfileReport(train_df)
test_df.head(2)
test_df.shape
test_df.isnull().sum()
train_df['Age']=train_df['Age'].fillna(train_df['Age'].median())
test_df.info()
train_df['Embarked']=train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])
test_df['Age']=test_df['Age'].fillna(test_df['Age'].median())
train_df.drop(['PassengerId', 'Name', 'Cabin'], axis=1, inplace=True)
test_df.drop(['PassengerId', 'Name', 'Cabin'], axis=1, inplace=True)
sns.heatmap(train_df.isnull(), yticklabels=False, cbar=False)
sns.heatmap(test_df.isnull(),yticklabels=False, cbar=False)
sns.countplot(x='Survived', hue='Sex', data=train_df)
sns.countplot(x='Embarked' , hue='Survived' , data=train_df)
sns.countplot(x='SibSp', hue='Survived', data=train_df)
plt.figure(figsize=(10,5))
sns.distplot(train_df['Age'], bins=24, color='b')
columns_obj=train_df.select_dtypes('object').columns
from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
train_df[columns_obj]=train_df[columns_obj].apply(LE.fit_transform)
test_df[columns_obj]=test_df[columns_obj].apply(LE.fit_transform)
train_df[columns_obj].head()
train_df.head()
test_df.head()
train_df.corr()
from xgboost import XGBClassifier
regressor=XGBClassifier()
# Machine Learning 
X = train_df.drop(['Survived'], 1).values # select independant features
y = train_df['Survived'].values #select dependant features
from sklearn.preprocessing import StandardScaler

scale = StandardScaler()
scale.fit(X)

X = scale.transform(X)
# Split data to 80% training data and 20% of test to check the accuracy of our model
from sklearn.model_selection import train_test_split, cross_val_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#hyperparameter optamization
booster=['gbtree','gblinear']
base_score=[0.25,0.5,0.75,1]
n_estimators=[100,500,900,1100,1500]
max_depth=[2,3,5,10,15]
booster=['gbtree','gblinear']
learning_rate=[0.05,0.1,0.15,0.20]
min_child_weight=[1,2,3,4]
hyperparameter_grid = {
     
     'n_estimators': n_estimators,
     'max_depth': max_depth,
     'learning_rate': learning_rate,
     'min_child_weight': min_child_weight,
     'booster': booster,
     #'base_score': base_score
 }
from sklearn.model_selection import RandomizedSearchCV
random_cv= RandomizedSearchCV( estimator=regressor, param_distributions=hyperparameter_grid, cv=5,n_iter=50,
                              scoring = 'neg_mean_absolute_error', n_jobs=4,
                             verbose=5, return_train_score=True,
                             random_state=42)
random_cv.fit(X_train,y_train)

       

random_cv.best_estimator_
import xgboost
regressor=xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0,
              learning_rate=0.05, max_delta_step=0, max_depth=3,
              min_child_weight=3, missing=None, n_estimators=100, n_jobs=1,
              nthread=None, objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=1, verbosity=1)


regressor.fit(X_train,y_train)
test_df.head()
# Predict our file test
test_X = test_df.values
test_X = scale.transform(test_X)
y_pred=regressor.predict(test_X)
y_pred
sub.head()
sub.to_csv('submission.csv', index=False)
sub.head()
sub['Survived'] = y_pred # Best Submission (Top 5% LB)
sub.to_csv('xgb_submission.csv', index=False)
sub.head(10)
