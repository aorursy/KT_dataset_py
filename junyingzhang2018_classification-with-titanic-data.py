import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('ggplot')
import seaborn as sns
import scipy.stats as stats
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
# Check number of observations in the train and test data
print(train.shape)
print(test.shape)
# check the columns in train and test
print(train.columns)
print(test.columns)
print('\n')
print('Variables in test but not in train is: ', set(train.columns)-set(test.columns))
# Check the first and last five observations of train and test data
train.head()
test.head()
# Check data type and NAN value
print(train.info())
print(test.info())
# Check basic descriptive information for numeric features
train.describe()
sns.countplot(x=train['Survived'])
train['Survived'].value_counts(normalize=True)
sns.countplot(train['Survived'], hue=train['Sex'])
# Add contigency table for Sex by Survived
pd.crosstab(train['Sex'],train['Survived'], normalize='index')
sns.countplot(train['Survived'], hue=train['Pclass'])
# Add contigency table for Pclass by Survived
pd.crosstab(train['Pclass'],train['Survived'], normalize='index')
plt.figure(figsize=(7,7))
plt.hist(train['Age'].dropna(), bins=30)
# We can see that there are some children and very senior people on the boat
print('The minimum age is: ', train['Age'].min())
print('The maximum age is: ', train['Age'].max())

def age_group(Age):
    if Age<5:
        return "Group 1: < 5 Years old"
    elif Age<10:
        return "Group 2: 5-10 Years old"
    elif Age<20:
        return "Group 3: 10-20 Years old"
    elif Age<40:
        return "Group 4: 20-40 Years old"
    elif Age<60:
        return "Group 5: 40-60 Years old"
    else:
        return "Group 6: >= 60 Years old"
train['Age_Group']=train['Age'].apply(age_group)
plt.figure(figsize=(15,7))
sns.countplot(train['Age_Group'], hue=train['Survived'])
pd.crosstab(train['Age_Group'],train['Survived'], normalize='index' )
# Check the number of siblings and spouse
sns.countplot(train['SibSp'])
# Most people do not have siblings or spouses on boat
sns.countplot(train['SibSp'], hue=train['Survived'])
pd.crosstab(train['SibSp'], train['Survived'], normalize='index')
# Check number of parents or children, 
print(train['Parch'].value_counts())
sns.countplot(train['Parch'])
# Most people do not bring parents or children
sns.countplot(train['Parch'], hue=train['Survived'])
pd.crosstab(train['Parch'], train['Survived'], normalize='index')
sns.distplot(train['Fare'], kde=False)
# Create categorical variables for Fare to check whether Survived has relationship with Fare Price
print(train['Fare'].min())
print(train['Fare'].max())
def fare_cat(Fare):
    if Fare<50:
        return "C1: <50"
    elif Fare<100:
        return "C2: <100"
    elif Fare<200:
        return "C3: <200"
    elif Fare<300:
        return "C4: <300"
    else:
        return "C5: >=300"
train['Fare_Cat']=train['Fare'].apply(fare_cat)
print(pd.crosstab(train['Fare_Cat'], train['Survived']))
print(pd.crosstab(train['Fare_Cat'], train['Survived'], normalize='index'))
sns.countplot(train['Embarked'], hue=train['Survived'])
pd.crosstab(train['Embarked'],train['Survived'], normalize='index')
# Drop variables not correlated
train.drop(['Name','Ticket','PassengerId'], axis=1, inplace=True)

TestId=test['PassengerId']
test.drop(['PassengerId', 'Name','Ticket'], axis=1, inplace=True)
train.drop(['Age_Group', 'Fare_Cat'], axis=1, inplace=True)
# use seaborn.heatmap to check NAN values
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
# From the heatmap, Age has some NAN values and most of the Cabin are missing, Embarked has few missing values
# Check missing percentage
train_nan_pct=((train.isnull().sum())/(train.isnull().count())).sort_values(ascending=False)
train_nan_pct[train_nan_pct>0]
# Drop the Cabin columns since too many NAN
train.drop(['Cabin'], axis=1, inplace=True)

# Since Age is skewed, impute with median
train['Age'].fillna(train['Age'].median(), inplace=True)
# Fill the numeric Embarked with mode
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)
# use seaborn.heatmap to check NAN values for test data
sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
# From the heatmap, Age has some NAN values and most of the Cabin are missing, Fare has few missing values
# Check missing percentage
test_nan_pct=((test.isnull().sum())/(test.isnull().count())).sort_values(ascending=False)
test_nan_pct[test_nan_pct>0]
test.drop(['Cabin'], axis=1, inplace=True)
test['Age'].fillna(test['Age'].median(), inplace=True)
test['Fare'].fillna(test['Fare'].median(), inplace=True)

# Check whether impute Age by Pclass will imporve prediction performance
train=pd.get_dummies(train, drop_first=True)
test=pd.get_dummies(test, drop_first=True)
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val=train_test_split(train.drop(['Survived'], axis=1), train['Survived'], test_size=0.3, random_state=100)
from sklearn.linear_model import LogisticRegression
cs=[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300]
score=[]
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
for c in cs:
    lr=LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=c, fit_intercept=True, intercept_scaling=1, class_weight=None)
    lr.fit(X_train, y_train)
    predicted=lr.predict(X_val)
    score.append(accuracy_score(predicted, y_val))
plt.scatter(x=cs, y=score)
score=pd.Series(score, index=cs)
print(score.argmax())
print(score.max())

from sklearn.linear_model import LogisticRegression
cs=np.arange(6,12, 0.2)
score=[]
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
for c in cs:
    lr=LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=c, fit_intercept=True, intercept_scaling=1, class_weight=None)
    lr.fit(X_train, y_train)
    predicted=lr.predict(X_val)
    score.append(accuracy_score(predicted, y_val))
plt.scatter(x=cs, y=score)
score=pd.Series(score, index=cs)
print(score.argmax())
print("The best accuracy score is: ", score.max())
lr=LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=6, fit_intercept=True, intercept_scaling=1, class_weight=None)
lr.fit(train.drop(['Survived'], axis=1), train['Survived'])
test_predicted=lr.predict(test)
submission=pd.DataFrame()
submission['PassengerId']=TestId
submission['Survived']=test_predicted
submission.to_csv('submission.csv', index=False)
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler(copy=True, with_mean=True, with_std=True)
sub_features=train[['Age','Fare']]
sub_features
scaler.fit(sub_features)
scaler.fit_transform(sub_features)
train[['Age','Fare']]=scaler.fit_transform(sub_features)
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val=train_test_split(train.drop(['Survived'], axis=1), train['Survived'], test_size=0.3, random_state=100)
from sklearn.svm import SVC
svc=SVC(C=1, kernel='rbf', tol=0.001)
svc.fit(X_train, y_train)
predicted=svc.predict(X_val)
print(confusion_matrix(y_val, predicted))
print('\n')
print(classification_report(y_val, predicted))
print('\n')
print('Accuracy score is: ', accuracy_score(y_val, predicted))
# The accuracy score is not good by using the above parameters. We will tune the hyperparameters using GridSearchCV
# Create a dictionary called param_grid and fill out some parameters for C and gamma.
# 'gamma':auto has  aaccuracy score of 0.8260
#param_grid = {'C': [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300], 'kernel': ['rbf'], 'gamma':['auto']}
# 'gamma':[0.1] has better aaccuracy of 0.8271
param_grid = {'C': [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300], 'kernel': ['rbf'], 'gamma': [10,1,0.1,0.01,0.001,0.0001]}

from sklearn.model_selection import GridSearchCV
# we can specify scoring='accuracy' (default), 'precision', 'f1', 'recall' to choose parameters
grid=GridSearchCV(SVC(), param_grid, verbose=1,  scoring='accuracy', refit=True)
grid.fit(train.drop(['Survived'], axis=1), train['Survived'])
# The best hyperparameters chosen is
print(grid.best_params_)
print(grid.best_estimator_)
print('Mean cross-validated score of the best_estimator: ', grid.best_score_)
print('The number of cross-validation splits (folds/iterations): ', grid.n_splits_)
# Re-tune the hyperparameters based on previous results C=9, gamma=0.05, 0.8316498316498316
param_grid = {'C': np.arange(1,20), 'kernel': ['rbf'], 'gamma': [0.01,0.03,0.04, 0.05, 0.06, 0.07, 0.1,0.13,0.15,0.17,0.2,0.23,0.25,0.27,0.3]}

from sklearn.model_selection import GridSearchCV
# we can specify scoring='accuracy' (default), 'precision', 'f1', 'recall' to choose parameters
grid=GridSearchCV(SVC(), param_grid, verbose=1,  scoring='accuracy', refit=True)
grid.fit(train.drop(['Survived'], axis=1), train['Survived'])

# The best hyperparameters chosen is
print(grid.best_params_)
print(grid.best_estimator_)
print('Mean cross-validated score of the best_estimator: ', grid.best_score_)
print('The number of cross-validation splits (folds/iterations): ', grid.n_splits_)
test_sub_features=test[['Age','Fare']]
test_sub_features
scaler.fit(test_sub_features)
scaler.fit_transform(test_sub_features)
test[['Age','Fare']]=scaler.fit_transform(test_sub_features)
test.head()
svc=SVC(C=9, gamma=0.05, kernel='rbf', tol=0.001)
svc.fit(train.drop(['Survived'], axis=1), train['Survived'])
test_predicted=svc.predict(test)
submission=pd.DataFrame()
submission['PassengerId']=TestId
submission['Survived']=test_predicted
submission.to_csv('submission.csv', index=False)














