import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Pandas options
pd.set_option('display.max_colwidth', 1000, 'display.max_rows', None, 'display.max_columns', None)

# Plotting options
%matplotlib inline
mpl.style.use('ggplot')
sns.set(style='whitegrid')
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_train.shape
df_test.shape
TestPassengerId = df_test.loc[:,'PassengerId']
df = df_train.append(df_test, ignore_index=True)
df.shape
df.info()
def incomplete_cols(df):
    """
    Returns a list of incomplete columns in df and their fraction of non-null values.
    
    Input: pandas DataFrame
    Returns: pandas Series
    """
    cmp = df.notnull().mean().sort_values()
    return cmp.loc[cmp<1]
incomplete_cols(df)
df.sample(5) # Display some random rows
df['Age'].notnull().mean()
df['Age_NA'] = df['Age'].isnull()
plt.figure(figsize=(12,4), dpi=90)
sns.distplot(df.loc[df['Age'].notnull(), 'Age'], bins=range(0, 90, 2), kde=False)
plt.ylabel('Count')
plt.title('Histogram of Passenger Age')
df['Cabin'].notnull().mean()
def find_cabin(s):
    try:
        return s[0]
    except:
        return 'NA'
df.loc[:,'Cabin'] = df['Cabin'].apply(find_cabin)
df['Cabin'].value_counts()
df['Embarked'].value_counts(dropna=False)
sns.countplot(x='Embarked', data=df)
plt.title('Passenger Ports of Embarkation')
df['Fare'].isnull().sum()
df.loc[df['Fare'].isnull()]
plt.figure(figsize=(12,4), dpi=90)
sns.distplot(df.loc[df['Fare'].notnull(), 'Fare'], kde=False)
plt.ylabel('Count')
plt.title('Histogram of Passenger Fares')
df['Fare'].skew()
df['Name'].notnull().mean()
df['Name'].sample(5)
df['Title'] = df['Name'].apply(lambda s: s.split(', ')[1].split(' ')[0])
df['Title'].nunique()
df[['Name', 'Title']].sample(10)
df['Title'].value_counts()
df.loc[df['Title']=='the']
df.drop(labels=['Name','Title'], axis=1, inplace=True)
df['Parch'].notnull().mean()
df['Parch'].value_counts()
plt.figure(dpi=80)
sns.countplot(x='Parch', data=df)
plt.title('Number of Parents/Children')
df.shape[0]
df['PassengerId'].nunique()
df['Pclass'].notnull().mean()
df['Pclass'].value_counts()
plt.figure(figsize=(4,4), dpi=90)
sns.countplot(x='Pclass', data=df)
plt.title('Passenger Ticket Class')
df['Sex'].notnull().mean()
df['Sex'].value_counts(normalize=True)
df['SibSp'].notnull().mean()
df['SibSp'].value_counts()
plt.figure(dpi=90)
sns.countplot(x='SibSp', data=df)
plt.title('Number of Siblings/Spouses')
df.loc[df['Survived'].notnull(), 'Survived'].value_counts(normalize=True)
df['Ticket'].notnull().mean()
(df['Ticket'].nunique(), df.shape[0])
df['Ticket'].sample(10)
def ticket_prefix(s):
    'Find the content of the ticket before the ticket number'
    temp = s.split(' ')
    if len(temp) > 1:
        return ' '.join(temp[:-1])
    else:
        return 'NONE'
df.loc[:,'Ticket Prefix'] = df['Ticket'].apply(ticket_prefix)
df['Ticket Prefix'].nunique()
df['Ticket Prefix'].value_counts()
df.loc[:,'Ticket Prefix'] = df['Ticket Prefix'].apply(lambda s: s.replace('.','').replace('/','').replace(' ','').upper())
df['Ticket Prefix'].nunique()
df['Ticket Prefix'].value_counts()
vals = df['Ticket Prefix'].value_counts()

def other_prefix(prefix):
    if vals[prefix] < 10:
        return "OTHER"
    else:
        return prefix
df.loc[:,'Ticket Prefix'] = df['Ticket Prefix'].apply(other_prefix)
df['Ticket Prefix'].value_counts()
def ticket_number(s):
    'Find the ticket number on a ticket'
    try:
        return np.int64(s.split(' ')[-1])
    except:
        return np.nan
df['Ticket Number'] = df['Ticket'].apply(ticket_number)
df['Ticket Number'].isnull().sum()
df.loc[df['Ticket Number'].isnull()]
plt.figure(figsize=(12,4), dpi=90)
sns.distplot(df.loc[df['Ticket Number'].notnull(), 'Ticket Number'], bins=400, kde=False)
plt.ylabel('Count')
plt.title('Histogram of Ticket Number')
plt.figure(figsize=(12,4), dpi=90)
sns.distplot(df.loc[df['Ticket Number'].notnull(), 'Ticket Number'], bins=500, kde=False)
plt.xlim([0, 500000])
plt.ylabel('Count')
plt.title('Histogram of Ticket Number')
def tick_num_class(tick_num):
    if tick_num < 100000:
        return 'A'
    elif tick_num < 200000:
        return 'B'
    elif tick_num < 300000:
        return 'C'
    elif tick_num < 400000:
        return 'D'
    elif tick_num >= 400000:
        return 'E'
    else:
        return 'NA'
df.loc[:,'Ticket Number'] = df['Ticket Number'].apply(tick_num_class)
df['Ticket Number'].value_counts()
df.drop(labels='Ticket', axis=1, inplace=True)
df.sample(5)
df.info()
incomplete_cols(df)
df.shape
df = pd.get_dummies(df, drop_first=True)
df.shape
df.sample(5)
X = df.drop(labels='Survived', axis=1)
y = df.loc[:,'Survived']
incomplete_cols(X)
from sklearn.preprocessing import Imputer
imputer = Imputer().fit(X)
X = pd.DataFrame(imputer.transform(X), columns=X.columns)
incomplete_cols(X)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X)
X = pd.DataFrame(scaler.transform(X), columns=X.columns)
X.sample(5)
X_train = X.loc[y.notnull()]
X_test = X.loc[y.isnull()]
y_train = y[y.notnull()].apply(np.bool)
from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFECV
selector = RFECV(estimator=LinearSVC(), scoring='accuracy', verbose=0, n_jobs=-1)
selector.fit(X_train, y_train)
selector.n_features_
X_train.columns[selector.support_]
selector.score(X_train, y_train)
X_train = X_train.loc[:,selector.support_]
X_test = X_test.loc[:,selector.support_]
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
param_grid = {'n_estimators': [10, 55, 100],
              'max_features': ['log2', 'sqrt', None],
              'max_depth': [5, 15, 30, None],
              'min_samples_split': [2, 10, 50],
              'min_samples_leaf': [1, 5, 10]
             }
model_rfc = GridSearchCV(estimator=RandomForestClassifier(n_jobs=-1), param_grid=param_grid, scoring=make_scorer(accuracy_score), n_jobs=-1, verbose=1)
model_rfc.fit(X_train, y_train)
model_rfc.best_params_
model_rfc.best_score_
plt.figure(figsize=(4,4), dpi=90)
sns.barplot(y=X_train.columns, x=model_rfc.best_estimator_.feature_importances_, color='darkblue', orient='h')
plt.xlabel('RF Feature Importance')
plt.ylabel('Feature')
plt.title('Random Forest Feature Importances')
from sklearn.ensemble import GradientBoostingClassifier
param_grid = {'max_depth': [3, 12, 25],
              'subsample': [0.6, 0.8, 1.0],
              'max_features': [None, 'sqrt', 'log2']
             }
model_gb = GridSearchCV(estimator=GradientBoostingClassifier(), param_grid=param_grid, scoring=make_scorer(accuracy_score), n_jobs=-1, verbose=1)
model_gb.fit(X_train, y_train)
model_gb.best_params_
model_gb.best_score_
from sklearn.linear_model import LogisticRegression
param_grid = {'penalty': ['l2'],
              'C': [10**k for k in range(-3,3)],
              'class_weight': [None, 'balanced'],
              'warm_start': [True]
             }
model_logreg = GridSearchCV(estimator=LogisticRegression(), param_grid=param_grid, scoring=make_scorer(accuracy_score), n_jobs=-1, verbose=1)
model_logreg.fit(X_train, y_train)
model_logreg.best_params_
model_logreg.best_score_
from sklearn.naive_bayes import GaussianNB
model_gnb = GaussianNB()
model_gnb.fit(X_train, y_train)
accuracy_score(y_train, model_gnb.predict(X_train))
from sklearn.svm import SVC
param_grid = {'C': [10**k for k in range(-3,4)],
              'class_weight': [None, 'balanced'],
              'shrinking': [True, False]
             }
model_svc = GridSearchCV(estimator=SVC(), param_grid=param_grid, scoring=make_scorer(accuracy_score), n_jobs=-1, verbose=1)
model_svc.fit(X_train, y_train)
model_svc.best_params_
model_svc.best_score_
from sklearn.neighbors import KNeighborsClassifier
param_grid = {'n_neighbors': [1, 2, 4, 8, 16, 32, 64, 128, 256],
              'weights': ['uniform', 'distance'],
              'p': [1, 2, 3, 4, 5]
             }
model_knn = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=param_grid, scoring=make_scorer(accuracy_score), n_jobs=-1, verbose=1)
model_knn.fit(X_train, y_train)
model_knn.best_params_
model_knn.best_score_
print('Training Accuracy Scores')
print('Random Forest: ', model_rfc.best_score_)
print('Gradient Boosting: ', model_gb.best_score_)
print('Logistic Regression: ', model_logreg.best_score_)
print('Gaussian Naive Bayes: ', accuracy_score(y_train, model_gnb.predict(X_train)))
print('Support Vector Classifier: ', model_svc.best_score_)
print('k-Nearest Neighbors: ', model_knn.best_score_)
y_preds = model_rfc.predict(X_test)
submission = pd.DataFrame({'PassengerId':TestPassengerId, 'Survived':np.uint8(y_preds)})
submission.shape
submission.sample(5)
submission.to_csv('my_submission.csv', index=False)