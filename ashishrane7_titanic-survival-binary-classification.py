import numpy as np
import pandas as pd
import matplotlib.pyplot as pp
import seaborn as sns
%matplotlib inline
sns.set_style('whitegrid')
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_train.head()
df_train.describe()
df_test.head()
# Missing Data
sns.heatmap(df_train.isnull(),cmap='Blues', yticklabels=False, cbar=False)
sns.countplot(x='Survived', data=df_train)
sns.countplot(x='Survived', data=df_train, hue='Sex')
sns.countplot(x='Survived', data = df_train, hue = 'Pclass')
sns.countplot(x='SibSp', data=df_train, hue='Sex')

sns.distplot(df_train['Age'].dropna(), bins=30, color='darkblue',)
sns.distplot(df_train['Fare'], bins=30, color='darkred',kde=False)
# data cleaning
df_train.columns
df_train[df_train['Embarked'].isnull()]
sns.countplot(x='Embarked', data= df_train.loc[df_train['Embarked'].notnull()], hue='Survived')
sns.countplot(x='Embarked', data= df_train.loc[df_train['Embarked'].notnull()], hue='Pclass')
# Another way to look at the data
g = sns.FacetGrid(df_train,col="Sex", row="Embarked", margin_titles=True, hue = "Survived")
g = g.map(pp.hist, "Age", edgecolor = 'white').add_legend();
g = sns.FacetGrid(df_train,col="Pclass", row="Embarked", margin_titles=True, hue = "Survived")
g = g.map(pp.hist, "Age", edgecolor = 'white').add_legend();
df_train.loc[df_train['Embarked'].isnull(), 'Embarked'] = 'S'
# Check if we have any null values for Embarked
df_train['Embarked'].isnull().any()
df_train.loc[df_train['Cabin'].isnull()].groupby(['Pclass']).count()
df_train.loc[(df_train['Cabin'].notnull()) & (df_train['Pclass'] == 2), 'Cabin'].count()
# all first class people have a cabin
df_train.loc[(df_train['Cabin'].isnull()) & (df_train['Pclass'] == 1), 'Cabin'].count()
df_train.loc[(df_train['Cabin'].notnull()) & (df_train['Pclass'] == 3), 'Cabin'].count()
df_train.loc[df_train['Cabin'].isnull() , 'Cabin'] = 'N'
df_test.loc[df_test['Cabin'].isnull() , 'Cabin'] = 'N'
df_train['Cabin'] = df_train['Cabin'].astype(str).str[0].apply(lambda x : x.upper())
df_train['Cabin'].unique()
# Do the same transformation for test
df_test['Cabin'] = df_test['Cabin'].astype(str).str[0].apply(lambda x : x.upper())
df_test['Cabin'].unique()
# Check now if we dont have any more missing values in Cabin
print(df_train.loc[(df_train['Cabin'].isnull()) & (df_train['Pclass'] == 2), 'Cabin'].count())
print(df_train.loc[(df_train['Cabin'].isnull()) & (df_train['Pclass'] == 3), 'Cabin'].count())
sns.boxplot(x='Pclass', y='Age', data=df_train)
# imputation
def imputeAge(cols):
    Pclass = cols[0]
    Age = cols[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 38
        elif Pclass == 2:
            return 29
        elif Pclass == 3:
            return 24
    else:
        return Age


df_train['Age'] = df_train[['Pclass','Age']].apply(imputeAge, axis = 1)
df_train['Age'].isnull().values.any()
# lets confirm that we dont have any missing value
df_train.isnull().any()
df_test.isnull().sum()
# many age values are missing. Use the sample kind of imputation as train data
df_test['Age'] = df_test[['Pclass','Age']].apply(imputeAge, axis = 1)
df_test['Age'].isnull().values.any()
# one of the fare is missing
df_test.loc[df_test['Fare'].isnull()]
fare = df_test.loc[ (df_test['Sex'] == 'male') & (df_test['Embarked'] == 'S') & (df_test['Cabin'] == 'N') & (df_test['Pclass'] == 3) & \
            (df_test['SibSp'] == 0) & (df_test['Parch'] == 0), 'Fare' ].mean()

df_test.loc[df_test['PassengerId'] == 1044, 'Fare'] = fare
df_test.isnull().any()
df_train['LastName'] = df_train['Name'].apply(lambda x: x.split(',')[0])
df_train.drop(['Name'], inplace=True, axis = 1)
df_train.head()
# Repeat same for Test Set
df_test['LastName'] = df_test['Name'].apply(lambda x: x.split(',')[0])
df_test.drop(['Name'], inplace=True, axis = 1)
df_test.head()
df_train = df_train.drop(['Ticket'], axis = 1)
df_test = df_test.drop(['Ticket'], axis = 1)
df_train.head()
# Convert the categorical featues with one hot encoding
features=['Sex', 'Embarked', 'Pclass']
df_train = pd.get_dummies(df_train, columns=features, drop_first=True)
df_test = pd.get_dummies(df_test, columns=features, drop_first=True)

df_train.columns
df_test.columns
from sklearn.preprocessing import LabelEncoder
def encodeLabels(cols):
    for col in cols:
        le = LabelEncoder()
        le = le.fit(df_train[col].values)
        df_train[col] = le.transform(df_train[col].values)
        le = LabelEncoder()
        le = le.fit(df_test[col].values)
        df_test[col] = le.transform(df_test[col].values)
features = ['LastName', 'Cabin']
encodeLabels(features)
df_train.head()
df_test.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_train.drop(['PassengerId', 'Survived'], axis = 1), df_train['Survived'], \
                                                   test_size=0.3, random_state =101)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='liblinear', max_iter=200)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
error_rate = np.mean(y_test != y_pred)
error_rate
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
print(f1_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scale=scaler.fit(X_train)
X_train_scale = scaler.transform(X_train)
svc = SVC()
svc.fit(X_train_scale, y_train)
X_test_scale = scaler.transform(X_test)
y_pred = svc.predict(X_test_scale)
error_rate = (y_test != y_pred).mean()
error_rate
print(f1_score(y_test, y_pred))
print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test, y_pred))
# Lets see if scores are different with kfold splits. Stratified Shuffle split and cross validation
from sklearn.model_selection import KFold, StratifiedShuffleSplit,cross_val_score
from sklearn.metrics import f1_score
X = df_train.drop(['PassengerId','Survived'], axis =1)
y = df_train['Survived']
scores = []
def doKFoldValidation(train):
    kfold = KFold(n_splits=5,shuffle=True, random_state=101)
    for train_index, test_index in kfold.split(train):
        model = LogisticRegression(solver='liblinear', random_state=101)
        X_train, X_test = X.loc[train_index, :], X.loc[test_index, :]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        scores.append(f1_score(y_test, y_pred))
        
doKFoldValidation(df_train)
print(np.mean(scores))
model = LogisticRegression(solver='liblinear', random_state=101)
scores = cross_val_score(model, X, y, scoring='f1',cv=5)
print(np.mean(scores))
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
cv = StratifiedShuffleSplit(n_splits = 5, test_size = .3, random_state = 101 )
param_grid = {'C': [0.1, 0.5, 1, 2, 5 ,10, 50,100, 500,1000], 'penalty': ['l1','l2'] }
model = LogisticRegression(solver='liblinear')
gridSearch = GridSearchCV(model, param_grid=param_grid, cv=3, scoring=['f1', 'accuracy'], refit = 'accuracy' , return_train_score=False)
best_model = gridSearch.fit(X, y)
print(best_model.best_score_)
print(best_model.best_estimator_)
gridSearch.best_params_
print(pd.DataFrame(gridSearch.cv_results_))
model = gridSearch.best_estimator_
y_pred = model.predict(df_test.drop('PassengerId', axis =1 ))
submission = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': y_pred})
submission.head()
submission.to_csv('submission.csv', index=False)
