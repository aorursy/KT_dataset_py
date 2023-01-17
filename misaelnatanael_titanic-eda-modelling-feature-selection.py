import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df = pd.read_csv('../input/titanic/train.csv')
df
print('PassengerId unique values : %d' % (len(df['PassengerId'].unique())))
print('Name unique values : %d' % (len(df['Name'].unique())))
print('Ticket unique values : %d' % (len(df['Ticket'].unique())))
df = df.drop(columns=['PassengerId', 'Ticket'])
df.isna().sum()
plt.figure(figsize=(6,4))
sns.countplot(x='Survived',data=df)
plt.title('Distribution of passenger survivability')
plt.figure(figsize=(6,4))
sns.countplot(x='Pclass',data=df)
plt.title('Distribution of passenger ticket class')
plt.figure(figsize=(6,4))
sns.countplot(x='Sex',data=df)
plt.title('Distribution of passenger sex')
plt.figure(figsize=(6,4))
sns.countplot(x='SibSp',data=df)
plt.title('Distribution of passenger siblings/spouse relationship')
plt.figure(figsize=(6,4))
sns.countplot(x='Parch',data=df)
plt.title('Distribution of passenger parents/children relationship')
plt.figure(figsize=(6,4))
sns.countplot(x='Embarked',data=df)
plt.title('Distribution of passenger embarkation port')
f, (ax1, ax2) = plt.subplots(figsize=(12,4), nrows=1, ncols=2)
sns.boxplot(df['Age'].dropna(), ax=ax1)
sns.distplot(df['Age'].dropna(), ax=ax2)
f.suptitle('Distribution of passenger age')
f, (ax1, ax2) = plt.subplots(figsize=(12,4), nrows=1, ncols=2)
sns.boxplot(df['Fare'].dropna(), ax=ax1)
sns.distplot(df['Fare'].dropna(), ax=ax2)
f.suptitle('Distribution of passenger fare')
df[['Age','Fare']].describe()
f, (ax1, ax2) = plt.subplots(figsize=(12,4), nrows=1, ncols=2)
sns.boxplot(x='Pclass',y='Fare',data=df, ax=ax1)
sns.boxplot(x='Embarked',y='Fare',data=df, ax=ax2)
ax1.title.set_text('Distribution of ticket fare for each class')
ax2.title.set_text('Distribution of ticket fare for each port')
cols = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']
fig = plt.figure(figsize=(15,10))
for c,i in zip(cols, range(1,6)):
    ax = fig.add_subplot(2,3,i)
    sns.countplot(x='Survived',hue=c,data=df)
    ax.set_title(c)
    ax.legend(loc="upper right") 
fig.tight_layout(pad=4.0) 
fig.suptitle('Passenger survivability based on each categorical variable')
f, (ax1, ax2) = plt.subplots(figsize=(12,4), nrows=1, ncols=2)
sns.boxplot(x='Survived',y='Age',data=df, ax=ax1)
sns.boxplot(x='Survived',y='Fare',data=df, ax=ax2)
ax1.title.set_text('Passenger survivability based on age')
ax2.title.set_text('Passenger survivability based on fare')
df[(df['Name'].str.contains('Duff Gordon')) | (df['Name'].str.contains('Countess')) | (df['Fare'] == df['Fare'].max())]
df.isna().sum()
df = df.drop(columns='Cabin')
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
df['Embarked'] = imp.fit_transform(np.array(df['Embarked']).reshape(-1,1))
from sklearn.impute import KNNImputer
temp = df.copy()
temp = pd.concat([temp,pd.get_dummies(temp[['Sex','Embarked']])],axis=1)
temp = temp.drop(columns=['Sex','Embarked'])

imputer = KNNImputer(n_neighbors=8)
wel = DataFrame(imputer.fit_transform(temp.drop(columns=['Name'])))
df1 = df.copy()
df1['Age'] = wel[2]
df1
df1[df1['Name'].str.contains('Sage,')]
c = df['Name'].str.split(', ').str[1]
title = c.str.split('.').str[0]
Title = title.unique()

fig = plt.figure(figsize=(15,15))
for c,i in zip(Title, range(1,18)):
    df0=df[df['Name'].str.contains(c)].dropna()
    ax = fig.add_subplot(5,5,i)
    sns.boxplot(df0['Age'])
    ax.set_title(c)
fig.tight_layout(pad=4.0)
fig.suptitle('Age distribution among passenger title name')
def age_pred(title):
    return (df[df['Name'].str.contains(title)]['Age'].median())

def age_predict(df):
    c = df['Name'].str.split(', ').str[1]
    title = c.str.split('.').str[0]
    df['title'] = title
    df.loc[df['Age'].isna(),'Age'] = df[df['Age'].isna()]['title'].apply(age_pred)
    del df['title']
    return(df)
df2 = age_predict(df)
df2
df2.isna().sum()
data = df2.drop(columns=['Name'])
data = pd.concat([data,pd.get_dummies(data[['Sex','Embarked']])],axis=1)
data = data.drop(columns=['Sex','Embarked'])
data
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
X = data.iloc[:,1:12]
y = data.iloc[:,0]
from sklearn.tree import DecisionTreeClassifier
param_grid = {'max_depth' : [4, 5, 6, 7, 8], 'ccp_alpha' : [0, 0.001, 0.002, 0.003, 0.004, 0.005]}
grid = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
grid.fit(X, y)
print('Best hyperparameter: ', grid.best_params_)
print('Best cross validation score: ', grid.best_score_)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(grid, X, y, display_labels=['Unsurvived','Survived'], cmap=plt.cm.Blues)
from sklearn.tree import plot_tree
plt.figure(figsize=(20,12))
plot_tree(DecisionTreeClassifier(**grid.best_params_).fit(X,y), feature_names=data.drop(columns='Survived').columns, 
          class_names=['Unsurvived','Survived'], filled=True, fontsize=9)
plt.show()
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
pipe = Pipeline([("scaler", StandardScaler()), ("svc", SVC())])
param_grid = {'svc__C': [0.3, 0.4, 0.5, 0.6, 0.7], 'svc__gamma' : [0.01, 0.05, 0.1, 0.5, 1]}
grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)
grid.fit(X,y)
print('Best hyperparameter: ', grid.best_params_)
print('Best cross validation score: ', grid.best_score_)
plot_confusion_matrix(grid, X, y, display_labels=['Unsurvived','Survived'], cmap=plt.cm.Blues)
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(random_state=0,n_jobs=4)
param_grid = {'n_estimators': [200, 300, 400], 'max_depth' : [8, 9, 10]}
grid = GridSearchCV(forest, param_grid=param_grid, cv=5)
grid.fit(X,y)
print('Best hyperparameter: ', grid.best_params_)
print('Best cross validation score: ', grid.best_score_)
plot_confusion_matrix(grid, X, y, display_labels=['Unsurvived','Survived'], cmap=plt.cm.Blues)
forest.fit(X, y)
indices = np.argsort(forest.feature_importances_)[::-1]
cc = DataFrame({'feature score':Series(forest.feature_importances_),'features':Series(X.columns)})    
plt.figure(figsize=(8,6))
sns.barplot(x='feature score',y='features',data=cc.head(50).sort_values(by='feature score',ascending=False))
new_col = np.array(Series(X.columns[indices]).head(7))
X_new = X[new_col]
grid.fit(X_new,y)
print('Best hyperparameter: ', grid.best_params_)
print('Best cross validation score: ', grid.best_score_)
dt = pd.read_csv('../input/titanic/test.csv')
dt
dt.isna().sum()
dt = dt.drop(columns=['PassengerId','Ticket','Cabin'])
dt = age_predict(dt)
dt.loc[dt['Fare'].isna(),'Fare'] = df[df['Pclass'] == dt[dt['Fare'].isna()]['Pclass'].values[0]]['Fare'].median()
dt
dt.isna().sum()
data_test = dt.drop(columns=['Name'])
data_test = pd.concat([data_test,pd.get_dummies(data_test[['Sex','Embarked']])],axis=1)
data_test = data_test.drop(columns=['Sex','Embarked'])
X_test = data_test[new_col]

y_pred = grid.predict(X_test)
final = pd.concat([dt, Series(y_pred,name='Survived')], axis=1)
final