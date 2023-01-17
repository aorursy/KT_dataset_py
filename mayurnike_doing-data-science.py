# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import Counter
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir(os.getcwd()))


# Any results you write to the current directory are saved as output.



# Creating data frames from the train and test data

train_df = pd.read_csv('../input/train.csv', index_col = 'PassengerId')
test_df = pd.read_csv('../input/test.csv', index_col = 'PassengerId')

# pandas dataframe
type(train_df)

# basic info in data frames
train_df.info()

test_df.info()
# Add Survived column to test dataframe
test_df['Survived'] = -1

test_df.info()
# merding the data frames row wise 
df = pd.concat((train_df, test_df), axis = 0)

df.tail()
df.Name
df['Name']
df[['Name', 'Sex', 'Age']]
# location based indexing  (loc for label based indexing)
df.loc[1, ['Fare', 'Cabin', 'Name', 'Age']]

# iloc for position based indexing 
df.iloc[1:3, 3:6]
# filtering
print(len(df[(df['Sex'] == 'male')]))

# filtering with complex logic
print(len(df[(df['Sex'] == 'male') & (df['Pclass'] == 1)]))
df.describe()
print(df['Age'].mean())
print(df['Age'].median())
print(df['Age'].min())
print(df['Age'].max())

print('Age Range : {0}'.format(df['Age'].max() - df['Age'].min()))

print('25 percentile : {0}'.format(df['Age'].quantile(.25)))
print('50 percentile : {0}'.format(df['Age'].quantile(.5)))
print('75 percentile : {0}'.format(df['Age'].quantile(.75)))

print('Age variance : {0}'.format(df["Age"].var()))
print("Age standard deviation : {0}".format(df["Age"].std()))
df["Fare"].plot(kind = "box")
df.describe(include='all')
df['Sex'].value_counts()
df['Sex'].value_counts(normalize = True)
df['Survived'][df['Survived'] != -1].value_counts()
df['Pclass'].value_counts().plot(kind = 'bar', rot = 0, title = "pclass");
df['Pclass'][df['Survived'] == 0].value_counts().plot(kind = 'bar', rot = 0, title = "pclass", color = 'c');
df['Age'].plot(kind='hist', title="Age histo", bins=20);
df['Age'].plot(kind='kde', title="KDE for age");
df['Age'].skew()
df['Fare'].plot(kind='hist', title='Fare histo');
df['Fare'].skew()
df.plot.scatter(x = 'Age', y = 'Fare', title = 'Scatter age vs fare', alpha = 0.1);
df.plot.scatter(x='Pclass', y='Fare', title='Scatter class vs fare', alpha = 0.15);
df['Age'].groupby(df["Sex"]).median()

df[['Fare', 'Age']].groupby(df['Pclass']).median()
df[['Fare', 'Age']].groupby(df['Pclass']).agg({'Fare': 'mean', 'Age': 'median'})
aggr = {
    'Fare' : {
        'mean_fare' : 'mean',
        'median_fare' : 'median',
        'max_fare' : max,
        'min_fare' : min
    },
    'Age' : {
        'mean_age' : 'mean',
        'median_age' : 'median',
        'max_age' : max,
        'min_age' : min,
        'range_age' : lambda x: max(x) - min(x)
    }
}
df[['Fare', 'Age']].groupby(df['Pclass']).agg(aggr)
df['Fare'].groupby([df['Pclass'], df['Embarked']]).median()
pd.crosstab(df['Sex'], df['Pclass']).plot(kind='bar')
pd.crosstab(df['Sex'], df['Pclass'])
df.pivot_table(index="Sex", columns='Pclass', values='Fare', aggfunc='mean')
df.info()
df[df['Embarked'].isnull()]
df['Embarked'][df['Survived'] == 1].value_counts()

pd.crosstab(df['Survived'][df['Survived'] != -1], df['Embarked'][df['Survived'] != -1])
Counter(df['Embarked']).most_common()[0][0]
df['Embarked'][df['Embarked'].isnull()] = Counter(df['Embarked']).most_common()[0][0]
df['Embarked'][df['Survived'] == 1].value_counts()
df['Fare'].groupby([df['Pclass'], df['Embarked']]).median()
df['Embarked'].fillna('C', inplace=True)
df[df['Fare'].isnull()]
df['Fare'][(df['Pclass'] == 3) & (df['Embarked'] == 'S')].median()
df['Fare'].fillna(df['Fare'][(df['Pclass'] == 3) & (df['Embarked'] == 'S')].median(), inplace=True)
df[df['Age'].isnull()]
df['Age'].plot(kind='hist', bins=20)
df['Age'].median()
df['Age'].groupby(df['Sex']).median()
df[df['Age'].notnull()].boxplot('Age', 'Sex')
df[df['Age'].notnull()].boxplot('Age', 'Pclass')
df['Age'].groupby(df['Pclass']).median()
def GetTitle(name):
    first_name = name.split(',')[1]
    title = first_name.split('.')[0].strip().lower()
    return title

df['Name'].map(lambda x: GetTitle(x)).unique()
def GetTitle(name):
    title_grouped = {
        'mr' : 'Mr',
        'mrs' : 'Mrs',
        'miss' : 'Miss',
        'master' : 'Master',
        'don' : 'Sir',
        'rev' : 'Sir',
        'dr' : 'Officer',
        'mme' : 'Mrs',
        'ms' : 'Mrs',
        'major' : 'Officer',
        'lady' : 'Lady',
        'sir' : 'Sir',
        'mlle' : 'Miss',
        'col' : 'Officer',
        'capt' : 'Officer',
        'the countess' : 'Lady',
        'jonkheer' : 'Sir',
        'dona' : 'Lady'
    } 
    first_name = name.split(',')[1]
    title = first_name.split('.')[0].strip().lower()
    return title_grouped[title]

df['Name'].map(lambda x: GetTitle(x)).unique()
# Create new feature in data frame
df['Title'] = df['Name'].map(lambda x: GetTitle(x))
df.head()
df[df['Age'].notnull()].boxplot('Age', 'Title')
df['Age'].groupby(df['Title']).median()
df['Age'].fillna(df['Age'].groupby(df['Title']).transform('median'), inplace=True)
df.info()
df['Age'].plot(kind='hist', bins=20)
df[df['Age'] > 70]
df['Fare'].plot(kind='hist', bins =20)
df['Fare'].plot(kind='box')
df[df['Fare'] == df['Fare'].max()]
np.log(df['Fare'] + 1).plot(kind='hist', bins=20)
pd.qcut(df['Fare'], 4, labels = ['vl', 'l', 'h', 'vh']).value_counts().plot(kind='bar', rot = 0)
df['Fare_Bin'] = pd.qcut(df['Fare'], 4, labels = ['vl', 'l', 'h', 'vh'])
df['AgeState'] = np.where(df['Age'] >= 18, 'Adult', 'Child')
df['AgeState'].value_counts()
#df['Survived'][df['Survived' != -1]]
#df['Fare'][(df['Pclass'] == 3) & (df['Embarked'] == 'S')]
pd.crosstab(df['Survived'][df['Survived'] != -1], df['AgeState'][df['Survived'] != -1])
df['FamilySize'] = df['Parch'] + df['SibSp'] + 1 # 1 for himself
df['FamilySize'].plot(kind='hist')
df[df['FamilySize'] == df['FamilySize'].max()]
pd.crosstab(df['Survived'][df['Survived'] != -1], df['FamilySize'][df['Survived'] != -1])
df.info()
df['Survived'][(df['Sex'] == 'female') & (df['Age'] > 18) & (df['Parch'] > 0) & (df['Title'] != 'Miss')].value_counts()
df['IsMother'] = np.where((df['Sex'] == 'female') & (df['Age'] > 18) & (df['Parch'] > 0) & (df['Title'] != 'Miss'), 1, 0)
df['IsMother'].value_counts()
pd.crosstab(df['Survived'][df['Survived'] != -1], df['IsMother'][df['Survived'] != -1])
df['Cabin'].unique()
df[df['Cabin'] == 'T']
df.loc[df['Cabin'] == 'T', 'Cabin'] = np.NaN
#df['Cabin'].map(lambda x: str(x)[0].upper())
df['deck'] = np.where(df['Cabin'].notnull(), df['Cabin'].map(lambda x: str(x)[0].upper()), 'Z')
df['deck'].value_counts()
pd.crosstab((df['Survived'][df['Survived'] != -1]), (df['deck'][df['Survived'] != -1]))
df.info()
df['Is_Male'] = np.where(df['Sex'] == 'male', 1, 0)
df = pd.get_dummies(df, columns = ['deck', 'Pclass', 'Title', 'Fare_Bin', 'Embarked', 'AgeState'])
df.drop(['Cabin', 'Name', 'Ticket', 'Parch', 'SibSp', 'Sex'], axis=1, inplace=True)
columns = [column for column in df.columns if column != 'Survived']
columns = ['Survived'] + columns
df = df[columns]
df.info()
df[df['Survived'] != -1].to_csv('train_processed.csv')
columns = [column for column in df.columns if column != 'Survived']
df.loc[df['Survived'] == -1, columns].to_csv('test_processed.csv')
import matplotlib.pyplot as plt
f, ax = plt.subplots()
ax.hist(df['Age'], bins=20, color='r')
ax.set_xlabel('Bins')
ax.set_ylabel('Count')
ax.set_title('Age histo')
plt.show()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,3))
ax1.hist(df['Fare'], bins=20, color='tomato')
ax1.set_xlabel('Bins')
ax1.set_ylabel('Count')
ax1.set_title('Fare histo')

ax2.hist(df['Age'], bins=20, color='g')
ax2.set_xlabel('Bins')
ax2.set_ylabel('Count')
ax2.set_title('Age histo')
plt.show()
f, ax = plt.subplots(3, 2, figsize=(14,7))
ax[0,0].hist(df['Fare'], bins=20, color='tomato')
ax[0,0].set_xlabel('Bins')
ax[0,0].set_ylabel('Count')
ax[0,0].set_title('Fare histo')

ax[0,1].hist(df['Age'], bins=20, color='g')
ax[0,1].set_xlabel('Bins')
ax[0,1].set_ylabel('Count')
ax[0,1].set_title('Age histo')

ax[1,0].boxplot(df['Fare'].values)
ax[1,0].set_xlabel('Fare')
ax[1,0].set_ylabel('Fare')
ax[1,0].set_title('Fare box')

ax[1,1].boxplot(df['Age'].values)
ax[1,1].set_xlabel('Age')
ax[1,1].set_ylabel('Age')
ax[1,1].set_title('Age box')

ax[2,0].scatter(df['Age'], df['Fare'])
ax[2,0].set_xlabel('Age')
ax[2,0].set_ylabel('Fare')
ax[2,0].set_title('scatter age vs fare')

ax[2,1].axis('off')
plt.tight_layout()
plt.show()
train_pro_df = pd.read_csv('train_processed.csv', index_col='PassengerId')
test_pro_df = pd.read_csv('test_processed.csv', index_col='PassengerId')
train_pro_df.info()
test_pro_df.info()
X = train_pro_df.loc[:, 'Age':].as_matrix().astype('float')
Y = train_pro_df['Survived'].ravel()
print(X.shape, Y.shape)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)
np.mean(Y_test)
import sklearn
sklearn.__version__
from sklearn.dummy import DummyClassifier
model_dummy = DummyClassifier(strategy='most_frequent', random_state=0)
model_dummy.fit(X_train, Y_train)
model_dummy.score(X_test, Y_test)
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
accuracy_score(Y_test, model_dummy.predict(X_test))
confusion_matrix(Y_test, model_dummy.predict(X_test))
precision_score(Y_test, model_dummy.predict(X_test))
recall_score(Y_test, model_dummy.predict(X_test))
test_X = test_pro_df.as_matrix().astype('float')

model_dummy.predict(test_X)
df_submission_base = pd.DataFrame({'PassengerId' : test_pro_df.index, 'Survived' : model_dummy.predict(test_X) })
df_submission_base.to_csv('dummy_base.csv', index=False)
from sklearn.linear_model import LogisticRegression
model_lr_1 = LogisticRegression(random_state=0)
model_lr_1.fit(X_train, Y_train)
model_lr_1.score(X_test, Y_test)
model_lr_1.predict(X_test)
print(accuracy_score(Y_test, model_lr_1.predict(X_test)))

print(confusion_matrix(Y_test, model_lr_1.predict(X_test)))

print(precision_score(Y_test, model_lr_1.predict(X_test)))

print(recall_score(Y_test, model_lr_1.predict(X_test)))

model_lr_1.coef_
train_pro_df = pd.read_csv('train_processed.csv', index_col='PassengerId')
test_pro_df = pd.read_csv('test_processed.csv', index_col='PassengerId')
X = train_pro_df.loc[:, 'Age':].as_matrix().astype('float')
Y = train_pro_df['Survived'].ravel()
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)
test_X_lr = test_pro_df.as_matrix().astype('float')
model_lr_1.predict(test_X_lr)
df_submission_lr = pd.DataFrame({'PassengerId' : test_pro_df.index, 'Survived' : model_lr_1.predict(test_X_lr) })
df_submission_lr.to_csv('model_lr_1.csv', index=False)
model_lr_2 = LogisticRegression(random_state=0)
from sklearn.model_selection import GridSearchCV
parameters = { 'C' : [1.0, 10.0, 50.0, 100.0, 1000.0], 'penalty' : ['l1', 'l2'] } 
clf = GridSearchCV(model_lr_2, param_grid = parameters, cv=3)
clf.fit(X_train, Y_train)
clf.best_params_
clf.score(X_test, Y_test)
test_X_lr_2 = test_pro_df.as_matrix().astype('float')
clf.predict(test_X_lr_2)
df_submission_lr_2 = pd.DataFrame({'PassengerId' : test_pro_df.index, 'Survived' : clf.predict(test_X_lr_2) })
df_submission_lr_2.to_csv('model_lr_2.csv', index=False)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_train_scaled[:,0].max()
X_test_scaled = scaler.fit_transform(X_test)
scaler = StandardScaler()
X_train_scaled_std = scaler.fit_transform(X_train)
X_test_scaled_std = scaler.fit_transform(X_test)
model_lr_3 = LogisticRegression(random_state=0)
from sklearn.model_selection import GridSearchCV
parameters = { 'C' : [1.0, 10.0, 50.0, 100.0, 1000.0], 'penalty' : ['l1', 'l2'] } 
clf = GridSearchCV(model_lr_2, param_grid = parameters, cv=3)
clf.fit(X_train_scaled, Y_train)
clf.score(X_train_scaled, Y_train)
#clf.fit(X_train_scaled_std, Y_train)
#clf.score(X_train_scaled_std, Y_train)
import pickle
model_file_pickle = open('lr_model.pkl', 'wb')
scaler_file_pickle = open('lr_scaler.pkl', 'wb')
pickle.dump(clf, model_file_pickle)
pickle.dump(scaler, scaler_file_pickle)
model_file_pickle.close()
scaler_file_pickle.close
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.

from sklearn import preprocessing, cross_validation, svm, neighbors
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import sklearn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import svm
train_pro_df = pd.read_csv('../input/post-processing-titanic/train_processed.csv', index_col='PassengerId')
test_pro_df = pd.read_csv('../input/post-processing-titanic/test_processed.csv', index_col='PassengerId')
test_pro_df.drop(['Survived'],1 , inplace=True)
test_pro_df.head()
X = train_pro_df.loc[:, 'Age':].as_matrix().astype('float')
y = train_pro_df['Survived'].ravel()
test_X = test_pro_df.as_matrix().astype('float')
def knn_model(X, y):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.21, random_state=0)
    model_knn = neighbors.KNeighborsClassifier(n_neighbors=50)
    model_knn.fit(X_train, y_train)
    print(model_knn.score(X_train, y_train))
    
    #print(accuracy_score(y_test, model_knn.predict(X_test)))
    #print(confusion_matrix(y_test, model_knn.predict(X_test)))
    #print(precision_score(y_test, model_knn.predict(X_test)))
    #print(recall_score(y_test, model_knn.predict(X_test)))
    
    knn_pred = model_knn.predict(test_X)
    return knn_pred
knn_pred = knn_model(X, y)
knn_submission = pd.DataFrame({'PassengerId' : test_pro_df.index, 'Survived' : knn_pred })
knn_submission.to_csv('knn_titanic.csv', index=False)
def svm_model(X, y):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25, random_state=0)
    model_svm = svm.SVC()
    model_svm.fit(X_train, y_train)
    print(model_svm.score(X_train, y_train))
    
    #print(accuracy_score(y_test, model_knn.predict(X_test)))
    #print(confusion_matrix(y_test, model_knn.predict(X_test)))
    #print(precision_score(y_test, model_knn.predict(X_test)))
    #print(recall_score(y_test, model_knn.predict(X_test)))
    
    svm_pred = model_svm.predict(test_X)
    return svm_pred
svm_pred = svm_model(X, y)
svm_submission = pd.DataFrame({'PassengerId' : test_pro_df.index, 'Survived' : knn_pred })
svm_submission.to_csv('svm_titanic.csv', index=False)
