import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
train_df.head()
def transform_df(df):
    #do all the crazy stuff, magic, polymerizations with df
    
    return df
train_df.describe()
train_df.head(20)
train_df.info()
train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[['Embarked','Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df.drop(['PassengerId','Ticket', 'Cabin'], 1, inplace=True)
train_df.head()
test_df.drop(['PassengerId', 'Ticket', 'Cabin'], 1, inplace=True)
test_df.head()
test_df.info()
train_df['Age'].hist()
# There are 177 null values, practically 1/5 of the dataset have nan ages!
print('Count:', train_df.shape[0])
print('Null age count:', train_df['Age'].isnull().sum())
print(train_df['Age'].isnull().sum() / train_df.shape[0] * 100,'% of ages are NAN', sep="")
age_mean = train_df['Age'].mean()
age_std = train_df['Age'].std()
age_nan_count = train_df['Age'].isnull().sum()
age_mean, age_std, age_nan_count
rand_ages = np.random.randint(age_mean - age_std, age_mean + age_std, age_nan_count)
fig, (axis1, axis2) = plt.subplots(1,2, figsize=(15,4))
axis1.set_title = 'Original values of ages'
axis2.set_title = 'New values of ages'

# Original values without NAN
train_df['Age'].dropna().astype(float).hist(bins=70, ax=axis1)

# New values with random ages
train_df['Age'][np.isnan(train_df['Age'])] = rand_ages
train_df['Age'].hist(bins=70, ax=axis2)
print('Now there are {} null values!'.format( train_df['Age'].isnull().sum()) )
train_df['AgeCategorical'] = pd.cut(train_df['Age'], bins=8)
train_df['AgeCategorical'].value_counts()

relation_age_survived = train_df[['AgeCategorical', 'Survived']].groupby('AgeCategorical', as_index=False).mean()
fig, axis1 = plt.subplots(1,1,figsize=(15,5))
sns.barplot(x='AgeCategorical',y='Survived', data=relation_age_survived)
train_df.loc[train_df['Age'] < 10.367, 'Age'] = 0
train_df.loc[(train_df['Age'] >= 10.367) & (train_df['Age'] < 20.315), 'Age'] = 1
train_df.loc[(train_df['Age'] >= 20.315) & (train_df['Age'] < 30.263), 'Age'] = 2
train_df.loc[(train_df['Age'] >= 30.263) & (train_df['Age'] < 40.21), 'Age'] = 3
train_df.loc[(train_df['Age'] >= 40.21) & (train_df['Age'] < 50.157), 'Age'] = 4
train_df.loc[(train_df['Age'] >= 50.157) & (train_df['Age'] < 60.105), 'Age'] = 5
train_df.loc[(train_df['Age'] >= 60.105) & (train_df['Age'] < 70.0525), 'Age'] = 6
train_df.loc[(train_df['Age'] >= 70.0525) & (train_df['Age'] <= 80), 'Age'] = 7

train_df['Age'] = train_df['Age'].astype(int)

# AgeCategorical isn't necessary anymore
train_df.drop('AgeCategorical', 1, inplace=True)

train_df.head()
train_df['Name'] = train_df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
train_df['Name'].value_counts()
pd.crosstab(train_df['Sex'], train_df['Name'])
train_df['Name'] = train_df['Name'].replace(['Capt', 'Dr','Rev','Mile','Col','Major','Countess','Jonkheer','Mme',\
                                            'Don', 'Ms','Sir','Capt','Lady', 'Mlle'], 'Low Appearence')
train_df['Name'].value_counts()
train_df[['Name','Survived']].groupby('Name', as_index=False).mean().sort_values(by='Survived', ascending=False)
#in test_df, we need to fillna
train_df['Name'] = train_df['Name'].map({'Mr':0, 'Low Appearence': 1, 'Master': 2, 'Miss': 3, 'Mrs': 4})
sns.barplot(x='Name', y='Survived', data=train_df)
train_df['Family'] = train_df['SibSp'] + train_df['Parch']
train_df.drop(['SibSp','Parch'], 1, inplace=True)
family_survived_relation = train_df[['Family','Survived']].groupby('Family', as_index=False).mean()
family_survived_relation.sort_values(by='Survived', ascending=False)
sns.barplot(x='Family', y='Survived', data=family_survived_relation)
train_df['Alone'] = train_df['Family'].copy()
train_df['Alone'].loc[train_df['Alone'] == 0] = -1
train_df['Alone'].loc[train_df['Alone'] > 0] = 0

train_df['Alone'].loc[train_df['Alone'] == -1] = 1
relation_alone_survived = train_df[['Alone','Survived']].groupby('Alone', as_index=False).mean()\
.sort_values(by='Alone', ascending=False)
train_df['Alone'].value_counts()
sns.barplot(x='Alone', y='Survived', data=relation_alone_survived)
train_df.drop('Alone', 1, inplace=True)
train_df.head()
train_df.info()
print('There are {} null values in Embarked'.format(train_df['Embarked'].isnull().sum()))

#Seeing the values
train_df['Embarked'].value_counts()
embarked_most_common = train_df['Embarked'].value_counts().idxmax()
train_df['Embarked'].fillna(embarked_most_common, inplace=True)

print('Now there are {} null values in Embarked'.format(train_df['Embarked'].isnull().sum()))
train_df[['Embarked','Survived']].groupby('Embarked', as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df['Embarked'] = train_df['Embarked'].map({'S': 0, 'Q': 1, 'C': 2}).astype(int)
test_df.info()
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
train_df['Fare'].hist(bins=4)
facet = sns.FacetGrid(train_df, col='Survived', aspect=2)
facet.map(plt.hist, 'Fare')
facet.set(xlim=(0, train_df['Fare'].max()))
facet.add_legend
train_df['FareCategorical'] = pd.qcut(train_df['Fare'], 4)
train_df['FareCategorical'].value_counts()
train_df[['FareCategorical','Survived']].groupby('FareCategorical', as_index=False).mean().\
sort_values(by='Survived',ascending=False)
train_df.loc[train_df['Fare'] < 7.91, 'Fare'] = 0
train_df.loc[(train_df['Fare'] >= 7.91) & (train_df['Fare'] < 14.454), 'Fare'] = 1
train_df.loc[(train_df['Fare'] >= 14.454) & (train_df['Fare'] < 31), 'Fare'] = 2
train_df.loc[train_df['Fare'] >= 31, 'Fare'] = 3

train_df['Fare'] = train_df['Fare'].astype(int)
train_df.drop('FareCategorical', 1, inplace=True)
train_df.head()
train_df['Sex'] = train_df['Sex'].map( {'male': 0, 'female': 1} )
train_df.head()
def transform_df(df):
    #do all the crazy stuff, magic, polymerizations with df
    df.drop(['PassengerId','Ticket', 'Cabin'], 1, inplace=True)
    
    age_mean = df['Age'].mean()
    age_std = df['Age'].std()
    age_nan_count = df['Age'].isnull().sum()
    
    rand_ages = np.random.randint(age_mean - age_std, age_mean + age_std, age_nan_count)
    df['Age'][np.isnan(df['Age'])] = rand_ages
    
    #df['Age'] = pd.cut(df['Age'], bins=8)
    
    df.loc[df['Age'] < 10.367, 'Age'] = 0
    df.loc[(df['Age'] >= 10.367) & (df['Age'] < 20.315), 'Age'] = 1
    df.loc[(df['Age'] >= 20.315) & (df['Age'] < 30.263), 'Age'] = 2
    df.loc[(df['Age'] >= 30.263) & (df['Age'] < 40.21), 'Age'] = 3
    df.loc[(df['Age'] >= 40.21) & (df['Age'] < 50.157), 'Age'] = 4
    df.loc[(df['Age'] >= 50.157) & (df['Age'] < 60.105), 'Age'] = 5
    df.loc[(df['Age'] >= 60.105) & (df['Age'] < 70.0525), 'Age'] = 6
    df.loc[(df['Age'] >= 70.0525) & (df['Age'] <= 80), 'Age'] = 7

    df['Age'] = df['Age'].astype(int)
    
    df['Name'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    df['Name'] = df['Name'].replace(['Capt', 'Dr','Rev','Mile','Col','Major','Countess','Jonkheer','Mme',\
                                            'Don', 'Ms','Sir','Capt','Lady', 'Mlle'], 'Low Appearence')
    df['Name'] = df['Name'].map({'Mr':0, 'Low Appearence': 1, 'Master': 2, 'Miss': 3, 'Mrs': 4})
    
    df['Name'] = df['Name'].fillna(4).astype(int)
    
    df['Family'] = df['SibSp'] + df['Parch']
    df.drop(['SibSp','Parch'], 1, inplace=True)

    '''df['Alone'] = df['Family'].copy()
    df['Alone'].loc[df['Alone'] == 0] = -1
    df['Alone'].loc[df['Alone'] > 0] = 0

    df['Alone'].loc[df['Alone'] == -1] = 1'''

    #df.drop('Family', 1, inplace=True)
    
    embarked_most_common = df['Embarked'].value_counts().idxmax()
    df['Embarked'].fillna(embarked_most_common, inplace=True)
    
    df['Embarked'] = df['Embarked'].map({'S': 0, 'Q': 1, 'C': 2}).astype(int)
    
    df['Fare'].fillna(df['Fare'].dropna().median(), inplace=True)
    #df['Fare'] = pd.qcut(df['Fare'], 4)
    
    df.loc[df['Fare'] < 7.91, 'Fare'] = 0
    df.loc[(df['Fare'] >= 7.91) & (df['Fare'] < 14.454), 'Fare'] = 1
    df.loc[(df['Fare'] >= 14.454) & (df['Fare'] < 31), 'Fare'] = 2
    df.loc[df['Fare'] >= 31, 'Fare'] = 3

    df['Fare'] = df['Fare'].astype(int)
    
    df['Sex'] = df['Sex'].map( {'male': 0, 'female': 1} )
    
    return df
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
train_df = transform_df(train_df)
test_df = transform_df(test_df)
train_df.head()
test_df.head()
from sklearn.model_selection import cross_val_score
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from xgboost import XGBClassifier
X_train = train_df.drop('Survived', 1)
y_train = train_df['Survived']

#number of cross validation folds
cv_value = 4

model_results = {}

X_train.head()
y_train.head()
X_train = X_train.as_matrix()
y_train = y_train.as_matrix()
test_df = test_df.as_matrix()
clf_dummy = DummyClassifier(strategy='most_frequent')

scores = cross_val_score(clf_dummy, X=X_train, y=y_train, cv=cv_value)

model_results['DummyClassifier'] = np.mean(scores)

np.mean(scores)
clf_m_naive_bayes = MultinomialNB()
scores = cross_val_score(clf_m_naive_bayes, X=X_train, y=y_train, cv=cv_value)

model_results['MultinomialNB'] = np.mean(scores)

np.mean(scores)
clf_svc = SVC()
scores = cross_val_score(clf_svc, X=X_train, y=y_train, cv=cv_value)

model_results['SVC'] = np.mean(scores)

np.mean(scores)
# Estimators = number of trees
clf_forest = RandomForestClassifier(n_estimators=200, max_depth=4, n_jobs=-1)
scores = cross_val_score(clf_forest, X=X_train, y=y_train, cv=cv_value)

model_results['RandomForest'] = np.mean(scores)

np.mean(scores)
clf_ada = AdaBoostClassifier()
scores = cross_val_score(clf_ada, X=X_train, y=y_train, cv=cv_value)

model_results['AdaBoost'] = np.mean(scores)

np.mean(scores)
clf_knn = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(clf_knn, X=X_train, y=y_train, cv=cv_value)

model_results['KNN'] = np.mean(scores)

np.mean(scores)
clf_nn = MLPClassifier(hidden_layer_sizes=(5,3), max_iter=300, learning_rate_init=0.01, random_state=101)
scores = cross_val_score(clf_nn, X=X_train, y=y_train, cv=cv_value)

model_results['Neural Network'] = np.mean(scores)

np.mean(scores)
clf_xgb = XGBClassifier(learning_rate=0.01)

scores = cross_val_score(clf_xgb, X=X_train, y=y_train, cv=cv_value)

model_results['XGBoost'] = np.mean(scores)

np.mean(scores)
results = pd.DataFrame(list(model_results.items()), columns=['Model','Score'])
results.sort_values(by='Score', ascending=False)
#There are algorithms that are really close to each other, like AdaBoost, MultinomialNB, NN and KNN. Let's pick these, thinking about
# diferent types algorithms trying to classify:
models_list = [clf_xgb, clf_forest, clf_nn, clf_svc, clf_knn]
for model in models_list:
    model.fit(X_train, y_train)
original_test_df = pd.read_csv('../input/test.csv')
passengers_id = original_test_df['PassengerId']
len(passengers_id)
from collections import Counter

y_predict = []

y = []

#for each model in the list, we are going to predict ALL the test_df, and the append it to 'y'
for model in models_list:
    y.append( model.predict(test_df[:]))

y = np.array(y)

# 5 columns for each algorithm prediction, 418 lines for each passenger prediction
print(y.shape)
# y[:,0] contains the predictions for the first passenger, by the algorithms from models_list.
# As we can see, all of them predicted as '0' (Not Survived)
# Counter(y[:,0]).most_common(3) shows the most commom value, and it's count [(value, count)]: 

print(y[:,0], Counter(y[:,0]).most_common(1))
# [(0, 5)] = We can say that the value '0' it's the most common, with 5 appearences in total.

print(y[:,1], Counter(y[:,1]).most_common(1))
# [(1, 3)] = We can say that the value '1' it's the most common, with 3 appearences in total.
# for each prediction (5 different predictions for each passenger),
# we are counting which one (survived/not survived) has most votes.
for i in range(y.shape[1]):
    y_predict.append([passengers_id[i], Counter(y[:,i]).most_common(3)[0][0]]) 
y_predict[:11]
submission = pd.DataFrame(data=y_predict, columns=['PassengerId', 'Survived'])
submission.head()
submission.to_csv('submission.csv', header=True, index=False)