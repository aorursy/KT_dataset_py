import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import re
import warnings
from statistics import mode
warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')
from copy import deepcopy
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
train.head()
plt.style.use('seaborn')
plt.figure(figsize=(10,5))
sns.heatmap(train.isnull(), yticklabels = False, cmap='plasma')
plt.title('Null Values in Training Set');
train.drop(columns=['Cabin', 'Name', 'Ticket'], inplace=True)
test.drop(columns=['Cabin', 'Name', 'Ticket'], inplace=True)
train.head()
test.head()
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.countplot(train.Survived)
plt.title('Number of passengers Survived');

plt.subplot(1,2,2)
sns.countplot(x='Survived', hue='Pclass', data=train)
plt.title('Number of passengers Survived');
pclass1 = train[train.Pclass == 1]['Survived'].value_counts(normalize=True).values[0]*100
pclass2 = train[train.Pclass == 2]['Survived'].value_counts(normalize=True).values[1]*100
pclass3 = train[train.Pclass == 3]['Survived'].value_counts(normalize=True).values[1]*100


print("Pclass-1: {:.1f}% People Survived".format(pclass1))
print("Pclass-2: {:.1f}% People Survived".format(pclass2))
print("Pclass-3: {:.1f}% People Survived".format(pclass3))
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.countplot(train.Survived)
plt.title('Number of passengers Survived');

plt.subplot(1,2,2)
sns.countplot(x='Survived', hue='Sex', data=train)
plt.title('Number of passengers Survived');
train['Age'].hist(bins=40)
plt.title('Age Distribution');
# set plot size
plt.figure(figsize=(15, 3))

# plot a univariate distribution of Age observations 
sns.distplot(train[(train["Age"] > 0)].Age, kde_kws={"lw": 3}, bins = 50)

# set titles and labels
plt.title('Distrubution of passengers age',fontsize= 14)
plt.xlabel('Age')
plt.ylabel('Frequency')
# clean layout
plt.tight_layout()
plt.figure(figsize=(15,5))

#Draw a box plot to show Age distributions with respect to survival status
sns.boxplot(y='Survived', x='Age', data=train, palette=["#3f3e6fd1", "#85c6a9"], fliersize = 0, orient = 'h')

#Add a scatterplot for each category
sns.stripplot(y='Survived', x='Age', data=train, palette=["#3f3e6fd1", "#85c6a9"], linewidth = 0.6, orient = 'h')

plt.yticks(np.arange(2), ['Drowned', 'Survived'])
plt.title('Age distribution grouped by surviving status')
plt.ylabel('Surviving status')
plt.tight_layout()
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.countplot(train['SibSp'])
plt.title('Number of siblings/spouses aboard');

plt.subplot(1,2,2)
sns.countplot(x='Survived', hue='SibSp', data=train)
plt.legend(loc='right')
plt.title('Number of siblings/spouses aboard');
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.countplot(train['Parch'])
plt.title('Number of parents/children aboard');

plt.subplot(1,2,2)
sns.countplot(x='Survived', hue='Parch', data=train)
plt.legend(loc='right')
plt.title('Number of parents/children aboard');
# set plot size
plt.figure(figsize=(15, 3))

# plot a univariate distribution of Age observations 
sns.distplot(train[(train["Fare"] > 0)].Fare, kde_kws={"lw": 3}, bins = 50)

# set titles and labels
plt.title('Distrubution of fare',fontsize= 14)
plt.xlabel('Fare')
plt.ylabel('Frequency')
# clean layout
plt.tight_layout()
plt.figure(figsize=(15,5))

#Draw a box plot to show Age distributions with respect to survival status
sns.boxplot(y='Survived', x='Fare', data=train, palette=["#3f3e6fd1", "#85c6a9"], fliersize = 0, orient = 'h')

#Add a scatterplot for each category
sns.stripplot(y='Survived', x='Fare', data=train, palette=["#3f3e6fd1", "#85c6a9"], linewidth = 0.6, orient = 'h')

plt.yticks(np.arange(2), ['Drowned', 'Survived'])
plt.title('Fare distribution grouped by surviving status')
plt.ylabel('Surviving status')
plt.tight_layout()
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.countplot(train['Embarked'])
plt.title('Name of Port of embarkation')

plt.subplot(1,2,2)
sns.countplot(x='Survived', hue='Embarked', data=train)
plt.legend(loc='right')
plt.title('Name of passenger Survived');
embark1 = train[train.Embarked == 'S']['Survived'].value_counts(normalize=True).values[1]*100
embark2 = train[train.Embarked == 'C']['Survived'].value_counts(normalize=True).values[0]*100
embark3 = train[train.Embarked == 'Q']['Survived'].value_counts(normalize=True).values[1]*100


print("S: {:.1f}% People Survived".format(embark1))
print("C: {:.1f}% People Survived".format(embark2))
print("Q: {:.1f}% People Survived".format(embark3))
sns.heatmap(train.corr(), annot=True);
train.loc[train.Age.isnull(), 'Age'] = train.groupby('Pclass').Age.transform('median')
test.loc[test.Age.isnull(), 'Age'] = test.groupby('Pclass').Age.transform('median')
train.Embarked.value_counts()
train['Embarked'] = train['Embarked'].fillna(mode(train['Embarked']))
test['Embarked'] = test['Embarked'].fillna(mode(test['Embarked']))
train.loc[train.Fare.isnull(), 'Fare'] = train.groupby('Pclass').Fare.transform('median')
test.loc[test.Fare.isnull(), 'Fare'] = test.groupby('Pclass').Fare.transform('median')

train['Sex'][train['Sex']=='male'] = 0
train['Sex'][train['Sex']=='female'] = 1

test['Sex'][test['Sex']=='male'] = 0
test['Sex'][test['Sex']=='female'] = 1
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()
temp = pd.DataFrame(encoder.fit_transform(train[['Embarked']]).toarray(), columns=['S', 'C', 'Q'])
train = train.join(temp)
train.drop(columns='Embarked', inplace=True)

temp = pd.DataFrame(encoder.transform(test[['Embarked']]).toarray(), columns=['S', 'C', 'Q'])
test = test.join(temp)
test.drop(columns='Embarked', inplace=True)
train.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train.drop(['Survived', 'PassengerId'], axis=1), train['Survived'], test_size = 0.2, random_state=2)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)

# We must apply the scaling to the test set that we computed for the training set
X_test_scaled = scaler.transform(X_test)
from sklearn.ensemble import RandomForestClassifier
rfclf = RandomForestClassifier(random_state=2)
# Set our parameter grid
param_grid = { 
    'criterion' : ['gini', 'entropy'],
    'n_estimators': [300, 400],
    'max_features': ['auto', 'log2'],
    'max_depth' : [6, 7]    
}
from sklearn.model_selection import GridSearchCV

randomForest_CV = GridSearchCV(estimator = rfclf, param_grid = param_grid, cv = 5)
randomForest_CV.fit(X_train, y_train)

randomForest_CV.best_params_
rf_clf = RandomForestClassifier(random_state = 2, criterion = 'entropy', max_depth = 7, max_features = 'auto', n_estimators = 400)

rf_clf.fit(X_train, y_train)
predictions = rf_clf.predict(X_test)
from sklearn.metrics import accuracy_score

accuracy_score(y_test, predictions) * 100
scaler = MinMaxScaler()

train_conv = scaler.fit_transform(train.drop(['Survived', 'PassengerId'], axis=1))
test_conv = scaler.transform(test.drop(['PassengerId'], axis = 1))
rf_clf = RandomForestClassifier(random_state = 2, criterion = 'entropy', max_depth = 7, max_features = 'auto', n_estimators = 400)

rf_clf.fit(train_conv, train['Survived'])
test2 = deepcopy(test)

test2['Survived'] = rf_clf.predict(test_conv)
test2[['PassengerId', 'Survived']].to_csv('MySubmissionRandomForest.csv', index = False)