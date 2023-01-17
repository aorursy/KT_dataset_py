# Loading Modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set_style('whitegrid')

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
# get titanic & test csv files as a DataFrame
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

# preview the data
train.head()
print("----------------------------------------")
test.head()
train.info()
print("----------------------------")
test.info()
#Relationship between Features and Survival
survived = train[train['Survived']==1]
not_survived = train[train['Survived']== 0]

train.Pclass.value_counts()
train.groupby('Pclass').Survived.value_counts()
train[['Pclass','Survived']].groupby(['Pclass'], as_index=False).mean()
sns.barplot(x='Pclass', y='Survived', data=train)
train.Sex.value_counts()
train.groupby('Sex').Survived.value_counts()
train[['Sex','Pclass']].groupby(['Sex'], as_index=False).mean()
sns.barplot(x='Sex',y='Survived',data=train)
#Pclass & Sex vs. Survival
tab = pd.crosstab(train['Pclass'], train['Sex'])
print(tab)
tab.div(tab.sum(axis=1).astype(float), axis=0).plot(kind="bar",stacked=True)
tab_corr = pd.crosstab([train['Pclass'],train['Sex']],train['Survived'])
print(tab_corr)
tab_corr.div(tab_corr.sum(axis=1).astype(float), axis=0).plot(kind="bar",stacked=True)
#Pclass, Sex & Embarked vs. Survival
sns.factorplot(x='Pclass', y='Survived', hue='Sex', col='Embarked', kind='point', data=train)
#Embarked vs. Survived
train.groupby('Embarked').Survived.value_counts()
train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()
#Correlating Features -Heatmap of Correlation between different features
plt.figure(figsize=(15,5))
sns.heatmap(train.drop('PassengerId', axis=1).corr(),vmax=0.6, square=True, annot=True)
#combining train and test dataset
train = train.drop(['PassengerId', 'Name'], axis=1)
test = test.drop(['PassengerId', 'Name'],axis=1)
train_test_data = [train, test]
train.shape, test.shape
#Age vs Survived
for dataset in train_test_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)
train['CategoricalAge'] = pd.cut(train['Age'],5)
train.head()
#Fare
# only for the test set, since there is a missing value
test['Fare'] = test['Fare'].fillna(test['Fare'].mean())

#convert from float to int
train['Fare'] = train['Fare'].astype(int)
test['Fare'] = test['Fare'].astype(int)

train['CategoricalFare'] = pd.qcut(train['Fare'],4)

for dataset in train_test_data:
    dataset.loc[dataset['Fare']<= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] >7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] >14.454) & (dataset['Fare'] <= 31.0), 'Fare'] = 2
    dataset.loc[(dataset['Fare'] >31.0) & (dataset['Fare'] <= 512.329), 'Fare'] = 3
for dataset in train_test_data:
    dataset.loc[dataset['Age']<= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] >16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] >32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] >48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] >64, 'Age'] = 4;
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
test.head()
# Family - if having any family member(whether parent, brother, ...etc) will increase chances of Survival or not.
for dataset in train_test_data:
    dataset['Family'] = dataset['Parch']+dataset['SibSp']
    dataset['Family'].loc[dataset['Family']>0]=1
    dataset['Family'].loc[dataset['Family']==0]=0

train[['Family','Survived']].groupby(['Family'], as_index=False).mean()
sns.barplot(x='Family',y='Survived',data=train)
test.head()
X_train = train.drop(['CategoricalAge','CategoricalFare','Ticket','Cabin','Survived'], axis=1)
Y_train = train['Survived']
X_test = test.drop(['Ticket','Cabin',], axis=1)
X_train.shape, Y_train.shape, X_test.shape
X_test.info()

# Logistic Regression measures the relationship between the categorical dependent variable (feature) and one or more independent variables (features) by estimating probabilities using a the cumulative logistic distribution
logreg = LogisticRegression()
logreg.fit(X_train,Y_train)
y_pred = logreg.predict(X_test).copy()
logreg.score(X_train, Y_train)
