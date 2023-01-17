import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random as rnd
train_ds=pd.read_csv('../input/titanic/train.csv')
test_ds=pd.read_csv('../input/titanic/test.csv')
combine=[train_ds,test_ds]
print(train_ds.columns.values)
train_ds.head()
train_ds.tail()
train_ds.info()
print('*'*50)
test_ds.info()
train_ds.describe()
train_ds.describe(include='object')
plt.figure(figsize=(15,8))
sns.kdeplot(train_ds["Age"][train_ds.Survived == 1], color="springgreen", shade=True)
sns.kdeplot(train_ds["Age"][train_ds.Survived == 0], color="salmon", shade=True)
plt.legend(['Survived', 'Died'])
plt.title('Density Plot of Age for Surviving Population')
plt.show()
k = sns.FacetGrid(train_ds, col='Survived')
k.map(plt.hist, 'Age', bins=20)
grid = sns.FacetGrid(train_ds, col='Survived', row='Pclass', height=2.0, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();
grid = sns.FacetGrid(train_ds, row='Embarked', height=2.0, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()
grid = sns.FacetGrid(train_ds, row='Embarked', col='Survived', height=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()
train_ds = train_ds.drop(['PassengerId','Ticket','Cabin'], axis=1)
test_ds = test_ds.drop(['Ticket', 'Cabin'], axis=1)

combine=[train_ds,test_ds]
for dataset in combine:
    dataset['Titles'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
train_ds = train_ds.drop(['Name'], axis=1)
test_ds = test_ds.drop(['Name'], axis=1)
combine = [train_ds, test_ds]
for dataset in combine:
    dataset['Titles'] = dataset['Titles'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')

    dataset['Titles'] = dataset['Titles'].replace(['Mlle','Ms','Mme'], ['Miss','Miss','Mrs'])
   
train_ds['Titles']=train_ds['Titles'].map({ "Master": 1,"Miss": 2,"Mr": 3, "Mrs": 4, "Other": 5})
test_ds['Titles']=test_ds['Titles'].map({ "Master": 1,"Miss": 2,"Mr": 3, "Mrs": 4, "Other": 5})
sns.barplot('Titles', 'Survived', data=train_ds, color="#2ecc71")
plt.show()
train_ds['Sex']=train_ds['Sex'].map({'female': 1, 'male': 0}).astype(int)
test_ds['Sex']=test_ds['Sex'].map({'female': 1, 'male': 0}).astype(int)
train_ds.head()
guess_ages = np.zeros((2,3))
guess_ages
for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                                  (dataset['Pclass'] == j+1)]['Age'].dropna()

            guess_ages[i,j] = int(guess_df.mean())
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

train_ds.head()

Embark_common=train_ds.Embarked.dropna().mode()[0]

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(Embark_common)
    
train_ds.info()
test_ds['Fare'].fillna(test_ds['Fare'].dropna().median(), inplace=True)
test_ds.info()
train_ds['Embarked']=train_ds['Embarked'].map({'S': 1, 'C': 2, 'Q':3}).astype(int)
test_ds['Embarked']=test_ds['Embarked'].map({'S': 1, 'C': 2, 'Q':3}).astype(int)
train_ds.head()
train_ds['AgeRange'] = pd.cut(train_ds['Age'], 5)
train_ds[['AgeRange', 'Survived']].groupby(['AgeRange'], as_index=False).mean().sort_values(by='AgeRange', ascending=True)
for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 1
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 4
    dataset.loc[ dataset['Age'] > 64, 'Age']=5
train_ds = train_ds.drop(['AgeRange'], axis=1)
combine = [train_ds, test_ds]
train_ds.head()
train_ds['FareRange'] = pd.qcut(train_ds['Fare'], 5)
train_ds[['FareRange', 'Survived']].groupby(['FareRange'], as_index=False).mean().sort_values(by='FareRange', ascending=True)
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.85, 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 7.85) & (dataset['Fare'] <= 10.5), 'Fare'] = 2
    dataset.loc[(dataset['Fare'] > 10.5) & (dataset['Fare'] <= 21.67), 'Fare'] = 3
    dataset.loc[(dataset['Fare'] > 21.67) & (dataset['Fare'] <= 39.68), 'Fare'] = 4
    dataset.loc[ dataset['Fare'] > 39.68, 'Fare'] = 5
    dataset['Fare'] = dataset['Fare'].astype(int)

train_ds = train_ds.drop(['FareRange'], axis=1)
combine = [train_ds, test_ds]
train_ds.head(10)
for dataset in combine:
    dataset['family_count']=dataset['SibSp']+dataset['Parch']
    dataset.loc[dataset['family_count']>0,'isAlone']=0
    dataset.loc[dataset['family_count']==0,'isAlone']=1
    dataset['isAlone']=dataset['isAlone'].astype(int)
train_ds.head()
sns.barplot('isAlone', 'Survived', data=train_ds, color="coral")
plt.show()
train_ds=train_ds.drop(['SibSp','Parch','family_count'],axis=1)
test_ds=test_ds.drop(['SibSp','Parch','family_count'],axis=1)
train_ds.head()
X_train = train_ds.drop("Survived", axis=1)
Y_train = train_ds["Survived"]
X_test  = test_ds.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape
#Fitting Logistic regression to the training set
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,Y_train)
 
#Predict the test set result
Y_pred=classifier.predict(X_test)

Confidence_score = round(classifier.score(X_train, Y_train) * 100, 2)
Confidence_score
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, Y_train)

# Predicting the Test set results
Y_pred = classifier.predict(X_test)

Confidence_score = round(classifier.score(X_train, Y_train) * 100, 2)
Confidence_score
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier.fit(X_train,Y_train)

Y_pred=classifier.predict(X_test)

Confidence_score = round(classifier.score(X_train, Y_train) * 100, 2)
Confidence_score
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, Y_train)

# Predicting the Test set results
Y_pred = classifier.predict(X_test)

Confidence_score = round(classifier.score(X_train, Y_train) * 100, 2)
Confidence_score
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y_train)

# Predicting the Test set results
Y_pred = classifier.predict(X_test)

Confidence_score = round(classifier.score(X_train, Y_train) * 100, 2)
Confidence_score
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 30, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y_train)

# Predicting the Test set results
Y_pred = classifier.predict(X_test)

Confidence_score = round(classifier.score(X_train, Y_train) * 100, 2)
Confidence_score
Result = pd.DataFrame({
        "PassengerId": test_ds["PassengerId"],
        "Survived": Y_pred
    })
Result.to_csv('Submission.csv', index=False)