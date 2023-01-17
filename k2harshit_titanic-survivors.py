import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt
%matplotlib inline
from matplotlib import style
style.use('ggplot')

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head()
train.info()
print('-'*40)
test.info()
# there are many NULL entries
print('-'*40)
# no of NULL entries
# train.isnull().sum()  # train.apply(lambda x: sum(x.isnull()), axis=0)
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing = pd.concat([total, percent], axis=1, keys=['Missing', 'Percent'])
missing.head(20)
# Add test and train into one dataframe:
titanic = train.append(test, ignore_index=True)
titanic.tail()
titanic.shape
# Names are unique across the dataset (count=unique=891)
train.describe(include=['O'])
# drop unnecessary columns
train = train.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
test = test.drop(['Name', 'Ticket'], axis=1)
# It has a lot of NaN values, so it won't cause a remarkable impact on prediction
train.drop("Cabin",axis=1,inplace=True)
test.drop("Cabin",axis=1,inplace=True)
# NA values in Embarked
train['Embarked'].isnull().sum()
# fill the two missing values with the most occurred value,
# Check the count in the next graph
# which is "S".
train['Embarked'] = train['Embarked'].fillna('S')
# Check NA value
train['Embarked'].isnull().any()
f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 4))
ax1.set_title('Count')
train['Embarked'].value_counts().plot(kind='bar', ax=ax1)
ax2.set_title('mean(Survived)')
train[['Embarked', 'Survived']].groupby(['Embarked']).mean().plot(kind='bar', ax=ax2)
train.drop(['Embarked'], axis=1, inplace=True)
test.drop(['Embarked'], axis=1, inplace=True)
# fill missing values of fair
test['Fare'].fillna(test['Fare'].median(), inplace=True)
# convert from float to int
train['Fare'] = train['Fare'].astype(int)
test['Fare'] = test['Fare'].astype(int)
# there are many values so histogram not bar chart
train.Fare.hist(bins=5)

# get fare for survived & didn't survive passengers 
fare_not_survived = train['Fare'][train['Survived']==0]
fare_survived = train['Fare'][train['Survived']==1]
# get average and std for fare of survived/not survived passengers
avg_fare = pd.DataFrame([fare_not_survived.mean(), fare_survived.mean()])
std_fare = pd.DataFrame([fare_not_survived.std(), fare_survived.std()])
# avg_fare
std_fare

# plot
train['Fare'].plot(kind='hist', figsize=(15, 3), bins=100, xlim=(0, 50))
avg_fare.index.names = std_fare.index.names = ['Survived']
# error bar plot
avg_fare.plot(yerr=std_fare, kind='bar', legend=False)
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(15,4))
ax1.set_title('Original Age values - Titanic')
train['Age'].dropna().astype(int).hist(bins=70, ax=ax1)
train['Age'] = train['Age'].fillna(train.Age.mean())
test['Age'] = test['Age'].fillna(test.Age.mean())
ax2.set_title('New Age values - Titanic')
train['Age'].hist(bins=70, ax=ax2)
# convert from float to int
train['Age'] = train['Age'].astype(int)
test['Age'] = test['Age'].astype(int)
# list all variables created so far
%who
f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 4))
ax1.set_title('Count')
train.Sex.value_counts().plot(kind='bar', ax=ax1)
ax2.set_title('mean(Survived)')
train[['Sex', 'Survived']].groupby(['Sex']).mean().plot(kind='bar', ax=ax2)
f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 4))
ax1.set_title('Number Of Passengers By Pclass')
ax1.set_ylabel('Count')
train['Pclass'].value_counts().plot(kind='bar', ax=ax1)
ax2.set_title('mean(Survived)')
train[['Pclass', 'Survived']].groupby(['Pclass']).mean().plot(kind='bar', ax=ax2)
train.columns
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

labelEnc=LabelEncoder()

cat_vars=['Sex']
for col in cat_vars:
    train[col]=labelEnc.fit_transform(train[col])
    test[col]=labelEnc.fit_transform(test[col])
X_train = train.drop('Survived', axis=1)
y_train = train['Survived']
X_test = test.drop('PassengerId', axis=1).copy()
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
train.head()
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
# y_pred = logreg.predict(X_test)
logreg.score(X_train, y_train)
# svm
svc = SVC()
svc.fit(X_train, y_train)
# y_pred = svc.predict(X_test)
svc.score(X_train, y_train)
# Naive bayes
gauss = GaussianNB()
gauss.fit(X_train, y_train)
# y_pred = gauss.predict(X_test)
gauss.score(X_train, y_train)
# kNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
# y_pred = knn.predict(X_test)
knn.score(X_train, y_train)
# random forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)
random_forest.score(X_train, y_train)
# submitting random forest
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': y_pred
})
submission.to_csv('titanic.csv', index=False)