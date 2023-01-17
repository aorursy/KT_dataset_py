# pandas, numpy

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas import Series,DataFrame



# matplotlib, seaborn

import matplotlib.pyplot as plt

import seaborn as sns

# to make the plots visible in notebook

%matplotlib inline 



# machine learning modules are imported in later part



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import warnings

warnings.filterwarnings("ignore")
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train_test_data = [train, test]
train.head()
train.describe()
print ('TRAINING DATA\n')

train.info()

print ("----------------------------------------\n")

print ('TESTING DATA\n')

test.info()
print ('#MISSING VALUES IN TRAINING DATA')

train.isnull().sum()
print ('#MISSING VALUES IN TESTING DATA')

test.isnull().sum()
train.describe(include=['O'])
survived = train[train['Survived'] == 1]

not_survived = train[train['Survived'] == 0]
sns.countplot(x='Survived',data=train)
print ("Survived: %i (%.1f%%)" %(len(survived), float(len(survived))/len(train)*100.0))

print ("Not Survived: %i (%.1f%%)" %(len(not_survived), float(len(not_survived))/len(train)*100.0))

print ("Total: %i" % (len(train)))
train.Pclass.value_counts()
train.groupby('Pclass').Survived.value_counts()
sns.factorplot(x="Pclass", hue="Sex", col="Survived",data=train, kind="count",size=5, aspect=1, palette="BuPu");
train.groupby('Pclass').Survived.mean()
sns.factorplot(x="Pclass", y="Survived", data=train,size=5, kind="bar", palette="BuPu", aspect=1.3)
train.Sex.value_counts()
train.groupby('Sex').Survived.value_counts()
train.groupby('Sex').Survived.mean()
sns.factorplot(x='Sex',y='Survived',data=train, size=5, palette='RdBu_r', ci=None, kind='bar', aspect=1.3)
plt.style.use('bmh')

sns.factorplot(x='Pclass', y='Survived', hue='Sex', col='Embarked', data=train)
# Filling NaN values

for dataset in train_test_data:

    avg = dataset['Age'].mean()

    std = dataset['Age'].std()

    null_count = dataset['Age'].isnull().sum()

    random = np.random.randint(avg-std, avg+std, size=null_count)

    dataset['Age'][np.isnan(dataset['Age'])] = random

    dataset['Age'] = dataset['Age'].astype(int)
age_survived = train['Age'][train['Survived'] == 1]

age_not_survived = train['Age'][train['Survived'] == 0]



# Plot

plt.style.use('bmh')

sns.kdeplot(age_survived, shade=True, label = 'Survived')

sns.kdeplot(age_not_survived, shade=True, label = 'Not Survived')
plt.style.use('bmh')

sns.set_color_codes("deep")

fig , (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(17,5))

sns.violinplot(x="Pclass", y="Age", hue="Survived", data=train, palette={0: "b", 1: "r"},split=True, ax=ax1)

sns.violinplot(x="Embarked", y="Age", hue="Survived", data=train,palette={0: "b", 1: "r"}, split=True, ax=ax2)

sns.violinplot(x="Sex", y="Age", hue="Survived", data=train,palette={0: "b", 1: "r"}, split=True, ax=ax3)
train.Embarked.value_counts()
# As there are only 2 missing values, we will fill those by most occuring "S"

train['Embarked'] = train['Embarked'].fillna('S')

train.Embarked.value_counts()
train.groupby('Embarked').Survived.value_counts()
train.groupby('Embarked').Survived.mean()
sns.factorplot(x='Embarked', y='Survived', data=train, size=4, aspect=2.5)
# As there is one missing value in test data, fill it with the median.

test['Fare'] = test['Fare'].fillna(test['Fare'].median())



# Convert the Fare to integer values

train['Fare'] = train['Fare'].astype(int)

test['Fare'] = test['Fare'].astype(int)



# Compute the Fare for Survived and Not Survived

fare_not_survived = train["Fare"][train["Survived"] == 0]

fare_survived = train["Fare"][train["Survived"] == 1]
sns.factorplot(x="Survived", y="Fare", data=train,size=5, kind="bar", ci=None, aspect=1.3)
train["Fare"][train["Survived"] == 1].plot(kind='hist', alpha=0.6, figsize=(15,3),bins=100, xlim=(0,60))

train["Fare"][train["Survived"] == 0].plot(kind='hist', alpha=0.4, figsize=(15,3),bins=100, xlim=(0,60), title='Fare of Survived(Red) and Not Survived(Blue)')
train.Parch.value_counts()
train.groupby('Parch').Survived.value_counts()
train.groupby('Parch').Survived.mean()
sns.barplot(x='Parch',y='Survived', data=train, ci=None, palette="Blues_d")
train.SibSp.value_counts()
train.groupby('SibSp').Survived.value_counts()
train.groupby('SibSp').Survived.mean()
sns.barplot(x='SibSp', y='Survived', data=train, ci=None, palette="Blues_d")
pd.crosstab(train['SibSp'], train['Parch'])
# Correlation of features

# Negative numbers : inverse proportionality

# Positive numbers : direct proportionality

plt.figure(figsize=(15,6))

sns.heatmap(train.drop('PassengerId',axis=1).corr(), square=True, annot=True, center=0)
train.isnull().sum()
test.isnull().sum()
train.dtypes.index
del train['PassengerId']

train.head()
pd.crosstab(train['Pclass'], train['Survived'])
train['Title'] = train['Name'].str.extract(' ([A-Za-z]+)\.')

test['Title'] = test['Name'].str.extract(' ([A-Za-z]+)\.')



# Delete the 'Name' columns from datasets

del train['Name']

del test['Name']
train['Title'].value_counts()
pd.crosstab(train['Title'], train['Pclass'])
pd.crosstab(train['Title'], train['Survived'])
for data in train_test_data:

    data['Title'] = data['Title'].replace(['Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Others')

    data['Title'] = data['Title'].replace(['Mlle', 'Ms'],'Miss')

    data['Title'] = data['Title'].replace(['Mme', 'Lady'],'Mrs')   

    

train.groupby('Title').Survived.mean()
for data in train_test_data:

    data['Title'] = data['Title'].map({ 'Master':0, 'Miss':1, 'Mr':2, 'Mrs':3, "Others":4 }).astype(int)
train.head()
for data in train_test_data:

    del data['Title']
def person(per):

    age,sex = per

    return 'child' if age < 16 else sex



train['Person'] = train[['Age', 'Sex']].apply(person, axis=1)

test['Person'] = test[['Age', 'Sex']].apply(person, axis=1)



# As 'Sex' column is not required.

del train['Sex']

del test['Sex']
train.head()
train['Person'].value_counts()
train.groupby('Person').Survived.mean()
g = sns.PairGrid(train, y_vars="Survived",x_vars="Person",size=3.5, aspect=1.7)

g.map(sns.pointplot, color=sns.xkcd_rgb["plum"])
for data in train_test_data:

    data['Person'] = data['Person'].map({ 'female':0, 'male':1, 'child':3 }).astype(int)

train.head()
test.Embarked.value_counts()
for data in train_test_data:

    data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
train.head()
# Divide 'Age' into groups

a = pd.cut(train['Age'], 5)

print (train.groupby(a).Survived.mean())
# Assign number to Age limits

for data in train_test_data:

    data.loc[ data['Age'] <= 16, 'Age'] = 0

    data.loc[(data['Age'] > 16) & (data['Age'] <= 32), 'Age'] = 1

    data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age'] = 2

    data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age'] = 3

    data.loc[ data['Age'] > 64, 'Age'] = 4  
train.head()
f = pd.qcut(train['Fare'], 4)

print (train.groupby(f).Survived.mean())
for data in train_test_data:

    data.loc[ data['Fare'] <= 7.91, 'Fare'] = 0

    data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.454), 'Fare'] = 1

    data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 31), 'Fare']   = 2

    data.loc[ data['Fare'] > 31, 'Fare'] = 3

    data['Fare'] = data['Fare'].astype(int)
train.head()
for data in train_test_data:

    data['Family'] = data['Parch'] + data['SibSp']

    data['Family'].loc[data['Family'] > 0] = 1

    data['Family'].loc[data['Family'] == 0] = 0



for data in train_test_data:

    del data['Parch']

    del data['SibSp']
train.head()
for data in train_test_data:

    del data['Cabin']

    del data['Ticket']
train.head()
# Importing modules

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB
X = train.drop('Survived', axis=1)

y = train.Survived



# Split into train-test data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)



# Shape of you

print (X_train.shape)

print (y_train.shape)

print (X_test.shape)

print (y_test.shape)
logReg = LogisticRegression()

logReg.fit(X_train, y_train)

y_pred = logReg.predict(X_test)

print ('Score: %.2f%%' % (round(logReg.score(X_test, y_test)*100, 4)))

print ('Accuracy: %.2f' % (accuracy_score(y_test, y_pred)))
naive_clf = GaussianNB()

naive_clf.fit(X_train, y_train)

y_pred = naive_clf.predict(X_test)

print ('Score: %.2f%%' % (round(naive_clf.score(X_test, y_test)*100, 4)))

print ('Accuracy: %.2f' % (accuracy_score(y_test, y_pred)))
dtree_clf = DecisionTreeClassifier()

dtree_clf.fit(X_train, y_train)

y_pred = dtree_clf.predict(X_test)

print ('Score: %.2f%%' % (round(dtree_clf.score(X_test, y_test)*100, 4)))

print ('Accuracy: %.2f' % (accuracy_score(y_test, y_pred)))
rtree_clf = RandomForestClassifier(n_estimators=100)

rtree_clf.fit(X_train, y_train)

y_pred = rtree_clf.predict(X_test)

print ('Score: %.2f%%' % (round(rtree_clf.score(X_test, y_test)*100, 4)))

print ('Accuracy: %.2f' % (accuracy_score(y_test, y_pred)))
svc_clf = SVC()

svc_clf.fit(X_train, y_train)

y_pred = svc_clf.predict(X_test)

print ('Score: %.2f%%' % (round(svc_clf.score(X_test, y_test)*100, 4)))

print ('Accuracy: %.2f' % (accuracy_score(y_test, y_pred)))
linear_clf = LinearSVC()

linear_clf.fit(X_train, y_train)

y_pred = linear_clf.predict(X_test)

print ('Score: %.2f%%' % (round(linear_clf.score(X_test, y_test)*100, 4)))

print ('Accuracy: %.2f' % (accuracy_score(y_test, y_pred)))
k_range = list(range(1, 30))

k_values = []

for k in k_range:

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    k_values.append(acc)

    print (k,acc)



plt.plot(k_range, k_values)

plt.xlabel('K Values')

plt.ylabel('Accuracy')
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print ('Score: %.2f%%' % (round(knn.score(X_test, y_test)*100, 4)))

print ('Accuracy: %.2f' % (accuracy_score(y_test, y_pred)))
e_range = list(range(1, 25))

estimator_values = []

for est in e_range:

    ada = AdaBoostClassifier(n_estimators=est)

    ada.fit(X_train, y_train)

    y_pred = ada.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    estimator_values.append(acc)

    print (est,acc)



plt.plot(e_range, estimator_values)

plt.xlabel('estimator values')

plt.ylabel('Accuracy')
ada = AdaBoostClassifier(n_estimators=7)

ada.fit(X_train, y_train)

y_pred = ada.predict(X_test)

print ('Score: %.2f%%' % (round(ada.score(X_test, y_test)*100, 4)))

print ('Accuracy: %.2f' % (accuracy_score(y_test, y_pred)))
iteration_values = []

for i in range(1,30):

    clf = Perceptron(max_iter=i, tol=None)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    iteration_values.append(acc)

    print (i,acc)



# Plot

plt.plot(range(1,30), iteration_values)

plt.xlabel('max_iter')

plt.ylabel('Accuracy')
per_clf = Perceptron(max_iter=4, tol=None)

per_clf.fit(X_train, y_train)

y_pred = per_clf.predict(X_test)

print ('Score: %.2f%%' % (round(per_clf.score(X_test, y_test)*100, 4)))

print ('Accuracy: %.2f' % (accuracy_score(y_test, y_pred)))
sgd_clf = SGDClassifier(max_iter=8, tol=None)

sgd_clf.fit(X_train, y_train)

y_pred = sgd_clf.predict(X_test)

print ('Score: %.2f%%' % (round(sgd_clf.score(X_test, y_test)*100, 4)))

print ('Accuracy: %.2f' % (accuracy_score(y_test, y_pred)))
e_range = list(range(1, 30))

estimator_values = []

for est in e_range:

    ada = BaggingClassifier(n_estimators=est)

    ada.fit(X_train, y_train)

    y_pred = ada.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    estimator_values.append(acc)

    print (est,acc)



plt.plot(e_range, estimator_values)

plt.xlabel('estimator values')

plt.ylabel('Accuracy')
bag = BaggingClassifier()

bag.fit(X_train, y_train)

y_pred = bag.predict(X_test)

print ('Score: %.2f%%' % (round(bag.score(X_test, y_test)*100, 4)))

print ('Accuracy: %.2f' % (accuracy_score(y_test, y_pred)))
test.head()
submission = pd.DataFrame({

    "PassengerId" : test['PassengerId'],

    "Survived" : rtree_clf.predict(test.drop('PassengerId', axis=1))

})
# submission.to_csv('titanic.csv', index=False)

submission.head()