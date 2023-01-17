import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns

sns.set()



import warnings

warnings.filterwarnings("ignore") 
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
train.head()
train.shape
train.describe()
# describe(include = ['O'])* will show the descriptive statistics of object data types.

train.describe(include=['O'])
# We use info() method to see more information of our train dataset.

train.info()
# checking if any column has some missing values

train.isnull().sum()
test.shape
test.head()
test.info()
test.isnull().sum()
survived = train[train['Survived'] == 1]

not_survived = train[train['Survived'] == 0]



print ("Survived: %i (%.1f%%)"%(len(survived), float(len(survived))/len(train)*100.0))

print ("Not Survived: %i (%.1f%%)"%(len(not_survived), float(len(not_survived))/len(train)*100.0))

print ("Total: %i"%len(train))
train.Pclass.value_counts()
pclass_survived = train.groupby('Pclass').Survived.value_counts()

pclass_survived
# plotting the pclass vs survived

pclass_survived.unstack(level=0).plot(kind='bar', subplots=False)
pclass_survived_average = train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()

pclass_survived_average
pclass_survived_average.plot(kind='bar', subplots=False)
# The above statement can be clearly understood from the plot below.

sns.barplot(x='Pclass', y='Survived', data=train)
train.Sex.value_counts()
sex_survival = train.groupby('Sex').Survived.value_counts()

sex_survival
sex_survival.unstack(level=0).plot(kind='bar', subplots=False)
sex_survived_average = train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()

sex_survived_average
sex_survived_average.plot(kind='bar', subplots=False)
sns.barplot(x='Sex', y='Survived', data=train)
tab = pd.crosstab(train['Pclass'], train['Sex'])

print (tab)



# sum(1) means the sum across axis 1.

tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=False)

plt.xlabel('Pclass')

plt.ylabel('Percentage')
sns.factorplot('Sex', 'Survived', hue='Pclass', size=4, aspect=2, data=train)
sns.factorplot(x='Pclass', y='Survived', hue='Sex', col='Embarked', data=train)
train.Embarked.value_counts()
train.groupby('Embarked').Survived.value_counts()
train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()
#train.groupby('Embarked').Survived.mean().plot(kind='bar')

sns.barplot(x='Embarked', y='Survived', data=train)
train.Parch.value_counts()
train.groupby('Parch').Survived.value_counts()
train[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean()
#train.groupby('Parch').Survived.mean().plot(kind='bar')

sns.barplot(x='Parch', y='Survived', ci=None, data=train) # ci=None will hide the error bar
train.SibSp.value_counts()
train.groupby('SibSp').Survived.value_counts()
train[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean()
#train.groupby('SibSp').Survived.mean().plot(kind='bar')

sns.barplot(x='SibSp', y='Survived', ci=None, data=train) 

# ci=None will hide the error bar
fig = plt.figure(figsize=(15,5))

ax1 = fig.add_subplot(131)

ax2 = fig.add_subplot(132)

ax3 = fig.add_subplot(133)



sns.violinplot(x="Embarked", y="Age", hue="Survived", data=train, split=True, ax=ax1)

sns.violinplot(x="Pclass", y="Age", hue="Survived", data=train, split=True, ax=ax2)

sns.violinplot(x="Sex", y="Age", hue="Survived", data=train, split=True, ax=ax3)
total_survived = train[train['Survived']==1]

total_not_survived = train[train['Survived']==0]



male_survived = train[(train['Survived']==1) & (train['Sex']=="male")]

female_survived = train[(train['Survived']==1) & (train['Sex']=="female")]



male_not_survived = train[(train['Survived']==0) & (train['Sex']=="male")]

female_not_survived = train[(train['Survived']==0) & (train['Sex']=="female")]
plt.figure(figsize=[15,5])

plt.subplot(111)

sns.distplot(total_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=True, color='blue')

sns.distplot(total_not_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=True, color='red', axlabel='Age')
plt.figure(figsize=[15,5])



plt.subplot(121)

sns.distplot(female_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=True, color='blue')

sns.distplot(female_not_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=True, color='red', axlabel='Female Age')



plt.subplot(122)

sns.distplot(male_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=True, color='blue')

sns.distplot(male_not_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=True, color='red', axlabel='Male Age')
plt.figure(figsize=(15,6))

sns.heatmap(train.drop('PassengerId',axis=1).corr(), vmax=0.6, square=True, annot=True)
 # combining train and test dataset

train_test_data = [train, test]



# extracting titles from Name column.

for dataset in train_test_data:

    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.')
train.head()
pd.crosstab(train['Title'], train['Sex'])
for dataset in train_test_data:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', \

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    

train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
sns.barplot(x='Title', y='Survived', ci=None, data=train)
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Other": 5}

for dataset in train_test_data:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)
train.head()
for dataset in train_test_data:

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
train.head()
train.Embarked.value_counts()
for dataset in train_test_data:

    dataset['Embarked'] = dataset['Embarked'].fillna('S')
train.head()
for dataset in train_test_data:

    #print(dataset.Embarked.unique())

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
train.head()
for dataset in train_test_data:

    age_avg = dataset['Age'].mean()

    age_std = dataset['Age'].std()

    age_null_count = dataset['Age'].isnull().sum()

    

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list

    dataset['Age'] = dataset['Age'].astype(int)

    

train['AgeBand'] = pd.cut(train['Age'], 5)



print (train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean())
train.head()
for dataset in train_test_data:

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4
train.head()
for dataset in train_test_data:

    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
train['FareBand'] = pd.qcut(train['Fare'], 4)

print (train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean())
train.head()
for dataset in train_test_data:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)
train.head()
for dataset in train_test_data:

    dataset['FamilySize'] = dataset['SibSp'] +  dataset['Parch'] + 1



print (train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())
sns.barplot(x='FamilySize', y='Survived', ci=None, data=train)
for dataset in train_test_data:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

    

print (train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())
train.head()
test.head()
features_drop = ['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'FamilySize']

train = train.drop(features_drop, axis=1)

test = test.drop(features_drop, axis=1)

train = train.drop(['PassengerId', 'AgeBand', 'FareBand'], axis=1)
train.head()
test.head()
X_train = train.drop('Survived', axis=1)

y_train = train['Survived']

X_test = test.drop("PassengerId", axis=1).copy()



X_train.shape, y_train.shape, X_test.shape
# Importing Classifier Modules

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier
clf = LogisticRegression()

clf.fit(X_train, y_train)

y_pred_log_reg = clf.predict(X_test)

acc_log_reg = round( clf.score(X_train, y_train) * 100, 2)

print ("Train Accuracy: " + str(acc_log_reg) + '%')
clf = SVC()

clf.fit(X_train, y_train)

y_pred_svc = clf.predict(X_test)

acc_svc = round(clf.score(X_train, y_train) * 100, 2)

print ("Train Accuracy: " + str(acc_svc) + '%')
clf = LinearSVC()

clf.fit(X_train, y_train)

y_pred_linear_svc = clf.predict(X_test)

acc_linear_svc = round(clf.score(X_train, y_train) * 100, 2)

print ("Train Accuracy: " + str(acc_linear_svc) + '%')
clf = KNeighborsClassifier(n_neighbors = 3)

clf.fit(X_train, y_train)

y_pred_knn = clf.predict(X_test)

acc_knn = round(clf.score(X_train, y_train) * 100, 2)

print ("Train Accuracy: " + str(acc_knn) + '%')
clf = DecisionTreeClassifier()

clf.fit(X_train, y_train)

y_pred_decision_tree = clf.predict(X_test)

acc_decision_tree = round(clf.score(X_train, y_train) * 100, 2)

print ("Train Accuracy: " + str(acc_decision_tree) + '%')
clf = RandomForestClassifier(n_estimators=100)

clf.fit(X_train, y_train)

y_pred_random_forest = clf.predict(X_test)

acc_random_forest = round(clf.score(X_train, y_train) * 100, 2)

print ("Train Accuracy: " + str(acc_random_forest) + '%')
clf = GaussianNB()

clf.fit(X_train, y_train)

y_pred_gnb = clf.predict(X_test)

acc_gnb = round(clf.score(X_train, y_train) * 100, 2)

print ("Train Accuracy: " + str(acc_gnb) + '%')
clf = Perceptron(max_iter=5, tol=None)

clf.fit(X_train, y_train)

y_pred_perceptron = clf.predict(X_test)

acc_perceptron = round(clf.score(X_train, y_train) * 100, 2)

print ("Train Accuracy: " + str(acc_perceptron) + '%')
clf = SGDClassifier(max_iter=5, tol=None)

clf.fit(X_train, y_train)

y_pred_sgd = clf.predict(X_test)

acc_sgd = round(clf.score(X_train, y_train) * 100, 2)

print ("Train Accuracy: " + str(acc_sgd) + '%')
from sklearn.metrics import confusion_matrix

import itertools



clf = RandomForestClassifier(n_estimators=100)

clf.fit(X_train, y_train)

y_pred_random_forest_training_set = clf.predict(X_train)

acc_random_forest = round(clf.score(X_train, y_train) * 100, 2)

print ("Accuracy: %i %% \n"%acc_random_forest)



class_names = ['Survived', 'Not Survived']



# Compute confusion matrix

cnf_matrix = confusion_matrix(y_train, y_pred_random_forest_training_set)

np.set_printoptions(precision=2)



print ('Confusion Matrix in Numbers')

print (cnf_matrix)

print ('')



cnf_matrix_percent = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]



print ('Confusion Matrix in Percentage')

print (cnf_matrix_percent)

print ('')



true_class_names = ['True Survived', 'True Not Survived']

predicted_class_names = ['Predicted Survived', 'Predicted Not Survived']



df_cnf_matrix = pd.DataFrame(cnf_matrix, 

                             index = true_class_names,

                             columns = predicted_class_names)



df_cnf_matrix_percent = pd.DataFrame(cnf_matrix_percent, 

                                     index = true_class_names,

                                     columns = predicted_class_names)



plt.figure(figsize = (15,5))



plt.subplot(121)

sns.heatmap(df_cnf_matrix, annot=True, fmt='d', cmap = "Blues")



plt.subplot(122)

sns.heatmap(df_cnf_matrix_percent, annot=True, cmap = "Blues")
models = pd.DataFrame({

    'Model': ['LR', 'SVM', 'L-SVC', 

              'KNN', 'DTree', 'RF', 'NB', 

              'Perceptron', 'SGD'],

    

    'Score': [acc_log_reg, acc_svc, acc_linear_svc, 

              acc_knn,  acc_decision_tree, acc_random_forest, acc_gnb, 

              acc_perceptron, acc_sgd]

    })



models = models.sort_values(by='Score', ascending=False)

models
sns.barplot(x='Model', y='Score', ci=None, data=models)
test.head()
submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": y_pred_random_forest

    })



submission.to_csv('gender_submission.csv', index=False)