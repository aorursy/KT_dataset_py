# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data['Source'] = 'Train'

train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data['Source'] = 'Test'

test_data.head()
combine = pd.concat([train_data, test_data], ignore_index = True, sort=False)

combine.head() # This is to print top 5 records.
combine.tail() # This is to print last 5 records
# As mentioned in Competation, that the Submission file should have 2 columns such as "PassengerId" and "Survived". So we can have PassengerId as our Index in all the 3 data frames, and the Survived for submission file will be predicted later.

train_data.set_index(['PassengerId'], inplace = True)

test_data.set_index(['PassengerId'], inplace = True)

combine.set_index(['PassengerId'], inplace = True)
train_data.head()
# Checking the Columns.

print(train_data.columns.values)

print('-'*50)

print(train_data.columns.values)

print('-'*50)

print(combine.columns.values)
# Checking the data-type of DF's.

train_data.info()

print('_'*40)

test_data.info()
# Alternate way of checking the size of data frames.

print(train_data.shape)

print('_'*20)

print(test_data.shape) # Here we do not have column with name Survived.

print('_'*20)

print(combine.shape)
# Lets see the status of our data.

train_data.describe()
train_data.describe(include=['O'])
# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
# Lets see the Correlation of each variable to Target Variable using HeatMap.

pearson_corr = train_data.corr(method ='pearson') 



kendall_corr = train_data.corr(method ='kendall') 
#plt.subplot(221)

sns.heatmap(pearson_corr, annot=True) # heatmap(pearson_corr, cmap="YlGnBu") # Just to have different colors.



#plt.subplot(222)

#sns.heatmap(kendall_corr, cmap = 'cubehelix') # heatmap(kendall_corr, cmap="YlGnBu") # Just to have different colors.



plt.show()
sns.heatmap(kendall_corr, annot=True, cmap = 'cubehelix') # heatmap(kendall_corr, cmap="YlGnBu") # Just to have different colors.

plt.show()
train_data.isnull().sum()
test_data.isnull().sum()
# Another way of identifying Null or Missing Values

null_value_stats = train_data.isnull().sum(axis=0)

null_value_stats[null_value_stats != 0]
# Another way of identifying Null or Missing Values

total = train_data.isnull().sum().sort_values(ascending=False)

percent_1 = train_data.isnull().sum()/train_data.isnull().count()*100

percent_2 = (round(percent_1, 1)).sort_values(ascending=False)

missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])

missing_data.head(5)
train_data['Age'].describe()
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].mean())

test_data['Age'] = test_data['Age'].fillna(test_data['Age'].mean())

combine['Age'] = combine['Age'].fillna(combine['Age'].mean())
train_data['Age'].describe()
train_data['Embarked'].describe()
print(train_data['Embarked'].mode())

print(combine['Embarked'].mode())
# Seems "S" is the most common value for Embarked. Thus replacing the NaN with the same ie Mode.

train_data['Embarked'] = train_data['Embarked'].fillna('S')

combine['Embarked'] = combine['Embarked'].fillna('S')
train_data['Embarked'].describe()
train_data['Embarked'].isnull().sum(axis=0)
sns.countplot(x='Survived', hue='Sex', data=train_data)
# From here onwards, we will be using the combine data frame.

sns.barplot(x="Embarked", y="Survived", hue="Sex", data=combine);
g = sns.FacetGrid(train_data, col='Survived')

g.map(plt.hist, 'Age', bins=20)
grid = sns.FacetGrid(train_data, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
sns.barplot(x='Pclass', y='Survived', data=train_data)
train_data.info()
train_data['Sex']
#genders = {"male": 0, "female": 1}

#for dataset in combine:

#    dataset['Sex']= dataset['Sex'].map( genders ).astype(int)

#

#combine.head()



train_data['Sex'] = train_data.Sex.apply(lambda x:0 if x == 'female' else 1)

test_data['Sex'] = test_data.Sex.apply(lambda x:0 if x == 'female' else 1)
train_data.head()
train_data.groupby('Embarked').size()
train_data['Embarked'] = train_data.Embarked.apply(lambda x:0 if x == 'S' else (1 if x == 'C' else 2 ))

test_data['Embarked'] = test_data.Embarked.apply(lambda x:0 if x == 'S' else (1 if x == 'C' else 2 ))
train_data.head()
train_data.groupby('Embarked').size()
# Group by Survived.

train_data.groupby('Survived').mean()
# Group by Sex.

train_data.groupby('Sex').mean()
# Defining Feature and Target.

features = ["Pclass", "Sex", "SibSp", "Parch", "Age", "Embarked"]

target = train_data["Survived"]
train_data[features]
# split the train_data into 2 DF's aka X_train, X_test, y_train, y_test.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_data[features], target, test_size=0.2)



print (X_train.shape, y_train.shape)

print (X_test.shape, y_test.shape)
# test_data 

X_test_df  = test_data[features].copy()
# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
# Logistic Regression



logreg = LogisticRegression()

logreg.fit(X_train, y_train)

Y_pred_lr = logreg.predict(X_test)

#print(Y_pred_lr)

acc_log = round(logreg.score(X_train, y_train) * 100, 2)

print("Accuracy (LogisticRegression)", acc_log)
# Predicting on test_data

Y_pred_test_df = logreg.predict(X_test_df)

Y_pred_test_df 
# Support Vector Machines



svc = SVC()

svc.fit(X_train, y_train)

Y_pred_svc = svc.predict(X_test)

acc_svc = round(svc.score(X_train, y_train) * 100, 2)

print("Accuracy (Support Vector Machines)", acc_svc)
# KNN

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, y_train)

Y_pred_knn = knn.predict(X_test)

acc_knn = round(knn.score(X_train, y_train) * 100, 2)

print("Accuracy (KNN)", acc_knn)
# Gaussian Naive Bayes



gaussian = GaussianNB()

gaussian.fit(X_train, y_train)

Y_pred_gnb = gaussian.predict(X_test)

acc_gaussian = round(gaussian.score(X_train, y_train) * 100, 2)

print("Accuracy (Gaussian Naive Bayes)", acc_gaussian)
# Perceptron



perceptron = Perceptron()

perceptron.fit(X_train, y_train)

Y_pred_per = perceptron.predict(X_test)

acc_perceptron = round(perceptron.score(X_train, y_train) * 100, 2)

print("Accuracy (Perceptron)", acc_perceptron)
# Linear SVC



linear_svc = LinearSVC()

linear_svc.fit(X_train, y_train)

Y_pred_lsvc = linear_svc.predict(X_test)

acc_linear_svc = round(linear_svc.score(X_train, y_train) * 100, 2)

print("Accuracy (Linear SVC)", acc_linear_svc)
# Stochastic Gradient Descent



sgd = SGDClassifier()

sgd.fit(X_train, y_train)

Y_pred_sgc = sgd.predict(X_test)

acc_sgd = round(sgd.score(X_train, y_train) * 100, 2)

print("Accuracy (Stochastic Gradient Descent)", acc_sgd)
# Decision Tree



decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, y_train)

Y_pred_dt = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)

print("Accuracy (Decision Tree)", acc_decision_tree)
# Random Forest



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, y_train)

Y_pred_rf = random_forest.predict(X_test)

acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)

print("Accuracy (random Forest)", acc_random_forest)
modelling_score = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 

              'Stochastic Gradient Decent', 'Linear SVC', 

              'Decision Tree'],

    'Score': [acc_svc, acc_knn, acc_log, 

              acc_random_forest, acc_gaussian, acc_perceptron, 

              acc_sgd, acc_linear_svc, acc_decision_tree]})

modelling_score.sort_values(by='Score', ascending=False)
X_test_df.shape
# Predicting on actual test_data

Y_pred_test_df = random_forest.predict(X_test_df)

Y_pred_test_df 
X_test_df.head()
PassengerId = X_test_df.index
PassengerId.shape
Y_pred_test_df.shape
submission = pd.DataFrame( { 'PassengerId': PassengerId , 'Survived': Y_pred_test_df } )
print("Submission File Shape ",submission.shape)

submission.head()

submission.to_csv( '/kaggle/working/titanic_prediction_submission.csv' , index = False )
#submission = pd.DataFrame({

#        "PassengerId": X_test_df.index,

#        "Survived": Y_pred_test_df

#    })

# submission.to_csv('../output/titanic_prediction_submission.csv', index=False)