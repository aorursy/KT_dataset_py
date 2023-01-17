# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2
#Import the necessary data

test_df = pd.read_csv('../input/test.csv')

train_df = pd.read_csv('../input/train.csv')

combine = [train_df, test_df]
#determine the quality of the provided data in order to determine the proceedure for data cleaning.

#missing values

#train_df.info()

print ("percentage of missing values for age: ",(1-((714/891)))*100)

print ("Percentage of missing values for cabin: ",(1-((204/891)))*100)

print ("The average age: ",train_df["Age"].mean())

print ("The mode of age in train dataset: ",train_df["Age"].mode())

print ("The mode of age in test dataset: ",test_df["Age"].mode())
#remove the cabin column

train_df.drop('Cabin', axis = 1, inplace = True)

test_df.drop('Cabin', axis = 1, inplace = True)

#while we are here, remove the name & tickets columns as it is obvious it will add no value

train_df.drop(['Name','Ticket'], axis = 1, inplace = True)

test_df.drop(['Name','Ticket'], axis = 1, inplace = True)

combine = [train_df,test_df]
train_df['Age'].fillna(24, inplace=True)

test_df['Age'].fillna(21, inplace=True)
train_df.head()
#Turn the sex category into a binary variable

gender_mapping = {"female": 0, "male": 1}

for dataset in combine:

    dataset['Sex'] = dataset['Sex'].map(gender_mapping)

    dataset['Sex'] = dataset['Sex'].fillna(0)



train_df.head()
#split the ages up into groups

#train_df['AgeBand'] = pd.cut(train_df['Age'], 5)

#train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)

#for dataset in combine:    

 #   dataset.loc[ dataset['Age'] <= 16.336, 'Age'] = 0

 #   dataset.loc[(dataset['Age'] > 16.336) & (dataset['Age'] <= 32.252), 'Age'] = 1

 #   dataset.loc[(dataset['Age'] > 32.252) & (dataset['Age'] <= 48.168), 'Age'] = 2

 #   dataset.loc[(dataset['Age'] > 48.168) & (dataset['Age'] <= 64.084), 'Age'] = 3

 #   dataset.loc[ dataset['Age'] > 64.084, 'Age']

 #   dataset['Age'] = dataset['Age'].astype(int)

#train_df.head()
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

test_df.head()
#now banding for fare

#train_df['FareBand'] = pd.cut(train_df['Fare'], 4)

#train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
#for dataset in combine:    

#    dataset.loc[ dataset['Fare'] <= 128.082, 'Fare'] = 0

#    dataset.loc[(dataset['Fare'] > 128.082) & (dataset['Fare'] <= 256.165), 'Fare'] = 1

#    dataset.loc[(dataset['Fare'] > 256.165) & (dataset['Fare'] <= 384.247), 'Fare'] = 2

#    dataset.loc[ dataset['Fare'] > 384.247, 'Fare'] = 3

#    dataset['Fare'] = dataset['Fare'].astype(int)

#train_df.head()
#drop the 2 band column

#train_df = train_df.drop(['FareBand','AgeBand'], axis=1)

#combine = [train_df, test_df]
#Need to turn embarked into a number

freq_port = train_df.Embarked.dropna().mode()[0]

freq_port

for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)



train_df.head()
train_df.corr(method = 'pearson')
#Taking the above correlation matrix into account, remove PassengerID, Age, Sibsp, Parch

train_df = train_df.drop(['PassengerId','Age','SibSp','Parch','Fare'], axis=1)

combine = [train_df, test_df]

train_df.corr(method = 'pearson')
#set up the data for the models

X_train = train_df.drop("Survived", axis=1)

Y_train = train_df["Survived"]

X_test  = test_df.drop(['PassengerId','Age','SibSp','Parch','Fare'], axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape

#print (X_train.head())
test = SelectKBest(score_func=chi2, k="all")

fit = test.fit(X_train, Y_train)

# summarize scores

np.set_printoptions(precision=3)

print(fit.scores_)

features = fit.transform(X_train)

# summarize selected features

print(features[0:5,:])
# Logistic Regression

logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

acc_log
# Support Vector Machines

svc = SVC()

svc.fit(X_train, Y_train)

Y_predSVM = svc.predict(X_test)

acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

acc_svc




knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

acc_knn



# Gaussian Naive Bayes



gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

Y_pred = gaussian.predict(X_test)

acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

acc_gaussian
# Perceptron



perceptron = Perceptron()

perceptron.fit(X_train, Y_train)

Y_pred = perceptron.predict(X_test)

acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)

acc_perceptron
# Linear SVC



linear_svc = LinearSVC()

linear_svc.fit(X_train, Y_train)

Y_pred = linear_svc.predict(X_test)

acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)

acc_linear_svc




# Stochastic Gradient Descent



sgd = SGDClassifier()

sgd.fit(X_train, Y_train)

Y_pred = sgd.predict(X_test)

acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

acc_sgd







# Decision Tree



decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

acc_decision_tree



# Random Forest



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_predRF = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

acc_random_forest
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 

              'Stochastic Gradient Decent', 'Linear SVC', 

              'Decision Tree'],

    'Score': [acc_svc, acc_knn, acc_log, 

              acc_random_forest, acc_gaussian, acc_perceptron, 

              acc_sgd, acc_linear_svc, acc_decision_tree]})

models.sort_values(by='Score', ascending=False)
submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_predSVM

    })



#submission.head()

submission.to_csv('submission.csv', index=False)
