# Modules for data manipulation

import pandas as pd

import numpy as np



# machine learning packages

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
# load data

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
# Drop the ticket# and cabin# features.  They don't help w/ hypotheses.

train_df = train_df.drop(['Ticket','Cabin'], axis=1)

test_df = test_df.drop(['Ticket','Cabin'], axis=1)

combine = [train_df, test_df]



# Change the 'Name' field to title -- provides better analysis

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    #convert female / male to 1, 0

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

    #Convert titles to numeric categories -- helps w/ sklearn algorithms

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)

    

#Now drop name & passenger id from train; name from test

train_df = train_df.drop(['Name','PassengerId'], axis=1)

test_df = test_df.drop(['Name'], axis=1)

combine = [train_df, test_df]
#Create age estimator for samples with no age data

#Make estimator the average age for passenger by class & gender

ages_est = np.zeros((2,3))

for dataset in combine:

    #Create estimator

    for i in range(0, 2):

        for j in range(0, 3):

            guess_df = dataset[(dataset['Sex'] == i) & \

                                  (dataset['Pclass'] == j+1)]['Age'].dropna()

            ages_est[i,j] = guess_df.median()

    #Fill blanks with estimates            

    for i in range(0, 2):

        for j in range(0, 3):

            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\

                    'Age'] = ages_est[i,j]



    dataset['Age'] = dataset['Age'].astype(int)



#Create port of embarkation estimator for samples w/ no port data -- just use most common

freq_port = train_df.Embarked.dropna().mode()[0]

for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

    #Convert port of embarkation to numeric categories

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)



#Create fare estimator for samples with no fare data -- use median for Pclass

fare_est = np.zeros((3,1))

for dataset in combine:

    for i in range(0,3):

        fare_est[i] = dataset[dataset['Pclass']==i+1]['Fare'].dropna().median()

        dataset.loc[(dataset.Fare.isnull()) & (dataset.Pclass == i+1),'Fare'] = fare_est[i]



#Create age bands -- resulted in groups of 16 years

#train_df['AgeBand'] = pd.cut(train_df['Age'], 5)

for dataset in combine:

    dataset.loc[dataset['Age'] < 17, 'Age'] = 0

    dataset.loc[(dataset['Age'] >= 17) & (dataset['Age'] < 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] >= 32) & (dataset['Age'] < 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] >= 48), 'Age'] = 3

    dataset['child'] = 0

    dataset.loc[(dataset['Age'] == 0),'child'] = 1

    

#Create Fare bands -- use high/low for each passenger class

for dataset in combine:

    dataset.Fare = [1 if x > fare_est[y-1] else 0 for (x,y) in zip(dataset.Fare,dataset.Pclass)]
#Look at family size -- create IsAlone feature

for dataset in combine:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1    

 

#Search for mixture models

for dataset in combine:

    dataset['WealthFactor'] = 0

    dataset.loc[(dataset['Sex']==0)&(dataset['Pclass']==1),'WealthFactor'] = 1

    dataset.loc[(dataset['Sex']==1)&(dataset['Pclass']==3),'WealthFactor'] = -1



#Remove unnecessary variables

train_df = train_df.drop(['SibSp','Parch'], axis=1)

test_df = test_df.drop(['SibSp','Parch'], axis=1)
#Model the data

X_train = train_df.drop("Survived", axis=1)

Y_train = train_df["Survived"]

X_test  = test_df.drop("PassengerId", axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape



# Logistic Regression

logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)



coeff_df = pd.DataFrame(train_df.columns.delete(0))

coeff_df.columns = ['Feature']

coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

print("Logistic regression coefs are: \n",coeff_df.sort_values(by='Correlation', ascending=False),'\n')
# Now try several other methods

# Support Vector Machines

svc = SVC()

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

acc_svc = round(svc.score(X_train, Y_train) * 100, 2)



# K nearest neighbor

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)



# Gaussian Naive Bayes

gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

Y_pred = gaussian.predict(X_test)

acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)



# Perceptron

perceptron = Perceptron()

perceptron.fit(X_train, Y_train)

Y_pred = perceptron.predict(X_test)

acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)



# Linear SVC

linear_svc = LinearSVC()

linear_svc.fit(X_train, Y_train)

Y_pred = linear_svc.predict(X_test)

acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)



# Stochastic Gradient Descent

sgd = SGDClassifier()

sgd.fit(X_train, Y_train)

Y_pred = sgd.predict(X_test)

acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)



# Decision Tree

decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 3)



# Random Forest

random_forest = RandomForestClassifier(max_features = 6, n_estimators=250)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 3)



# Gradient Boosting Classifier

grad_boosting = GradientBoostingClassifier(n_estimators=250, max_depth=6)

grad_boosting.fit(X_train, Y_train)

Y_pred = grad_boosting.predict(X_test)

grad_boosting.score(X_train, Y_train)

acc_grad_boosting = round(grad_boosting.score(X_train, Y_train) * 100, 3)



#Now evaluate models as a group

models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 

              'Stochastic Gradient Decent', 'Linear SVC', 

              'Decision Tree', 'Gradient Boosting'],

    'Score': [acc_svc, acc_knn, acc_log, 

              acc_random_forest, acc_gaussian, acc_perceptron, 

              acc_sgd, acc_linear_svc, acc_decision_tree, acc_grad_boosting]})

print("The models as a whole perform as:\n",models.sort_values(by='Score', ascending=False))
#Finally submit the results

submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('../output/submission.csv', index=False)