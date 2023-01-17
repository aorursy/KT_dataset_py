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

from sklearn.feature_selection import RFE

from sklearn.decomposition import PCA

from sklearn.ensemble import ExtraTreesClassifier
#Import the necessary data

test_df = pd.read_csv('../input/test.csv')

train_df = pd.read_csv('../input/train.csv')

combine = [train_df, test_df]
#determine the quality of the provided data in order to determine the proceedure for data cleaning.

#missing values

#train_df.info()

print ("percentage of missing values for age: ",(1-((714/891)))*100)

print ("Percentage of missing values for cabin: ",(1-((204/891)))*100)

print ("The mode of age in train dataset: ",train_df["Age"].mode())

print ("The mode of age in test dataset: ",test_df["Age"].mode())
train_df['Age'].fillna(24, inplace=True)

test_df['Age'].fillna(21, inplace=True)

test_df['Fare'].fillna(test_df['Fare'].mean(), inplace=True)
#remove the cabin column

train_df.drop(['Cabin','Ticket'], axis = 1, inplace = True)

test_df.drop(['Cabin','Ticket'], axis = 1, inplace = True)

combine = [train_df,test_df]
for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_df['Title'], train_df['Sex'])

for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()



title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in combine:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)



train_df.head()



print('Min age: ',train_df['Age'].min())

print('Max age: ',train_df['Age'].max())

print('Min fare: ',train_df['Fare'].min())

print ('Max fare: ',train_df['Fare'].max())

#split the ages up into groups

train_df['AgeRange'] = pd.cut(train_df['Age'], 8)

train_df[['AgeRange', 'Survived']].groupby(['AgeRange'], as_index=False).mean().sort_values(by='AgeRange', ascending=True)   
for dataset in combine:    

    dataset.loc[ dataset['Age'] <= 10.368, 'AgeBand'] = 0

    dataset.loc[(dataset['Age'] > 10.368) & (dataset['Age'] <= 20.315), 'AgeBand'] = 1

    dataset.loc[(dataset['Age'] > 20.315) & (dataset['Age'] <= 30.263), 'AgeBand'] = 2

    dataset.loc[(dataset['Age'] > 30.263) & (dataset['Age'] <= 40.21), 'AgeBand'] = 3

    dataset.loc[(dataset['Age'] > 40.21) & (dataset['Age'] <= 50.158), 'AgeBand'] = 4

    dataset.loc[(dataset['Age'] > 50.158) & (dataset['Age'] <= 60.105), 'AgeBand'] = 5

    dataset.loc[(dataset['Age'] > 60.105) & (dataset['Age'] <= 70.052), 'AgeBand'] = 6

    dataset.loc[ dataset['Age'] > 70.052, 'AgeBand']=7

    dataset['AgeBand'] = dataset['AgeBand'].astype(int)

#train_df.head()
plt.hist(train_df['Fare'], bins = 'auto')

plt.show()
train_df[train_df.Fare != 512.3292]
plt.hist(train_df['Fare'], bins = 'auto')

print ('Max fare: ',train_df['Fare'].max())

plt.show()
#now banding for fare

train_df['FareRange'] = pd.cut(train_df['Fare'], 6)

train_df[['FareRange', 'Survived']].groupby(['FareRange'], as_index=False).mean().sort_values(by='FareRange', ascending=True)

for dataset in combine:    

    dataset.loc[ dataset['Fare'] <= 43.833, 'FareBand'] = 0

    dataset.loc[(dataset['Fare'] > 43.833) & (dataset['Fare'] <= 87.667), 'FareBand'] = 1

    dataset.loc[(dataset['Fare'] > 87.667) & (dataset['Fare'] <= 131.5), 'FareBand'] = 2

    dataset.loc[(dataset['Fare'] > 131.5) & (dataset['Fare'] <= 175.333), 'FareBand'] = 3

    dataset.loc[(dataset['Fare'] > 175.333) & (dataset['Fare'] <= 219.167), 'FareBand'] = 4

    dataset.loc[ dataset['Fare'] > 219.167, 'FareBand'] = 5

    dataset['FareBand'] = dataset['FareBand'].astype(int)

train_df.head()

#Turn the sex category into a binary variable

gender_mapping = {"female": 0, "male": 1}

for dataset in combine:

    dataset['Sex'] = dataset['Sex'].map(gender_mapping)

    dataset['Sex'] = dataset['Sex'].fillna(0)



train_df.head()
#Need to turn embarked into a number

freq_port = train_df.Embarked.dropna().mode()[0]

freq_port

for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)



train_df.head()
for dataset in combine:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1



train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
#drop the 2 band column

train_df = train_df.drop(['FareRange','AgeRange','Name','Parch', 'SibSp', 'FamilySize'], axis=1)

combine = [train_df, test_df]
#set up the data for the models

X_train = train_df.drop(['Survived','PassengerId',], axis=1)

Y_train = train_df["Survived"]

X_test  = test_df.drop(['PassengerId','Name','Parch', 'SibSp', 'FamilySize'], axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape

#print (X_train.head())
#Feature selection using RFE

model = LogisticRegression()

rfe = RFE(model)

fit = rfe.fit(X_train, Y_train)

print("Num Features: ",fit.n_features_)

print("Selected Features: ",fit.support_)

print("Feature Ranking: ",fit.ranking_)

print(X_train.head())
print ("Top 4 features for RFE are: PClass, Sex, Embarked, Title")
#Feature selection using SelectKBest

test = SelectKBest(score_func=chi2, k=4)

fit = test.fit(X_train, Y_train)

# summarize scores

np.set_printoptions(precision=3)

print(fit.scores_)

features = fit.transform(X_train)

# summarize selected features

print(features[0:5,:])

print (X_train.head())
print ("Top 4 features for SelectKBest are: Sex, Fare Title & FareBand")
# feature extraction using extra trees classifier

model = ExtraTreesClassifier()

model.fit(X_train, Y_train)

print(model.feature_importances_)

X_train.head()
print ("Top 4 features  for Extra trees Classifier are: Fare, Sex, Age, & Pclass")
train_df.corr(method = 'pearson')
# For each X, calculate VIF and save in dataframe

from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()

vif["VIF Factor"] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

vif["features"] = X_train.columns

vif.round(1)
X_train = X_train.drop(['Age','Embarked','AgeBand','Fare','IsAlone'], axis=1)

Y_train = train_df["Survived"]

X_test  = X_test.drop(['Age','Embarked','AgeBand','Fare','IsAlone'], axis=1)

X_train.shape, Y_train.shape, X_test.shape

X_test.info()
 #Logistic Regression

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

Y_predKNN = knn.predict(X_test)

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

Y_predDT = decision_tree.predict(X_test)

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

        "Survived": Y_predRF

    })



#submission.head()

submission.to_csv('submission.csv', index=False)
