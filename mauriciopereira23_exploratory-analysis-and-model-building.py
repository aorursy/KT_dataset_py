#basic

import numpy as np

import pandas as pd



#file system

import os

print(os.listdir("../input"))



#other

import random as rnd



#vizualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



#machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier



#rmse

from sklearn.metrics import mean_squared_error

from math import sqrt
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

combine = [train_df,test_df]
print(train_df.columns.values)

print(test_df.columns.values)

train_df.info()

print('_'*40)

test_df.info()
train_df.describe()
train_df.describe(include=['O'])
#correlation matrix

corrmat = train_df.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);
#only saleprice matrix

k = 10 #number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train_df[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)



train_df = train_df[['Id','OverallQual', 'GrLivArea', 'TotalBsmtSF','SalePrice']]

test_df = test_df[['Id','OverallQual', 'GrLivArea', 'TotalBsmtSF']]

combine = [train_df, test_df]



"After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape
train_df.describe()
#scatter plot

plt.scatter(train_df['GrLivArea'], train_df['SalePrice']);
#the outliers are the ones with a high GrLivArea

sorted_train_df = train_df.sort_values(by = 'GrLivArea', ascending = False)[:2]

sorted_train_df.head()
#Deleting the outliers

train_df = train_df.drop(train_df[train_df['Id'] == 1299].index)

train_df = train_df.drop(train_df[train_df['Id'] == 524].index)

train_df = train_df.drop(train_df[train_df['GrLivArea']>=3000].index)

train_df = train_df.drop(train_df[train_df['TotalBsmtSF']<0].index)
train_df.describe()
train_df['GrLivAreaBand'] = pd.cut(train_df['GrLivArea'], 5)

train_df[['GrLivAreaBand', 'SalePrice']].groupby(['GrLivAreaBand'], as_index=False).count().sort_values(by='GrLivAreaBand', ascending=True)
train_df['TotalBsmtSFBand'] = pd.cut(train_df['TotalBsmtSF'], 5)

train_df[['TotalBsmtSFBand', 'SalePrice']].groupby(['TotalBsmtSFBand'], as_index=False).count().sort_values(by='TotalBsmtSFBand', ascending=True)
#Combining new dataframes

combine = [train_df,test_df]
for dataset in combine:

    dataset.loc[ dataset['GrLivArea'] <= 862.8, 'GrLivArea'] = 0

    dataset.loc[(dataset['GrLivArea'] > 862.8) & (dataset['GrLivArea'] <= 1391.6), 'GrLivArea'] = 1

    dataset.loc[(dataset['GrLivArea'] > 1391.6) & (dataset['GrLivArea'] <= 1920.4), 'GrLivArea'] = 2

    dataset.loc[(dataset['GrLivArea'] > 1920.4) & (dataset['GrLivArea'] <= 2449.2), 'GrLivArea'] = 3

    dataset.loc[ dataset['GrLivArea'] > 2449.2, 'GrLivArea'] = 4

    #dataset['GrLivArea'] = dataset['GrLivArea'].astype(int)



    dataset.loc[ dataset['TotalBsmtSF'] <= 641.2, 'TotalBsmtSF'] = 0

    dataset.loc[(dataset['TotalBsmtSF'] > 641.2) & (dataset['TotalBsmtSF'] <= 1282.4), 'TotalBsmtSF'] = 1

    dataset.loc[(dataset['TotalBsmtSF'] > 1282.4) & (dataset['TotalBsmtSF'] <= 1923.6), 'TotalBsmtSF'] = 2

    dataset.loc[(dataset['TotalBsmtSF'] > 1923.6) & (dataset['TotalBsmtSF'] <= 2564.8), 'TotalBsmtSF'] = 3

    dataset.loc[ dataset['TotalBsmtSF'] > 2564.8, 'TotalBsmtSF'] = 4

    #dataset['TotalBsmtSF'] = dataset['TotalBsmtSF'].astype(int)



train_df = train_df.drop(['GrLivAreaBand'], axis=1)

train_df = train_df.drop(['TotalBsmtSFBand'], axis=1)

combine = [train_df, test_df]

train_df.head(10)

train_df.tail(10)
train_df.describe()
train_df.isnull().sum().max()
train_df.isna().sum()
train_df.max()
print(train_df.Id[train_df.Id == np.inf].count())

print(train_df.Id[train_df.Id == -np.inf].count())

print(train_df.OverallQual[train_df.OverallQual == np.inf].count())

print(train_df.OverallQual[train_df.OverallQual == -np.inf].count())

print(train_df.GrLivArea[train_df.GrLivArea == np.inf].count())

print(train_df.GrLivArea[train_df.GrLivArea == -np.inf].count())

print(train_df.TotalBsmtSF[train_df.TotalBsmtSF == np.inf].count())

print(train_df.TotalBsmtSF[train_df.TotalBsmtSF == -np.inf].count())

print(train_df.SalePrice[train_df.SalePrice == np.inf].count())

print(train_df.SalePrice[train_df.SalePrice == -np.inf].count())
#droping house Id and Sale Price

X_train = train_df.drop(['Id', 'SalePrice'], axis=1)

Y_train = train_df["SalePrice"]

X_test  = test_df.drop("Id", axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape
X_train.describe()
Y_train.describe()
X_test['TotalBsmtSF'][660] = 1.151578
X_test.describe()
logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

acc_log
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

acc_random_forest
# Support Vector Machines



svc = SVC()

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

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

Y_pred = random_forest.predict(X_test)

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

        "Id": test_df["Id"],

        "SalePrice": Y_pred

    })

submission.to_csv('submission.csv', index=False)