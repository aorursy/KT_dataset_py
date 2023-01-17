import pandas as pd
import numpy as np
import sys
import random as rnd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
train_df = pd.read_csv("D:\Learning\ML_projects\Health_Insurance_cross_sell_prediction/train.csv")
test_df = pd.read_csv("D:\Learning\ML_projects\Health_Insurance_cross_sell_prediction/test.csv")
combine = [train_df, test_df]
print(train_df.columns.values)
train_df.head()
train_df.tail()
test_df.head()
test_df.tail()
train_df.info()
print("="*100)
test_df.info()
train_df.describe()
def health_in(data):
    correlation = data.corr()
    sns.heatmap(correlation, annot =True, cbar = True, cmap="RdYlGn")
    
health_in(train_df)
train_df = train_df.drop(['id'], axis=1)
test_df = test_df.drop(['id'], axis=1)
train_df = train_df.drop(['Region_Code'], axis=1)
test_df = test_df.drop(['Region_Code'], axis=1)
train_df.head()
train_df.loc[train_df['Gender'] == 'Male', 'Gender'] = 0
train_df.loc[train_df['Gender'] == 'Female', 'Gender'] = 1
test_df.loc[test_df['Gender'] == 'Male', 'Gender'] = 0
test_df.loc[test_df['Gender'] == 'Female', 'Gender'] = 1

train_df.loc[train_df['Vehicle_Age'] == '< 1 Year', 'Vehicle_Age'] = 0
train_df.loc[train_df['Vehicle_Age'] == '1-2 Year', 'Vehicle_Age'] = 1
train_df.loc[train_df['Vehicle_Age'] == '> 2 Years', 'Vehicle_Age'] = 2
test_df.loc[test_df['Vehicle_Age'] == '< 1 Year', 'Vehicle_Age'] = 0
test_df.loc[test_df['Vehicle_Age'] == '1-2 Year', 'Vehicle_Age'] = 1
test_df.loc[test_df['Vehicle_Age'] == '> 2 Years', 'Vehicle_Age'] = 2


train_df.head()
train_df.loc[train_df['Vehicle_Damage'] == 'Yes', 'Vehicle_Damage'] = 1
train_df.loc[train_df['Vehicle_Damage'] == 'No', 'Vehicle_Damage'] = 0
test_df.loc[test_df['Vehicle_Damage'] == 'Yes', 'Vehicle_Damage'] = 1
test_df.loc[test_df['Vehicle_Damage'] == 'No', 'Vehicle_Damage'] = 0

test_df.head()

train_df.head()
X_train = train_df.drop(['Response'], axis =1)
Y_train = train_df['Response']

X_test = test_df
X_train.shape, Y_train.shape, X_test.shape
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn 

linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc
sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest
models = pd.DataFrame({
    'Model': ['KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [ acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, 
              acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)
test_sf = pd.read_csv("D:\Learning\ML_projects\Health_Insurance_cross_sell_prediction/test.csv")

submission = pd.DataFrame({
        "Id": test_sf["id"],
        "Response": Y_pred
    })
submission.to_csv('D:\Learning\ML_projects\Health_Insurance_cross_sell_prediction/submission.csv', index=False)
