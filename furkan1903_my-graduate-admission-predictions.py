import numpy as np

import pandas as pd

import os

from matplotlib import pyplot as plt

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import BernoulliNB,GaussianNB

import seaborn as sns

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report

sns.set(style='white')

sns.set(style='whitegrid', color_codes=True)
df = pd.read_csv("../input/Admission_Predict.csv")

df.head()
df.rename(columns = {'Chance of Admit ':'Chance of Admit', 'LOR ':'LOR'}, inplace=True)

df.drop(labels='Serial No.', axis=1, inplace=True)
targets = df['Research']

features = df.drop(columns = {'Research'})



X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.3, random_state=42)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.fit_transform(X_test)
mean_chance = df['Research'].mean()
y_train_binary = (y_train > 0.72).astype(int)

y_test_binary = (y_test > 0.72).astype(int)
logreg = LogisticRegression()

logreg.fit(X_train, y_train_binary)

y_predict = logreg.predict(X_test)

logreg_score = (logreg.score(X_test, y_test_binary))*100

print(' Logistic Regression Accuracy = ',(logreg.score(X_test, y_test_binary))*100)

dec_tree = DecisionTreeClassifier(random_state=0, max_depth=6)

dec_tree.fit(X_train, y_train_binary)

y_predict = dec_tree.predict(X_test)

dec_tree_score = (dec_tree.score(X_test, y_test_binary))*100

print(' Decision Trees Accuracy = ',(dec_tree.score(X_test, y_test_binary))*100)

forest = RandomForestClassifier(n_estimators=110,max_depth=6,random_state=0)

forest.fit(X_train, y_train_binary)

y_predict = forest.predict(X_test)

forest_score = (forest.score(X_test, y_test_binary))*100

print(' Decision Trees Accuracy = ',(forest.score(X_test, y_test_binary))*100)

Methods = ['Logistic Regression', 'Decision Trees', 'Random Forests']

Scores = np.array([logreg_score, dec_tree_score, forest_score])



fig, ax = plt.subplots(figsize=(8,6))

sns.barplot(Methods, Scores)

plt.title('Algorithm Prediction Accuracies')

plt.ylabel('Accuracy')