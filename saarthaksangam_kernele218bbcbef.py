import numpy as np

import pandas as pd

#import os

from matplotlib import pyplot as plt

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

import seaborn as sns

sns.set(style='white')

sns.set(style='whitegrid', color_codes=True)
df = pd.read_csv("../input/Admission_Predict.csv")

df.head()
df.describe()
df.rename(columns = {'Chance of Admit ':'Chance of Admit', 'LOR ':'LOR'}, inplace=True)

df.drop(labels='Serial No.', axis=1, inplace=True)
fig, ax = plt.subplots(figsize=(10,10))

sns.heatmap(df.corr(), annot=True, cmap='Blues')
plt.figure(figsize=(20,6))

plt.subplot(1,2,1)

sns.distplot(df['CGPA'])

plt.title('CGPA Distribution of Applicants')



plt.subplot(1,2,2)

sns.regplot(df['CGPA'], df['Chance of Admit'])

plt.title('CGPA vs Chance of Admit')
plt.figure(figsize=(20,6))

plt.subplot(1,2,1)

sns.distplot(df['GRE Score'])

plt.title('Distributed GRE Scores of Applicants')



plt.subplot(1,2,2)

sns.regplot(df['GRE Score'], df['Chance of Admit'])

plt.title('GRE Scores vs Chance of Admit')
plt.figure(figsize=(20,6))

plt.subplot(1,2,1)

sns.distplot(df['TOEFL Score'])

plt.title('Distributed TOEFL Scores of Applicants')



plt.subplot(1,2,2)

sns.regplot(df['TOEFL Score'], df['Chance of Admit'])

plt.title('TOEFL Scores vs Chance of Admit')
fig, ax = plt.subplots(figsize=(8,6))

sns.countplot(df['Research'])

plt.title('Research Experience')

plt.ylabel('Number of Applicants')

ax.set_xticklabels(['No Research Experience', 'Has Research Experience'])
fig, ax = plt.subplots(figsize=(8,6))

sns.countplot(df['University Rating'])

plt.title('University Rating')

plt.ylabel('Number of Applicants')
targets = df['Chance of Admit']

features = df.drop(columns = {'Chance of Admit'})



X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.fit_transform(X_test)
linreg = LinearRegression()

linreg.fit(X_train, y_train)

y_predict = linreg.predict(X_test)

linreg_score = (linreg.score(X_test, y_test))*100

linreg_score
dec_tree = DecisionTreeRegressor(random_state=0, max_depth=6)

dec_tree.fit(X_train, y_train)

y_predict = dec_tree.predict(X_test)

dec_tree_score = (dec_tree.score(X_test, y_test))*100

dec_tree_score
forest = RandomForestRegressor(n_estimators=110,max_depth=6,random_state=0)

forest.fit(X_train, y_train)

y_predict = forest.predict(X_test)

forest_score = (forest.score(X_test, y_test))*100

forest_score
Methods = ['Linear Regression', 'Decision Trees', 'Random Forests']

Scores = np.array([linreg_score, dec_tree_score, forest_score])



fig, ax = plt.subplots(figsize=(8,6))

sns.barplot(Methods, Scores)

plt.title('Algorithm Prediction Accuracies')

plt.ylabel('Accuracy')
myGRE = int(input("Enter GRE Marks (260-340): "))

myTOEFL = int(input("Enter TOEFL Marks (0-120): "))

myUniv = int(input("Enter Undergrad University Tier (1-4): "))

mySOP = float(input("Enter SOP Rating (0-5 (0.5 increments)): "))

myLOR = float(input("Enter LOR Rating (0-5 (0.5 increments)): "))

myCGPA = float(input("Enter CGPA: "))

myResearch = float(input("Done Research (0/1): "))

my_prof = [[myGRE, myTOEFL, myUniv, mySOP, myLOR, myCGPA, myResearch]]

my_prof_scaled = scaler.fit_transform(my_prof)

y_predict_scaled = linreg.predict(my_prof_scaled)

y_predict_scaled

y_pred_per = str(round(y_predict_scaled[0]*100,2))

print("Your chances of getting into UCLA are "+ y_pred_per + "%")