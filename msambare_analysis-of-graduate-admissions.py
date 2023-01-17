import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('../input/graduate-admissions/Admission_Predict_Ver1.1.csv')

df.columns = ['Serial No.','GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research', 'Chance of Admit']

df.head()
df = df.drop(['Serial No.'], axis=1)

df.isnull().sum()
df.describe()
fig = plt.figure(figsize=(14, 21))

fig.add_subplot(321)

sns.distplot(df['GRE Score'], kde=True)

plt.title("Distribution of GRE Scores")



fig.add_subplot(322)

sns.distplot(df['TOEFL Score'], kde=True)

plt.title("Distribution of TOEFL Scores")



fig.add_subplot(323)

sns.distplot(df['CGPA'], kde=True)

plt.title("Distribution of CGPA")



fig.add_subplot(324)

sns.distplot(df['SOP'], kde=False)

plt.title("Distribution of SOP Ratings")



fig.add_subplot(325)

sns.distplot(df['University Rating'], kde=False)

plt.title("Distribution of University Rating")



plt.show()
fig = plt.figure(figsize=(14, 14))

fig.add_subplot(221)

sns.regplot(df['GRE Score'],df['Chance of Admit'])

plt.title("GRE Scores vs Chance of Admit")



fig.add_subplot(222)

sns.regplot(df['TOEFL Score'],df['Chance of Admit'])

plt.title("TOEFL Scores vs Chance of Admit")



fig.add_subplot(223)

sns.regplot(df['CGPA'],df['Chance of Admit'])

plt.title("CGPA vs Chance of Admit")



plt.show()
corr = df.corr()

fig, ax = plt.subplots(figsize=(8, 8))

colormap = sns.diverging_palette(220, 10, as_cmap=True)

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

sns.heatmap(corr, cmap=colormap, linewidths=.5, annot=True, fmt=".2f", mask=mask)

plt.show()
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(df.iloc[:,:-1],df.iloc[:,-1],test_size = 0.20, shuffle=False)
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR

from sklearn.metrics import accuracy_score, mean_squared_error



models =[['Linear Regression', LinearRegression()],

           ['Random Forest',RandomForestRegressor()],

           ['K-Neighbours', KNeighborsRegressor(n_neighbors = 2)],

           ['SVM', SVR()]]



model_output = {}

for name,model in models:

    model = model

    model.fit(X_train, Y_train)

    pred = model.predict(X_test)

    model_output[f'{name}'] = np.sqrt(mean_squared_error(Y_test, pred))



results = pd.DataFrame(model_output.items())

results.columns = ['Model', 'RMSE']

results.index = np.arange(1,len(results)+1)

results = results.sort_values(by=['RMSE'], ascending=True)

print("Models Trained")
fig = plt.figure(figsize=(8, 8))

sns.barplot(results['Model'],results['RMSE'],palette=reversed(sns.color_palette("rocket")))

plt.title("Comparing all the Models")

plt.show()
model = RandomForestRegressor()

X = df.iloc[:,:-1]

Y = df.iloc[:,-1]

model.fit(X,Y)

feature_names = X.columns

features = pd.DataFrame()

features['Features'] = X.columns

features['Importance'] = model.feature_importances_

features = features.sort_values(by=['Importance'], ascending=False)

features.index = np.arange(1,len(X.columns)+1)
fig = plt.figure(figsize=(8, 8))

sns.barplot(features['Features'],features['Importance'],palette=sns.color_palette("rocket"))

plt.title("Feature Importance")

plt.show()