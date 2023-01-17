import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
dataset = pd.read_csv('../input/Admission_Predict_Ver1.1.csv')
dataset.columns
dataset.head()
dataset.info()
dataset.describe()
plt.figure(1, figsize=(10,6))
plt.subplot(1,3, 1)
plt.boxplot(dataset['GRE Score'])
plt.title('GRE Score')

plt.subplot(1,3,2)
plt.boxplot(dataset['TOEFL Score'])
plt.title('TOEFL Score')

plt.subplot(1,3,3)
plt.boxplot(dataset['University Rating'])
plt.title('University Rating')

plt.show()
university_rating = dataset.groupby('University Rating')['GRE Score'].mean()
plt.bar(university_rating.index, university_rating.values)
plt.title('University Rating X GRE Score')
plt.ylabel('GRE Score')
plt.xlabel('University Rating')
plt.show()
pd.DataFrame(dataset.corr()['Chance of Admit '])
dataset.drop(columns=['Serial No.'], axis=1, inplace=True)
plt.figure(figsize=(10,6))
plt.scatter(dataset['GRE Score'], dataset['TOEFL Score'])
plt.title('GRE Score X TOEFL Score')
plt.xlabel('GRE Score')
plt.ylabel('TOEFL Score')
plt.show()
dataset['GRE Score'] = preprocessing.StandardScaler().fit_transform(dataset['GRE Score'].values.reshape(-1,1))
dataset['TOEFL Score'] = preprocessing.StandardScaler().fit_transform(dataset['TOEFL Score'].values.reshape(-1,1))
X_train, X_test, y_train, y_test = train_test_split(dataset.iloc[:,0:], dataset.iloc[:,-1], random_state=42)
from sklearn import linear_model
lr = linear_model.Ridge(alpha=0.5)
lr.fit(X_train, y_train)
lr.score(X_test,y_test)
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()
rfr.fit(X_train, y_train)
rfr.score(X_test,y_test)