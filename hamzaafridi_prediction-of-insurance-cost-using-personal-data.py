import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import r2_score
#import data to using pandas function read_csv and storing it into personal_data
personal_data = pd.read_csv('../input/insurance.csv')
#show the first five entries in personal_data
personal_data.head()
personal_data['sex'] = personal_data['sex'].map({'female': 1, 'male': 0})
personal_data['smoker'] = personal_data['smoker'].map({'yes':1,'no':0})
personal_data['region'] = personal_data['region'].map({'southwest':0,'southeast':1,'northwest':2,'northeast':3})
personal_data.head()
figure, ax = plt.subplots(2,4, figsize=(24,8))
sns.distplot(personal_data['age'],ax=ax[0,0])
sns.countplot(personal_data['sex'],ax=ax[0,1])
sns.distplot(personal_data['bmi'],ax= ax[0,2])
sns.distplot(personal_data['children'],ax= ax[0,3])
sns.countplot(personal_data['smoker'],ax= ax[1,0])
sns.countplot(personal_data['region'],ax= ax[1,1])
sns.distplot(personal_data['charges'],ax= ax[1,2])
train_personal_data, test_personal_data = train_test_split(personal_data, test_size=0.2)
print("size of train data set:", train_personal_data.shape)
print("size of test data set:", test_personal_data.shape)
# graph to see if there is any linearity and what is the co relation between independent features and charges
figure, ax = plt.subplots(2,3, figsize=(24,8))
sns.regplot(x=train_personal_data["age"], y=train_personal_data["charges"], ax=ax[0,0])
sns.regplot(x=train_personal_data["sex"], y=train_personal_data["charges"], ax=ax[0,1])
sns.regplot(x=train_personal_data["bmi"], y=train_personal_data["charges"], ax=ax[0,2])
sns.regplot(x=train_personal_data["children"], y=train_personal_data["charges"], ax=ax[1,0])
sns.regplot(x=train_personal_data["smoker"], y=train_personal_data["charges"], ax=ax[1,1])
sns.regplot(x=train_personal_data["region"], y=train_personal_data["charges"], ax=ax[1,2])

age_lm = linear_model.LinearRegression()
age_lm.fit(train_personal_data.as_matrix(['age']),train_personal_data.as_matrix(['charges']))

# Have a look at R sq to give an idea of the fit 
print('R sq: ',age_lm.score(train_personal_data.as_matrix(['age']),train_personal_data.as_matrix(['charges'])))

lm = linear_model.LinearRegression()
X = train_personal_data.drop('charges', axis = 1)
lm.fit (X,train_personal_data.charges)
print('R sq: ',lm.score(X,train_personal_data.charges))
X_test = test_personal_data.drop('charges', axis=1)
Y_Predicted = lm.predict(X_test)
print (r2_score(test_personal_data.charges,Y_Predicted))
sns.regplot(test_personal_data.charges, Y_Predicted)