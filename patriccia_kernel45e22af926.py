import pandas as pd

import numpy as np

from numpy import nan as Nan
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/tmdb-box-office-prediction/train.csv')
train.shape
train.head()
from numpy import nan as Nan

import ast

belongs_to_collection_data = pd.DataFrame(index=['id', 'name', 'poster_path', 'backdrop_path'])

belongs_to_collection_data = pd.DataFrame(ast.literal_eval(train['belongs_to_collection'].iloc[0]))

nan = pd.DataFrame([Nan,Nan,Nan,Nan], index=['id', 'name', 'poster_path', 'backdrop_path']).T

for i in range(1,3000) :

    try :

        belongs_to_collection_data = belongs_to_collection_data.append(ast.literal_eval(train['belongs_to_collection'].iloc[i]))

    except :

        belongs_to_collection_data = belongs_to_collection_data.append(nan)

train = pd.merge(train, belongs_to_collection_data['name'], left_index=True, right_index=True)
box = train[['budget', 'original_language', 'popularity', 'runtime', 'revenue']]

box.head()
english = box['original_language'] == 'en'
#je divise mon dataset en english et other

box_en = box[english]

box_other = box[english == False]
box_other.shape
box_en.shape
# X

X = box_en[['budget', 'popularity', 'runtime']]
# y

y = box_en['revenue']
box_en.isna().sum()
y.isna().sum()
box_en = box_en[box_en.budget != 0]
box_en = box_en[box_en.popularity != 0]
box_en = box_en[box_en.runtime != 0]
box_en = box_en[box_en.revenue != 0]
box_en.shape
#Diviser le dataset entre training et test set

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
#standardiser

from sklearn.preprocessing import StandardScaler



sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
#time to train algorithm

from sklearn.linear_model import LinearRegression



regressor = LinearRegression()

regressor.fit(X_train, y_train) #training the algorith
#Afficher les coefficients de la régression

print(regressor.intercept_)

print(regressor.coef_)
#calcul de la prédiction

y_predict = regressor.predict(X_test)

y_predict
# Calculer le MSE, RMSE et le R Squared

from sklearn.metrics import r2_score

from sklearn import metrics



print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_predict))

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_predict)))

print('r2_score:',r2_score(y_test, y_predict))
# Afficher le graph de la régression

import matplotlib.pyplot as plt



plt.scatter(y_predict, y_test, color='blue')

plt.plot(y_predict,y_predict, color='red', linewidth=2)

plt.xlabel('y_predict')

plt.ylabel('x_test')

plt.show()
y_predict.shape