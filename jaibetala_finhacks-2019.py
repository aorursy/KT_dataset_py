#This cell is to import the required libraries needed to follow along

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import LabelEncoder

from sklearn import metrics

from sklearn.model_selection import train_test_split
#importing our dataset and getting it ready

melb_data = pd.read_csv("../input/melbourne-housing-snapshot/melb_data.csv")
melb_data.head()
melb_corr = melb_data.corr()

f, ax = plt.subplots(figsize=(11, 9))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(melb_corr, cmap=cmap, vmax=1.0, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
melb_data = melb_data.drop(['Date', 'Method', 'SellerG', 'CouncilArea', 'Address', 'Bedroom2'], axis=1)
#Dropping columns with text data.

optionUno = melb_data.drop(['Suburb', 'Type', 'Regionname'], axis = 1)

optionUno = optionUno.dropna(axis=0)

y = optionUno.Price

X = optionUno.drop('Price', axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

modelOne = RandomForestRegressor(random_state=1)

modelOne.fit(X_train, y_train)

y_pred = modelOne.predict(X_test)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_pred, y_test))
#Changing text data to numerical

optionDeux = melb_data

optionDeux = optionDeux.dropna(axis=0)

label_encode = LabelEncoder()

optionDeux['Regionname'] = label_encode.fit_transform(optionDeux['Regionname'])

optionDeux['Type'] = label_encode.fit_transform(optionDeux['Type'])

optionDeux['Suburb'] = label_encode.fit_transform(optionDeux['Suburb'])

y2 = optionDeux.Price

X2 = optionDeux.drop(['Price'], axis = 1)



X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, random_state=1)

modelTwo = RandomForestRegressor(random_state=1)

modelTwo.fit(X_train2, y_train2)

y_pred2 = modelTwo.predict(X_test2)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_pred2, y_test2))
thirdOp = melb_data

thirdOp = thirdOp.dropna(axis=0)

features = ['Rooms', 'Landsize', 'Lattitude', 'Longtitude', 'Bathroom']

modelThree = RandomForestRegressor(random_state=5)

y3 = thirdOp.Price

X3 = thirdOp[features]

X_train3, X_test3, y_train3, y_test3 = train_test_split(X3, y3, random_state=1)

print(X_test3)

modelThree.fit(X_train3, y_train3)

y_pred3 = modelThree.predict(X_test3)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_pred3, y_test3))