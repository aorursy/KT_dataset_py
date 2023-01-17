import os

import pandas as pan

import numpy as np

import matplotlib.pyplot as plt



orgdf = pan.read_csv("../input/california-housing-dataset/housing.csv")

orgdf.info()
from mpl_toolkits.basemap import Basemap



# 1. Draw the map background

fig = plt.figure(figsize=(8, 8))

m = Basemap(projection='lcc', resolution='l', 

            lat_0=37.5, lon_0=-119,

            width=1E6, height=1.2E6)

m.shadedrelief()

m.drawcoastlines(color='black')

m.drawcountries(color='black')

m.drawstates(color='black')



lon = np.array(orgdf['longitude'])

lat = np.array(orgdf['latitude'])

price = np.array(orgdf['median_house_value'])

popu = np.array(orgdf['population'])



m.scatter(lon, lat, latlon=True,

          c=price, s=popu/50,

          cmap='RdBu_r', alpha=0.9)



plt.colorbar(label=r'House value')

plt.clim(min(price), 300000)



for a in [100, 500]:

    plt.scatter([], [], c='k', alpha=0.5, s=a,

                label=str(a) + ' population density')

plt.legend(scatterpoints=1, frameon=False,

           labelspacing=1, loc='lower left');
conv = {'<1H OCEAN':1.0,'INLAND':2.0,'ISLAND':3.0,'NEAR BAY':4.0,'NEAR OCEAN':5.0}

orgdf['ocean_proximity'] = orgdf['ocean_proximity'].map(lambda x: conv[x])

X = orgdf.drop('median_house_value', axis=1)

Y = orgdf['median_house_value']

X = X.fillna(X.median())

Y = Y.fillna(Y.median())
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import StandardScaler



model = Pipeline([

    ("poly_features", PolynomialFeatures(degree=2, include_bias=False)),

    ("std_scaler", StandardScaler()),

    ("lin_reg", LinearRegression())])



model.fit(X_train, Y_train)

Y_train_predict = model.predict(X_train)

Y_test_predict = model.predict(X_test)

r2tr = r2_score(Y_train, Y_train_predict)

r2es = r2_score(Y_test, Y_test_predict)

print( 'R2 of the training dataset: ' , str(np.around( r2tr , 2)) )

print( 'R2 of the test dataset: ' , str(np.around( r2es , 2)) )
def plot_learning_curves(model, X, y):

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)

    train_errors, val_errors = [], []

    for m in range(1, len(X_train)):

        model.fit(X_train[:m], y_train[:m])

        y_train_predict = model.predict(X_train[:m])

        y_val_predict = model.predict(X_val)

        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))

        val_errors.append(mean_squared_error(y_val, y_val_predict))



    plt.figure(figsize=(8, 5), dpi=80, facecolor='w', edgecolor='k')

    plt.plot(np.sqrt(train_errors), "r-+" ,linewidth=2, label="train")

    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")

    plt.legend(loc="upper right", fontsize=14)   # not shown in the book

    plt.xlabel("Training set size", fontsize=14) # not shown

    plt.ylabel("RMSE", fontsize=14)              # not shown

    plt.ylim(0, 500000)



plot_learning_curves( model , X_train[:1400] , Y_train[:1400] )

plt.show()