import matplotlib

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import sklearn
#loading data

oecd_bli = pd.read_csv('../input/life-satisfaction-dataset/oecd_bli_2015.csv', thousands=',')

gdp_per_capita = pd.read_csv('../input/life-satisfaction-dataset/gdp_per_capita.csv', thousands=',', delimiter='\t', encoding='latin1', na_values="n/a")
# Checking the life satisfaction datsummary

oecd_bli.info()

oecd_bli.head(5)
# checking the gdp data summary

gdp_per_capita.info()

gdp_per_capita.head(5)
def prepare_country_stats(oecd_bli, gdp_per_capita):

    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"] == "TOT"]

    oecd_bli = oecd_bli.pivot(

        index="Country", columns="Indicator", values="Value")

    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)

    gdp_per_capita.set_index("Country", inplace=True)

    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,

                                  left_index=True, right_index=True)

    full_country_stats.sort_values(by="GDP per capita", inplace=True)

    remove_indices = [0, 1, 6, 8, 33, 34, 35]

    keep_indices = list(set(range(36)) - set(remove_indices))

 

    return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]
country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)

X = np.c_[country_stats["GDP per capita"]]

y = np.c_[country_stats["Life satisfaction"]]
# plotting life satisfaction against GDP per capita

country_stats.plot(kind='scatter', x="GDP per capita", y='Life satisfaction')

plt.show()
from sklearn.linear_model import LinearRegression



# selecting a liner model

lin_reg_model = LinearRegression()

lin_reg_model.fit(X, y)
# and now for making a prediction for cyprus

X_new = [[22587]]

print(lin_reg_model.predict(X_new))
from sklearn.neighbors import KNeighborsRegressor

clf = KNeighborsRegressor(n_neighbors=3)

clf.fit(X, y)

X_new = [[22587]]

print(clf.predict(X_new))