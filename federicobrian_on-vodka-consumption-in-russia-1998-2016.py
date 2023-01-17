from sklearn.datasets import make_regression

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import PolynomialFeatures

import pandas as pd 

from matplotlib import pyplot as plt

import numpy as np

from sklearn.linear_model import Ridge
alcohol_ds = pd.read_csv("../input/alcohol-consumption-in-russia/russia_alcohol.csv", parse_dates = ['year'])
plt.rcParams['figure.figsize'] = [14, 10]



plt.xlabel('Years')

plt.ylabel('Vodka consumption in litres per capita')

plt.scatter(alcohol_ds.year, alcohol_ds.vodka, marker='.', color='blue')

plt.show()
vodka = list()

for i in range(0,19):

    tot = 0.0

    for j in range(0,85):

        num = float(alcohol_ds.vodka.tolist()[(i*85) + j])

        if "nan" not in str(num):

            tot += num

    vodka.append(tot/85)

    

print(vodka)
plt.rcParams['figure.figsize'] = [14, 10]



plt.xlabel('Years')

plt.ylabel('Vodka consumption in litres per capita')

plt.scatter(alcohol_ds.year.unique(), vodka, marker='o', color='blue')

plt.show()
alphas = [.1, .5, 1, 10, 100]

predictions = []

years = StandardScaler().fit_transform(alcohol_ds.year.unique().reshape(-1, 1))



for alpha in alphas:

    ridge_reg = Ridge(alpha = alpha)

    ridge_reg.fit(years, vodka) 

    predictions.append(ridge_reg.predict(years))

plt.rcParams['figure.figsize'] = [14, 10]



plt.xlabel('Years')

plt.ylabel('Vodka consumption in litres per capita')

plt.scatter(alcohol_ds.year.unique(), vodka, marker='o', color='blue')



colors = ["green", "red", "purple", "black", "orange"]

i = 0 

for prediction in predictions:

    label = "alpha = " + str(alphas[i])

    plt.plot(alcohol_ds.year.unique(), prediction, color=colors[i], linewidth='1', label=label)

    

    i += 1

    

plt.legend()    

plt.show()
degrees = [1, 2, 4, 6, 10]

predictions = []



for degree in degrees:

    model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=alphas[-1]))

    model.fit(years, vodka)

    predictions.append(model.predict(years))

plt.rcParams['figure.figsize'] = [14, 10]



plt.xlabel('Years')

plt.ylabel('Vodka consumption in litres per capita')

plt.scatter(alcohol_ds.year.unique(), vodka, marker='o', color='blue')



colors = ["green", "red", "purple", "black", "orange"]

i = 0 

for prediction in predictions:

    label = "degree = " + str(degrees[i])

    plt.plot(alcohol_ds.year.unique(), prediction, color=colors[i], linewidth='1', label=label)

    

    i += 1

    

plt.legend()    

plt.show()