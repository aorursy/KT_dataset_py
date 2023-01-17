#tout les imports utilisé

import numpy as np

import pandas as pd

import random



from sklearn.preprocessing import PolynomialFeatures

from sklearn import linear_model, metrics

from sklearn.model_selection import cross_val_score, train_test_split



from datetime import datetime, timedelta

from dateutil.relativedelta import *



import statistics 



import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
#charger le dataset

dataset = pd.read_csv("../input/housesalesprediction/kc_house_data.csv")

dataset.head(4)

# dataset cleaning

dataset['date'] = dataset['date'].apply(lambda x: x.split("T")[0])

dataset.head(4)
# split dataset to train, test, validation dataset for train_test validation

train_val, test = train_test_split(dataset, test_size=0.2)

train, validation = train_test_split(train_val, test_size=0.25)

#plot for watch repartition

datasetName = ('Train', 'Test', 'Validation', 'Total')

y_pos = np.arange(len(datasetName))

performance = [len(train),len(test),len(validation),len(dataset)]



plt.bar(y_pos, performance, align='center', alpha=0.5)

plt.xticks(y_pos, datasetName)

plt.ylabel('Nombre de valeurs')

plt.title('Répartition des datasets')



plt.show()
df1=dataset[['price', 'bedrooms', 'bathrooms', 'sqft_living',

    'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',

    'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',

    'lat', 'long', 'sqft_living15', 'sqft_lot15']]

h = df1.hist(bins=25,figsize=(16,16),xlabelsize='10',ylabelsize='10',xrot=-15)

sns.despine(left=True, bottom=True)

[x.title.set_size(12) for x in h.ravel()];

[x.yaxis.tick_left() for x in h.ravel()];
sns.set(style="whitegrid", font_scale=1)

f, axes = plt.subplots(1, 2,figsize=(15,5))

sns.boxplot(x=dataset['bedrooms'],y=dataset['price'], ax=axes[0])

sns.boxplot(x=dataset['floors'],y=dataset['price'], ax=axes[1])

sns.despine(left=True, bottom=True)

axes[0].set(xlabel='Bedrooms', ylabel='Price')

axes[0].yaxis.tick_left()

axes[1].yaxis.set_label_position("right")

axes[1].yaxis.tick_right()

axes[1].set(xlabel='Floors', ylabel='Price')



f, axe = plt.subplots(1, 1,figsize=(12.18,5))

sns.despine(left=True, bottom=True)

sns.boxplot(x=dataset['bathrooms'],y=dataset['price'], ax=axe)

axe.yaxis.tick_left()

axe.set(xlabel='Bathrooms / Bedrooms', ylabel='Price');

#observable features

features = ['date','bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',

       'waterfront', 'view', 'condition', 'grade', 'sqft_above',

       'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long',

       'sqft_living15', 'sqft_lot15']

#price

y = 'price'
#model use polynomilear regression linear (degree = 2) after test this is a best degree

polyfeat = PolynomialFeatures(degree=2)

X_allpoly = polyfeat.fit_transform(dataset[features])

X_trainpoly = polyfeat.fit_transform(train[features])

X_testpoly = polyfeat.fit_transform(test[features])

X_valpoly = polyfeat.fit_transform(validation[features])

#train model

regressor = linear_model.LinearRegression()

poly = regressor.fit(X_trainpoly, train[y])
#predict model, accuracy

pred1 = poly.predict(X_testpoly)

rmsepoly1 = float(format(np.sqrt(metrics.mean_squared_error(test[y],pred1)),'.3f'))

rtrpoly1 = float(format(poly.score(X_trainpoly,train[y]),'.3f'))

rtepoly1 = float(format(poly.score(X_testpoly,test[y]),'.3f'))

cv1 = float(format(cross_val_score(linear_model.LinearRegression(), X_valpoly,validation[y]).mean(),'.3f'))

print(f"accuracy: {cv1}")
def allDate(year=2, date=datetime.now(), format_str='%Y%m%d'):

    allDate = list()

    for i in range(year):

        for k in range(12):

            d = date + relativedelta(months=+i, days=+k)

            allDate.append(int(d.strftime(format_str)))

    return allDate
d = allDate()

def meanMetrics(features=[], exept=[]):

    metrics = list()

    for feature in features:

        if feature not in exept:

            metrics.append(dataset[feature][5])

    return metrics

metrics = meanMetrics(features=features,exept=['date'])

metrics.insert(0, 20141013)

price = poly.predict(polyfeat.fit_transform([metrics]))[0]

price = float(format(price,'.2f'))
def priceBym2Living(allDate):

    result = list()

    for date in allDate:

        metrics = meanMetrics(features=features,exept=['date'])

        metrics.insert(0, date)

        metrics[1] = random.randint(1, 3)

        metrics[2] = random.randint(1, 2)

        metrics[3] = random.randint(400, 700)

        metrics[5] = 3

        price = poly.predict(polyfeat.fit_transform([metrics]))[0]

        price = float(format(price / 1000000,'.2f'))

        priceByM2 = float(format(price / metrics[3],'.2f'))

        result.append(priceByM2)

    return result
def priceBym2Total(allDate):

    result = list()

    for date in allDate:

        metrics = meanMetrics(features=features,exept=['date'])

        metrics.insert(0, date)

        price = poly.predict(polyfeat.fit_transform([metrics]))[0]

        price = float(format(price,'.2f'))

        priceByM2 = float(format(price / metrics[4],'.2f'))

        result.append(priceByM2)

    return result
def priceByBedRoom(allDate, bedroom):

    result = list()

    for date in allDate:

        metrics = meanMetrics(features=features,exept=['date'])

        metrics.insert(0, date)

        metrics[1] = bedroom

        price = poly.predict(polyfeat.fit_transform([metrics]))[0]

        price = float(format(price,'.2f'))

        priceByM2 = float(format(price / metrics[4],'.2f'))

        result.append(priceByM2)

    return result
def graphDate(year=2, date=datetime.now(), format_str='%Y-%m'):

    graphDate = list()

    for i in range(year):

        for k in range(12):

            d = date + relativedelta(months=+k, years=i)

            graphDate.append(d.strftime(format_str))

    return graphDate
y = priceBym2Living(d)

yM2Total = priceBym2Total(d)
#Data Visualization

fig = plt.figure(figsize=(24,4))

ax = fig.add_subplot(111)

ax.plot(graphDate(), y)

ax.set(xlabel='temps', ylabel='prix du m²',

       title='Prix du m² habitable en fonction du temps')

ax.grid()

plt.show()
l = list()

k = 0

for date in graphDate():

    l.append([date, int(y[k])])

    k = k + 1

a = np.asarray(l)

my_df = pd.DataFrame(a)

my_df.to_csv('prix_maison.csv', index=False)