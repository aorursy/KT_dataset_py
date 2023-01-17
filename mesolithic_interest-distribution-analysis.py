import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from Levenshtein import distance

from sklearn import preprocessing, tree



data = pd.read_json('../input/train.json', convert_dates=["created"])

pg, bins = pd.qcut(data["price"], 8, retbins=True, labels=False)

data['pg'] = pg

data['low'] = 0.0

data['medium'] = 0.0

data['high'] = 0.0

dist = {'low': 0.694683, 'medium': 0.227529, 'high': 0.077788} 



lvl = ['low', 'medium', 'high']

for i in lvl:

    data[i] = data['interest_level'] == i

    data[i] = data[i].astype(float)



data['pricet'] = data['price'] / (data['bedrooms'] + 1.0) 

data['rooms'] = data['bathrooms'] / data['bedrooms']

data['toobig'] = data['bedrooms'].apply(lambda x: 1 if x > 4 else 0)

data["num_photos"] = data["photos"].apply(len)

data["num_features"] = data["features"].apply(len)

data["nophoto"] = data["num_photos"].apply(lambda x: 1 if x == 0 else 0)

data["halfbr"] = data["bathrooms"].apply(lambda x: 0 if round(x) == x else 1)

data["address_distance"] = data[["street_address", "display_address"]].apply(lambda x: distance(*x), axis=1)

data["address_distance"] = data["address_distance"].apply(lambda x: 120 if x > 120 else x)

lat, bins = pd.qcut(data["latitude"], 20, retbins=True, labels=False)

data['latbin'] = lat

lon, bins = pd.qcut(data["longitude"], 20, retbins=True, labels=False)

data['lonbin'] = lon

data["month"] = data["created"].apply(lambda x: x.month)

data["yearmonth"] = data["created"].apply(lambda x: str(x.year) + "_" + str(x.month))

data["day"] = data["created"].apply(lambda x: x.dayofweek)



data['chars'] = len(data['description'])

data['exclaim'] = [x.count('!') for x in data['description']]

data['shock'] = data['exclaim'] * 1.0 / data['chars'] * 100

data['first'] = data['description'].apply(lambda x: x.index('!') if '!' in x else 0)

data['spam'] = data['exclaim'] / data['first'] * 100 



# cross variables

abc_list = []

classes=12

for i in range(97, 123):

    abc_list.append(str(chr(i)))

train_lon, lon_bins = pd.qcut(data["longitude"], classes, retbins=True, labels=abc_list[0:classes])

train_lat, lat_bins = pd.qcut(data["latitude"], classes, retbins=True, labels=abc_list[0:classes])

train_lon = train_lon.astype(object)

train_lat = train_lat.astype(object)

data["grid"] = train_lon + train_lat

le = preprocessing.LabelEncoder()

le.fit(data["grid"])

data["grid"] = le.transform(data["grid"])



clf = tree.DecisionTreeClassifier()

params = ['bedrooms', 'bathrooms', 'num_features', 'grid']

clf = clf.fit(data[params], data['price'])

data["exp_price"] = pd.DataFrame(clf.predict(data[params]).tolist()).set_index(data.index)

data["overprice"] = data["price"] - data["exp_price"]



# second argument for binning continuous features



def doit(vari, bins=0):

    df = data[:]

    if bins:

        df[vari], labels = pd.qcut(df[vari], bins, retbins=True, labels=False)

        print('Bin intervals')

        print(labels.astype(int))

    a = df.groupby(vari)[lvl].sum()

    b = df.groupby(vari)[lvl].count()

    c = a/b

    d = pd.DataFrame(dist, c.index)

    e = (c-d) * 100

    

    ax = e.plot()

    ax.set_ylabel('Deviation from global (%)')

    plt.axhline(y=0, color='k', linestyle='--')

    plt.show()

    plt.close()

    e.plot(kind='hist', alpha=0.5)

    plt.show()

    plt.close()
doit('spam', 5)
# price group

doit('price', 10)
doit('listing_id', 20)
doit('pricet', 10)
doit('latitude', 20)
doit('latitude', 20)
doit('listing_id', 10)