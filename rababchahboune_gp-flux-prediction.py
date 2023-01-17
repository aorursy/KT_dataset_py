import pandas as pd

import numpy as np

from sklearn.model_selection import cross_validate

from sklearn import preprocessing

from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
dataset = "../input"

data = pd.read_csv(dataset+'/training_set.csv')

list(data)

data
#first make the mjd integer and if we have duplicated values take the means

#put all this is a black box train it on GP then out put to a file

#input to model object and pace

#notes : TimeSeriesSplit

#update : correct the prdiction list then add cross validation.
objects = data.drop_duplicates(subset=['object_id'])['object_id']

data['mjd'] = data['mjd'].astype(int)

grouped = data[data['object_id'] == 615 ]

grouped = data.groupby(['mjd','passband']).mean()

grouped.reset_index(level=['passband','mjd'],inplace=True)

grouped.mjd = grouped.mjd - grouped.mjd.min()

subset = grouped[['mjd', 'passband']]

tuples = [tuple(x) for x in subset.values]
def generateY(tuples,interval,maxVal):

    allMjds = list(range(0,maxVal+1,interval))

    allPassbands = [i for i in range(0,6)]

    allPredict = []

    for i in range(len(allMjds)) :

        allPredict = allPredict + [(m, p) for m in [allMjds[i]] for p in allPassbands]

    toPredict = set(allPredict) - set(tuples)

    toPredict = list(toPredict)

    toPredict.sort(key=lambda tup: tup[1])

    return pd.DataFrame(toPredict,columns=['mjd', 'passband'])
p = generateY(tuples,1,data.mjd.max() - data.mjd.min())

p[p['passband'] == 0 ].mjd
def gaussianFill(objectDf,maxVal,interval):

    grouped = objectDf.groupby(['mjd','passband']).mean()

    grouped.reset_index(level=['passband','mjd'],inplace=True)

    grouped.mjd = grouped.mjd - grouped.mjd.min()

    subset = grouped[['mjd', 'passband']]

    tuples = [tuple(x) for x in subset.values]

    predict = generateY(tuples,interval,maxVal)

    result = pd.DataFrame({'object_id':[],'mjd':[],'passband':[],'flux':[],'flux_err':[]})

    frames = []

    for i in range(0,6):

        flux = []

        filtered = grouped[grouped['passband'] == i ]

        X_train = filtered[['mjd']]

        y_train = filtered[['flux','flux_err']]

        kernel = DotProduct() + WhiteKernel()

        gaussian = GaussianProcessRegressor()

        gaussian.fit(X_train,y_train)

        predictable = pd.DataFrame(predict[predict['passband'] == 0 ].mjd,columns=['mjd'])

        flux = gaussian.predict(predictable)

        ids = [objectDf['object_id'].iloc[0]]*len(flux)

        passband = [i]*len(flux)

        df = pd.DataFrame({'object_id':ids,'passband':passband,'mjd':predictable.mjd,'flux':[i[0] for i in flux],'flux_err':[i[1] for i in flux]})

        frames = frames + [df]

    result = pd.concat(frames)

    return pd.concat(frames)
result = pd.DataFrame({'object_id':[],'mjd':[],'passband':[],'flux':[],'flux_err':[]})

predictions = []

for o in objects:

    predictions.append(gaussianFill(data[data['object_id'] == o ],data.mjd.max() - data.mjd.min(),1))

    #break;

result = pd.concat(predictions)

result.to_csv('flux_predictions.csv')