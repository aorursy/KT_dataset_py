# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import tensorflow as tf

import math

from tensorflow import keras

from sklearn import preprocessing

from sklearn.model_selection import cross_val_score

import keras.backend as kb

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



filepath = {}



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        filepath[filename[:-4]] = os.path.join(dirname, filename)





# Any results you write to the current directory are saved as output.
inputData = pd.read_csv(filepath["train"])



inputData = inputData.rename(columns = {"Country_Region" : "Place"})



# len(inputData['Province_State'])

# len(inputData['Place'])

for i in range(inputData.shape[0]):

    if str(inputData['Province_State'][i]) != "nan":

        inputData['Place'][i] = inputData['Province_State'][i] + ", " + inputData["Place"][i]





places = inputData.Place.unique()

inDates = inputData.Date.unique()



nplaces = len(places)

nInDates = len(inDates)



caseData = {}

fatalityData = {}



for date in inDates:

    caseData[date] = []

    fatalityData[date] = []

    

a = 0

b = nInDates

for i in range(313):    

    for j in range(a, b):

        caseData[inputData['Date'][j]].append(inputData['ConfirmedCases'][j])

        fatalityData[inputData['Date'][j]].append(inputData['Fatalities'][j])   

    a+=nInDates

    b+=nInDates





casesdf = pd.DataFrame(caseData, index = [i for i in range(nplaces)])

fataldf = pd.DataFrame(fatalityData, index = [i for i in range(nplaces)])
outputData = pd.read_csv(filepath['test'])

outputDates = outputData.Date.unique()



beginOutDate = outputDates[0]

endOutDate = outputDates[-1]

endInDate = inDates[-1]



outputDates = outputDates[np.where(outputDates == endInDate)[0][0]+1:]

outputDates
mins = []

maxes = []



def norm(series):

    if np.min(series.values) != np.max(series.values):

        return np.min(series.values), np.max(series.values), (series-np.min(series.values))/(np.max(series.values)-np.min(series.values))

    return 0,1,series



def invnorm(x, mini, maxi):

    return x*(maxi-mini) + mini
def custom_loss(actual, predicted):

    return kb.maximum(0.0, kb.abs(actual-predicted)+kb.log(actual/predicted))



def get_model():

    model = keras.Sequential([

    keras.layers.Dense(64, activation='relu', input_shape=[nInDates-1]),

    keras.layers.Dense(32, activation='relu'),

    keras.layers.Dense(1)

    ])



    optimizer = tf.keras.optimizers.RMSprop(0.001)



    model.compile(loss=custom_loss, optimizer = optimizer)

    

    return model
normed_casesdf = casesdf.copy()



for i in range(313):

    mini, maxi, normed = norm(casesdf.loc[i,:])

    mins.append(mini)

    maxes.append(maxi)

    for j in range(nInDates):

        normed_casesdf[inDates[j]][i] = normed[j]



normed_casesdf
epochs = [(i+1)*10 for i in range(10)]



cvscores = []



for epochsize in epochs:    

    totalloss = 0

    

    print(epochsize)

    

    for i in range(10):

        model = get_model()

        

        validation_df = normed_casesdf.iloc[(31*i):(31*(i+1))]

        training_df = normed_casesdf.copy()

        training_df.drop(training_df.index[(31*i):(31*(i+1))])

        

        model.fit(training_df[training_df.columns[:-1]], training_df[training_df.columns[-1]], epochs = epochsize, verbose=0)

        

        loss = model.evaluate(validation_df[validation_df.columns[:-1]], validation_df[validation_df.columns[-1]])

        

        totalloss+=loss

        

    

    avg = totalloss/10

    cvscores.append(avg)
cvscores
optimalepochsize = epochs[cvscores.index(min(cvscores))]
model = get_model()



normed_train_X = normed_casesdf[normed_casesdf.columns[:-1]]

normed_train_y = normed_casesdf[normed_casesdf.columns[-1]]



model.fit(normed_train_X, normed_train_y, epochs = optimalepochsize)



for i in range(len(outputDates)):

    normed_X = normed_casesdf[normed_casesdf.columns[i+1:]]

    normed_predictions = model.predict(normed_X).reshape(nplaces)

      

    for j in range(nplaces):

        normed_predictions[j] = max(normed_predictions[j], normed_casesdf[normed_casesdf.columns[-1]][j])    #ensure non-decreasing predictions

 

    unnormed_predictions = [invnorm(normed_predictions[j], mins[j], maxes[j]) for j in range(nplaces)]

    

    casesdf[outputDates[i]] = unnormed_predictions              #add predictions directly to dataframe which stores all confirmed cases values

    normed_casesdf[outputDates[i]] = normed_predictions
mins = []

maxes = []



normed_fataldf = fataldf.copy()



for i in range(313):

    mini, maxi, normed = norm(fataldf.loc[i,:])

    mins.append(mini)

    maxes.append(maxi)

    for j in range(nInDates):

        normed_fataldf[inDates[j]][i] = normed[j]



normed_fataldf
model2 = get_model()



normed_train_X = normed_fataldf[inDates[:-1]]

normed_train_y = normed_fataldf[inDates[-1]]



model2.fit(normed_train_X, normed_train_y, epochs = optimalepochsize)



for i in range(len(outputDates)):

    normed_train_X = normed_fataldf[normed_fataldf.columns[i+1:]]

    normed_predictions = model.predict(normed_train_X).reshape(nplaces)

    

    unnormed_predictions = []

    

    for j in range(nplaces):

        normed_predictions[j] = max(normed_predictions[j], normed_fataldf[normed_fataldf.columns[-1]][j])

        

        unnormed_prediction = invnorm(normed_predictions[j], mins[j], maxes[j])      #different from confirmed cases code

        if unnormed_prediction == 0:                                                 #ensures fatalities do not stay at 0 by considering cases

            unnormed_prediction = fataldf[fataldf.columns[-14]][j]*.05

        unnormed_predictions.append(unnormed_prediction)

    

    fataldf[outputDates[i]] = unnormed_predictions

    normed_fataldf[outputDates[i]] = normed_predictions
casesdf
fataldf
inDatesTail = inDates[np.where(inDates == beginOutDate)[0][0]:]

submissionDates = [i for i in inDatesTail] + [i for i in outputDates]
caseSub = casesdf[submissionDates]

fatalSub = fataldf[submissionDates]



caseSub
submissiondf = pd.read_csv(filepath['submission'])
for place in range(nplaces):

    for date in range(len(submissionDates)):

        submissiondf['ConfirmedCases'][place*len(submissionDates) + date] = caseSub[submissionDates[date]][place]

        submissiondf['Fatalities'][place*len(submissionDates) + date] = fatalSub[submissionDates[date]][place]



submissiondf.set_index(['ForecastId'])



submissiondf
submissiondf.to_csv('submission.csv', index=False)