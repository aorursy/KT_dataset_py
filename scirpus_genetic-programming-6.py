import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import log_loss
def Outputs(p):

    return 1.0/(1.0+np.exp(-p))



def GPLow(data):

    p = (np.tanh((((2.275860 + (((2.275860 + (19.666700 + ((data["num_photos"] * 2.0) * 2.0))) + data["bedrooms"]) * data["price"]))/2.0) * 2.0)) +

         np.tanh((((((((data["price"] * 2.0) * 2.0) * 2.0) - ((data["num_features"] + data["bedrooms"])/2.0)) * 2.0) + (data["num_photos"] * data["num_photos"]))/2.0)) +

         np.tanh(((((data["price"] + data["latitude"]) * 2.0) * 2.0) - (((data["created_hour"] * 2.0) + ((((data["num_description_words"] + data["bedrooms"])/2.0) + data["bedrooms"])/2.0))/2.0))) +

         np.tanh((((0.220930 + ((data["created_hour"] + ((-((data["created_hour"] / 2.0))) * (data["created_hour"] / 2.0)))/2.0))/2.0) * (data["num_features"] + data["created_hour"]))) +

         np.tanh((((data["bathrooms"] - (data["bedrooms"] * 2.0)) * data["price"]) + (data["latitude"] * ((data["price"] + data["latitude"]) * 31.0)))) +

         np.tanh(((data["longitude"] * (data["longitude"] + ((5.0) + (3.750000 * (data["bedrooms"] * 2.0))))) + (data["price"] + data["latitude"]))) +

         np.tanh(((0.065574 * (data["num_description_words"] * (((((data["created_hour"] * 2.0) + data["num_features"])/2.0) + (data["num_description_words"] - 1.022220))/2.0))) * 2.0)) +

         np.tanh(((((data["price"] - (data["bathrooms"] / 2.0)) / 2.0) + (((0.065574 + data["price"]) + data["longitude"]) * 2.0)) + data["longitude"])))

    return Outputs(p)



def GPMedium(data):

    p = (np.tanh(((data["num_features"] + ((data["num_features"] * (3.857140 - data["num_features"])) - (data["created_day"] + 10.0)))/2.0)) +

         np.tanh((((((data["num_photos"] + data["created_hour"])/2.0) + ((-(data["latitude"])) - (data["num_photos"] * data["num_photos"])))/2.0) - ((data["price"] * 2.0) * 2.0))) +

         np.tanh((((((((data["bedrooms"] + data["num_features"])/2.0) + data["num_features"])/2.0) + (data["bedrooms"] - 0.591837))/2.0) - (((data["price"] * 2.0) * 2.0) * 2.0))) +

         np.tanh((((data["price"] * (((7.0) + ((-(data["bathrooms"])) * (data["price"] * (7.0))))/2.0)) * data["bedrooms"]) - 0.220930)) +

         np.tanh(((((0.090909 - data["latitude"]) * (((data["latitude"] + data["num_description_words"])/2.0) + data["bedrooms"])) - data["latitude"]) - data["price"])) +

         np.tanh((((data["latitude"] * (data["latitude"] * ((-((10.0 + data["bathrooms"]))) * 2.0))) - data["longitude"]) - (data["longitude"] * 2.0))) +

         np.tanh((0.090909 * ((data["created_hour"] + 1.169230) + ((-((data["num_features"] + data["created_hour"]))) * (data["created_hour"] + data["num_description_words"]))))) +

         np.tanh((((data["num_photos"] + (((-(data["num_photos"])) / 2.0) * (data["num_photos"] / 2.0)))/2.0) * ((-1.0 + (data["num_photos"] * data["num_photos"]))/2.0))))

    return Outputs(p)



def GPHigh(data):

    p = (np.tanh((19.666700 * (-1.0 - ((data["price"] * (((((19.666700 + data["created_hour"])/2.0) * 2.0) + 5.764710)/2.0)) / 2.0)))) +

         np.tanh((-((((1.514290 + (((((5.764710 * 2.0) * 2.0) + data["bedrooms"])/2.0) * data["price"])) * 2.0) * 2.0)))) +

         np.tanh(((((-((data["price"] * 31.0))) + (-(3.071430))) + (data["bedrooms"] + (data["created_hour"] + data["bedrooms"])))/2.0)) +

         np.tanh((((data["num_features"] / 2.0) / 2.0) - (data["price"] + ((data["price"] * 2.0) + (((data["num_photos"] * data["num_photos"]) + 0.090909)/2.0))))) +

         np.tanh((((data["latitude"] * 2.0) + data["price"]) * ((data["created_hour"] - (1.653850 + (31.0 * data["latitude"]))) - 1.653850))) +

         np.tanh((((((((0.090909 + data["price"])/2.0) * 2.0) * 2.0) * ((-(1.362070)) + (data["bedrooms"] * 2.0))) - data["longitude"]) * 2.0)) +

         np.tanh((((((-(data["price"])) + (((-1.0 + data["num_photos"])/2.0) - (data["longitude"] * 19.666700)))/2.0) - data["price"]) - data["price"])) +

         np.tanh((data["latitude"] + (((-(((data["created_day"] + ((((data["created_hour"] + data["bedrooms"])/2.0) * data["bedrooms"]) - data["bedrooms"]))/2.0))) / 2.0) / 2.0))))

    return Outputs(p)
train = pd.read_json('../input/train.json')

test = pd.read_json('../input/test.json')

train.loc[train.interest_level=='low','interest_level'] = 0

train.loc[train.interest_level=='medium','interest_level'] = 1

train.loc[train.interest_level=='high','interest_level'] = 2

train.interest_level = train.interest_level.astype(float)

train["created"] = pd.to_datetime(train["created"])

test["created"] = pd.to_datetime(test["created"])

train["num_photos"] = train["photos"].apply(len)

test["num_photos"] = test["photos"].apply(len)

train["num_features"] = train["features"].apply(len)

test["num_features"] = test["features"].apply(len)

train["created_year"] = train["created"].dt.year

test["created_year"] = test["created"].dt.year

train["created_month"] = train["created"].dt.month

test["created_month"] = test["created"].dt.month

train["created_day"] = train["created"].dt.day

test["created_day"] = test["created"].dt.day

train["created_hour"] = train["created"].dt.hour

test["created_hour"] = test["created"].dt.hour

train["num_description_words"] = train["description"].apply(lambda x: len(x.split(" ")))

test["num_description_words"] = test["description"].apply(lambda x: len(x.split(" ")))
actualcolumns = ['bathrooms', 'bedrooms',

                 'num_photos', 'num_features', 'num_description_words',

                 'latitude','longitude',

                 'price',

                 'created_year',

                 'created_month',

                 'created_day',

                 'created_hour']

classes = ['low','medium','high']

ss = StandardScaler()

ss.fit(pd.concat([train[actualcolumns],test[actualcolumns]]))

predictions  = np.zeros((train.shape[0],3))

gptrain = train[actualcolumns].copy()

gptrain[actualcolumns] = ss.transform(train[actualcolumns])

for i in range(3):

    if(i==0):

        predictions[:,0] = GPLow(gptrain)

    elif(i==1):

        predictions[:,1] = GPMedium(gptrain)

    else:

        predictions[:,2] = GPHigh(gptrain)
print('Log Loss', log_loss(train.interest_level,predictions))
predictions  = np.zeros((test.shape[0],3))

gptest = test[actualcolumns].copy()

gptest[actualcolumns] = ss.transform(test[actualcolumns])

for i in range(3):

    if(i==0):

        predictions[:,0] = GPLow(gptest)

    elif(i==1):

        predictions[:,1] = GPMedium(gptest)

    else:

        predictions[:,2] = GPHigh(gptest)
print('Started producing Submission File')

out_df = pd.DataFrame(predictions)

out_df.columns = ["low", "medium", "high" ]

out_df[["low", "medium", "high" ]] = out_df[["low", "medium", "high" ]].div(out_df[["low", "medium", "high" ]].sum(axis=1), axis=0)

out_df["listing_id"] = test.listing_id.values

out_df = out_df[['high', 'medium', 'low','listing_id']]

out_df.to_csv("loo_xgb_starter.csv", index=False)

print('Finished producing Submission File')