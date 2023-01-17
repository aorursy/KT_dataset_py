

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import kagglegym

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
with pd.HDFStore("../input/train.h5", "r") as train:

    # Note that the "train" dataframe is the only dataframe in the file

    df = train.get("train")
env = kagglegym.make()

observation = env.reset()
observation.train.head()
list(observation.train.columns.values)
observation.features.head()
target = observation.train["y"].values
def count_missing(df):

    stats={}

    for x in range(len(df.columns)):        

        stats[df.columns[x]]=df[df.columns[x]].isnull().sum()/len(df[df.columns[x]])*100

    return stats
res=count_missing(df)
low_err={k:v for (k,v) in res.items() if v<0.5}

del low_err['id']

del low_err['timestamp']

del low_err['y']

print((low_err),low_err.values())
observation.train = observation.train.interpolate()
observation.train


features = observation.train[list(low_err.keys())]

features = features.fillna(0)
features.shape
features = features.values
from sklearn import datasets, linear_model
regr = linear_model.LinearRegression()
my_regression = regr.fit(features, target)
print(my_regression.score(features, target))
observation.features.head()
observation.target.head()
test_features = observation.features[list(low_err.keys())].values
prediction = my_regression.predict(test_features)
prediction
action = observation.target
action['y'] = prediction
observation, reward, done, info = env.step(action)
reward
done
info
while True: 

    observation.features = observation.features.interpolate()

    test_features = observation.features[list(low_err.keys())].values

    prediction = my_regression.predict(test_features)

    action = observation.target

    action['y'] = prediction

    

    timestamp = observation.features["timestamp"][0]

    

    if timestamp % 100 == 0:

        print("Timestamp #{}".format(timestamp))

        print(prediction)



    observation, reward, done, info = env.step(action)

    if done:        

        break
info["public_score"]