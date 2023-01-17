import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.metrics import r2_score

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
features = ['X315', 'X127', 'X47', 'X118',

            'X314', 'X54', 'X29', 'X48',

            'X312', 'X261', 'X316', 'X30',

            'X115', 'X179', 'X13', 'X354',

            'X313', 'X267', 'X231', 'X189',

            'X176']
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

trainy = train.y.ravel()
train = train[features]

test = test[features]

train['y'] = trainy
for c in train.columns:

    if train[c].dtype == 'object':

        lbl = LabelEncoder()

        lbl.fit(list(train[c].values) + list(test[c].values))

        train[c] = lbl.transform(list(train[c].values))

        test[c] = lbl.transform(list(test[c].values))



train = train.fillna(train.median())

test = test.fillna(train.median())

ss = StandardScaler()

ss.fit(pd.concat([(train[features]),(test[features])]))

train[features] = ss.transform((train[features]))

test[features] = ss.transform((test[features]))
def GP1(data):

    p = (6.124150 +

         1.000000*(((((np.maximum( (data["X127"]),  (data["X118"])) + data["X315"])/2.0) + data["X314"])/2.0)) +

         0.983700*((np.maximum( (data["X127"]),  (data["X29"])) * np.minimum( (0.636620),  (data["X47"])))) +

         0.989500*((0.636620 * (data["X54"] * ((data["X47"] < data["X315"]).astype(float))))) +

         1.000000*((((data["X48"] > (data["X47"] * data["X127"])).astype(float)) * 0.318310)) +

         1.000000*((np.minimum( (data["X315"]),  ((3.39852762222290039))) * np.minimum( ((0.53182137012481689)),  (data["X312"])))) +

         0.477700*(((((((data["X118"] + 0.318310)/2.0) / 2.0) / 2.0) / 2.0) / 2.0)))

    return 16.19*p



def GP2(data):

    p = (6.124150 +

         1.000000*(((data["X314"] + ((data["X315"] + np.maximum( (data["X127"]),  (data["X118"])))/2.0))/2.0)) +

         0.965600*((((np.minimum( (data["X314"]),  ((-(data["X29"])))) / 2.0) / 2.0) / 2.0)) +

         0.777500*((((data["X47"] > (-(data["X48"]))).astype(float)) / 2.0)) +

         0.903900*((((np.minimum( (1.570796),  (data["X54"])) / 2.0) / 2.0) * data["X315"])) +

         0.798500*((((data["X312"] > 0.318310).astype(float)) * (0.636620 + data["X354"]))) +

         1.000000*((data["X316"] * ((0.318310 < np.minimum( (data["X179"]),  (data["X13"]))).astype(float)))))

    return 16.19*p



def GP3(data):

    p = (6.124150 +

         1.000000*(((((np.maximum( (data["X127"]),  (data["X118"])) + data["X315"])/2.0) + data["X314"])/2.0)) +

         0.907100*(((((data["X261"] + (data["X261"] * data["X29"]))/2.0) / 2.0) / 2.0)) +

         0.992900*(np.minimum( (0.318310),  (((((data["X47"] + data["X48"])/2.0) / 2.0) / 2.0)))) +

         1.000000*((data["X54"] * (((data["X315"] > (-(data["X30"]))).astype(float)) / 2.0))) +

         1.000000*(((data["X54"] * np.maximum( (data["X313"]),  (data["X312"]))) * data["X267"])) +

         0.944200*(((((-(data["X315"])) < data["X115"]).astype(float)) * (data["X231"] / 2.0))))

    return 16.19*p



def GP4(data):

    p = (6.124150 +

         1.000000*(((data["X314"] + ((np.maximum( (data["X127"]),  (data["X118"])) + data["X315"])/2.0))/2.0)) +

         0.965600*((((np.minimum( (data["X314"]),  ((-(data["X29"])))) / 2.0) / 2.0) / 2.0)) +

         1.000000*(((0.32382854819297791) * ((data["X48"] > (data["X47"] * data["X312"])).astype(float)))) +

         1.000000*((np.minimum( (data["X54"]),  ((2.36594843864440918))) * ((data["X315"] > data["X312"]).astype(float)))) +

         0.995500*(((((data["X47"] * data["X316"]) > 1.0).astype(float)) / 2.0)) +

         1.000000*((((0.07979037612676620) * ((data["X316"] < data["X118"]).astype(float))) * 0.636620)))

    return 16.19*p



def GP5(data):

    p = (6.124150 +

         1.000000*(((((np.maximum( (data["X127"]),  (data["X118"])) + data["X315"])/2.0) + data["X314"])/2.0)) +

         1.000000*((((data["X127"] / 2.0) * (data["X29"] / 2.0)) / 2.0)) +

         1.000000*(np.minimum( (0.318310),  (np.maximum( (data["X48"]),  ((data["X47"] * 0.318310)))))) +

         1.000000*((((((-(data["X30"])) < data["X315"]).astype(float)) * data["X54"]) / 2.0)) +

         0.764000*((((((data["X261"] > data["X115"]).astype(float)) / 2.0) / 2.0) / 2.0)) +

         1.000000*((((-(data["X189"])) / 2.0) * ((data["X315"] > data["X176"]).astype(float)))))

    return 16.19*p



def GP(data):

    return .2*(GP1(data)+GP2(data)+GP3(data)+GP4(data)+GP5(data))
print(r2_score(train.y,GP1(train)))

print(r2_score(train.y,GP2(train)))

print(r2_score(train.y,GP3(train)))

print(r2_score(train.y,GP4(train)))

print(r2_score(train.y,GP5(train)))

print(r2_score(train.y,GP(train)))
plt.scatter(GP(train),train.y)