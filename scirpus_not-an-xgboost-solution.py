import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.metrics import r2_score

%matplotlib inline
#Four log loss models based on

# (train.y<=90)

# ((train.y>90)&(train.y<=100))

# ((train.y>100)&(train.y<=110))

# (train.y>110)



def Output(p):

    return 1./(1.+np.exp(-p))



def GP1(data):

    v = pd.DataFrame()

    v["i0"] = np.tanh(((-(data["X162_mean"])) - 3.141593))

    v["i1"] = np.tanh(((((5.0) - (8.0)) - data["X246_mean"]) - data["X328_mean"]))

    v["i2"] = np.tanh((np.cos((-(np.sin(data["X159_mean"])))) - (data["X54_mean"] + 3.141593)))

    v["i3"] = np.tanh((((-(data["X29_mean"])) - data["X272_mean"]) - (4.19060230255126953)))

    v["i4"] = np.tanh((((-((2.0 + data["X29_mean"]))) - (-(np.cos(data["X166_mean"])))) * 2.0))

    v["i5"] = np.tanh((-((np.cos((-(np.tanh(np.cos(data["X265_mean"]))))) - (-(data["X54_mean"]))))))

    v["i6"] = np.tanh(((-(np.cos(np.cos(np.cos((-(data["X310_mean"]))))))) - data["X29_mean"]))

    v["i7"] = np.tanh((-(np.cos((-(data["X187_mean"]))))))

    v["i8"] = np.tanh((data["X218_mean"] * data["X88_mean"]))

    v["i9"] = np.tanh(np.tanh((-((data["X265_mean"] + data["X91_mean"])))))

    return Output(v.sum(axis=1))



def GP2(data):

    v = pd.DataFrame()

    v["i0"] = np.tanh(((data["X127_mean"] - data["X127_mean"]) - (data["X127_mean"] * 2.0)))

    v["i1"] = np.tanh((((-(data["X127_mean"])) - (data["X127_mean"] * 2.0)) - data["X127_mean"]))

    v["i2"] = np.tanh(((((data["X127_mean"] - data["X127_mean"]) - data["X127_mean"]) - data["X127_mean"]) - data["X127_mean"]))

    v["i3"] = np.tanh((((data["X127_mean"] - data["X127_mean"]) - data["X127_mean"]) - data["X127_mean"]))

    v["i4"] = np.tanh(((((-(data["X313_mean"])) - data["X127_mean"]) - data["X313_mean"]) - data["X313_mean"]))

    v["i5"] = np.tanh((data["X21_mean"] - (data["X127_mean"] + (data["X127_mean"] + data["X47_mean"]))))

    v["i6"] = np.tanh((data["X168_mean"] + (((data["X369_mean"] + data["X285_mean"]) + data["X285_mean"]) - data["X47_mean"])))

    v["i7"] = np.tanh(((data["X231_mean"] - (data["X47_mean"] + data["X47_mean"])) - data["X316_mean"]))

    v["i8"] = np.tanh((data["X101_mean"] + ((data["X61_mean"] - data["X23_mean"]) + (data["X115_mean"] - data["X47_mean"]))))

    v["i9"] = np.tanh((((data["X184_mean"] / 2.0) - ((data["X261_mean"] > np.tanh(data["X237_mean"])).astype(float))) - data["X47_mean"]))

    return Output(v.sum(axis=1))



def GP3(data):

    v = pd.DataFrame()

    v["i0"] = np.tanh(((data["X79_mean"] - data["X376_mean"]) - 3.141593))

    v["i1"] = np.tanh((((data["X261_mean"] - np.cos(data["X314_mean"])) - np.cos(data["X82_mean"])) - data["X118_mean"]))

    v["i2"] = np.tanh((data["X314_mean"] - (np.cos((data["X118_mean"] - data["X314_mean"])) + np.cos(data["X171_mean"]))))

    v["i3"] = np.tanh((((data["X314_mean"] - np.cos(data["X63_mean"])) - data["X118_mean"]) * 2.0))

    v["i4"] = np.tanh((((data["X313_mean"] - data["X5_mean"]) - data["X115_mean"]) + (data["X246_mean"] - data["X118_mean"])))

    v["i5"] = np.tanh((((data["X47_mean"] + data["X314_mean"]) + (data["X315_mean"] + data["X47_mean"])) + data["X255_mean"]))

    v["i6"] = np.tanh(((data["X53_mean"] + (data["X47_mean"] - data["X198_mean"])) + data["X340_mean"]))

    v["i7"] = np.tanh((((data["X370_mean"] - (data["X118_mean"] - data["X47_mean"])) / 2.0) - data["X231_mean"]))

    v["i8"] = np.tanh(((((data["X359_mean"] - data["X5_mean"]) * data["X314_mean"]) / 2.0) - data["X231_mean"]))

    v["i9"] = np.tanh((data["X240_mean"] + (((data["X313_mean"] > data["X0_mean"]).astype(float)) - ((data["X186_mean"] > data["X313_mean"]).astype(float)))))

    return Output(v.sum(axis=1))



def GP4(data):

    v = pd.DataFrame()

    v["i0"] = np.tanh(((data["X0_mean"] - np.cos((data["X0_mean"] * data["X0_mean"]))) * 2.0))

    v["i1"] = np.tanh((((data["X0_mean"] - np.cos(data["X0_mean"])) * 2.0) * 2.0))

    v["i2"] = np.tanh(((data["X0_mean"] - np.cos((data["X0_mean"] - data["X0_mean"]))) * 2.0))

    v["i3"] = np.tanh((data["X0_mean"] - ((data["X112_mean"] < np.cos(data["X0_mean"])).astype(float))))

    v["i4"] = np.tanh((((data["X5_mean"] + data["X187_mean"]) + (data["X0_mean"] + data["X5_mean"])) + data["X73_mean"]))

    v["i5"] = np.tanh((data["X363_mean"] - ((data["X118_mean"] < ((data["X261_mean"] < ((data["X261_mean"] < data["X261_mean"]).astype(float))).astype(float))).astype(float))))

    v["i6"] = np.tanh((data["X315_mean"] + ((data["X0_mean"] - np.cos(data["X315_mean"])) - data["X255_mean"])))

    v["i7"] = np.tanh((((data["X241_mean"] - data["X340_mean"]) + (data["X61_mean"] - data["X63_mean"])) + data["X70_mean"]))

    v["i8"] = np.tanh(((data["X115_mean"] + data["X115_mean"]) * (data["X0_mean"] + data["X5_mean"])))

    v["i9"] = np.tanh(((((data["X315_mean"] + data["X151_mean"]) + data["X78_mean"]) / 2.0) - data["X78_mean"]))

    return Output(v.sum(axis=1))
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

features = train.columns[2:]
for c in features:

    stats = train.groupby([c])['y'].mean().reset_index().rename(columns={c: c,'y': c+'_mean'})

    x = test.merge(stats, on=c, how='left')

    test[c+'_mean'] = x[c+'_mean']

    test.drop(c,inplace=True,axis=1)

for c in features:

    stats = train.groupby([c])['y'].mean().reset_index().rename(columns={c: c, 'y': c+'_mean'})

    x = train.merge(stats, on=c, how='left')

    train[c+'_mean'] = x[c+'_mean']

    train.drop(c,inplace=True,axis=1)
features = train.columns[2:]

test[features] = test[features].fillna(train[features].mean())

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

ss.fit(pd.concat([train[features],test[features]]))

train[features]= ss.transform(train[features])
z = np.zeros((train.shape[0],4))

z[:,0] = GP1(train)

z[:,1] = GP2(train)

z[:,2] = GP3(train)

z[:,3] = GP4(train)

z = pd.DataFrame(z)

# Choose your own cut values

cuts = [79.21462314023944,94.18905617655874,104.38137315295712,112.57027583344716]

preds = z.apply(lambda r: cuts[r.argmax()],axis=1)
plt.scatter(train.ID,preds)
print(r2_score(train.y,preds))