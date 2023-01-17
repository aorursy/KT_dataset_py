import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))

import matplotlib.pyplot as plt





from tqdm import tqdm_notebook as tqdm

df_test = pd.read_csv('../input/test.csv')

df_test.drop(['ID_code'], axis=1, inplace=True)

df_test = df_test.values



unique_samples = []

unique_count = np.zeros_like(df_test)

for feature in tqdm(range(df_test.shape[1])):

    _, index_, count_ = np.unique(df_test[:, feature], return_counts=True, return_index=True)

    unique_count[index_[count_ == 1], feature] += 1



# Samples which have unique values are real the others are fake

real_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) > 0)[:, 0]

synthetic_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) == 0)[:, 0]



print(len(real_samples_indexes))

print(len(synthetic_samples_indexes))
train=pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")
#test=test.iloc[real_samples_indexes]

features = [c for c in train.columns if c not in ['ID_code', 'target']]
for x in features[0:20]:

    aux=train.groupby(x).agg({'ID_code':'count'})

    aux['freq']=aux.ID_code/aux.ID_code.sum()

    del aux['ID_code']

    train = train.merge(aux, on=x,how='left')

    

    auxt=test.groupby(x).agg({'ID_code':'count'})

    auxt['freqt']=auxt.ID_code/auxt.ID_code.sum()

    del auxt['ID_code']

    test = train.merge(auxt, on=x,how='left')

    

    aux1=train.groupby('freq').agg({'ID_code':'count','target':'sum'}).reset_index()

    auxt=test.groupby('freq').agg({'ID_code':'count','target':'sum'}).reset_index()

    aux1['target0']=aux1.ID_code-aux1.target

    aux1['target0']=aux1.target0/aux1.target0.sum()

    aux1['target']=aux1.target/aux1.target.sum()

    auxt['ID_code']=auxt.ID_code/auxt.ID_code.sum()

    aux1['ID_code']=aux1.ID_code/aux1.ID_code.sum()

    

    plt.figure(figsize=(20,4))

    plt.plot(aux1['freq'],aux1.target0,alpha=0.5,label='0')

    plt.plot(aux1['freq'],aux1.target,color='red',alpha=0.5,label='1')

    plt.plot(auxt['freq'],auxt.ID_code,color='green',alpha=0.5,label='test',linestyle="--")

    plt.plot(aux1['freq'],aux1.ID_code,color='black',alpha=0.5,label='train',linestyle="--")

    plt.legend()

    plt.title(x+'target0/1')

    plt.show()

    del train['freq']
for x in features[20:40]:

    aux=train.groupby(x).agg({'ID_code':'count'})

    aux['freq']=aux.ID_code/aux.ID_code.sum()

    del aux['ID_code']

    train = train.merge(aux, on=x,how='left')

    

    auxt=test.groupby(x).agg({'ID_code':'count'})

    auxt['freqt']=auxt.ID_code/auxt.ID_code.sum()

    del auxt['ID_code']

    test = train.merge(auxt, on=x,how='left')

    

    aux1=train.groupby('freq').agg({'ID_code':'count','target':'sum'}).reset_index()

    auxt=test.groupby('freq').agg({'ID_code':'count','target':'sum'}).reset_index()

    aux1['target0']=aux1.ID_code-aux1.target

    aux1['target0']=aux1.target0/aux1.target0.sum()

    aux1['target']=aux1.target/aux1.target.sum()

    auxt['ID_code']=auxt.ID_code/auxt.ID_code.sum()

    aux1['ID_code']=aux1.ID_code/aux1.ID_code.sum()

    

    plt.figure(figsize=(20,4))

    plt.plot(aux1['freq'],aux1.target0,alpha=0.5,label='0')

    plt.plot(aux1['freq'],aux1.target,color='red',alpha=0.5,label='1')

    plt.plot(auxt['freq'],auxt.ID_code,color='green',alpha=0.5,label='test',linestyle="--")

    plt.plot(aux1['freq'],aux1.ID_code,color='black',alpha=0.5,label='train',linestyle="--")

    plt.legend()

    plt.title(x+'target0/1')

    plt.show()

    del train['freq']
for x in features[40:60]:

    aux=train.groupby(x).agg({'ID_code':'count'})

    aux['freq']=aux.ID_code/aux.ID_code.sum()

    del aux['ID_code']

    train = train.merge(aux, on=x,how='left')

    

    auxt=test.groupby(x).agg({'ID_code':'count'})

    auxt['freqt']=auxt.ID_code/auxt.ID_code.sum()

    del auxt['ID_code']

    test = train.merge(auxt, on=x,how='left')

    

    aux1=train.groupby('freq').agg({'ID_code':'count','target':'sum'}).reset_index()

    auxt=test.groupby('freq').agg({'ID_code':'count','target':'sum'}).reset_index()

    aux1['target0']=aux1.ID_code-aux1.target

    aux1['target0']=aux1.target0/aux1.target0.sum()

    aux1['target']=aux1.target/aux1.target.sum()

    auxt['ID_code']=auxt.ID_code/auxt.ID_code.sum()

    aux1['ID_code']=aux1.ID_code/aux1.ID_code.sum()

    

    plt.figure(figsize=(20,4))

    plt.plot(aux1['freq'],aux1.target0,alpha=0.5,label='0')

    plt.plot(aux1['freq'],aux1.target,color='red',alpha=0.5,label='1')

    plt.plot(auxt['freq'],auxt.ID_code,color='green',alpha=0.5,label='test',linestyle="--")

    plt.plot(aux1['freq'],aux1.ID_code,color='black',alpha=0.5,label='train',linestyle="--")

    plt.legend()

    plt.title(x+'target0/1')

    plt.show()

    del train['freq']
for x in features[60:80]:

    aux=train.groupby(x).agg({'ID_code':'count'})

    aux['freq']=aux.ID_code/aux.ID_code.sum()

    del aux['ID_code']

    train = train.merge(aux, on=x,how='left')

    

    auxt=test.groupby(x).agg({'ID_code':'count'})

    auxt['freqt']=auxt.ID_code/auxt.ID_code.sum()

    del auxt['ID_code']

    test = train.merge(auxt, on=x,how='left')

    

    aux1=train.groupby('freq').agg({'ID_code':'count','target':'sum'}).reset_index()

    auxt=test.groupby('freq').agg({'ID_code':'count','target':'sum'}).reset_index()

    aux1['target0']=aux1.ID_code-aux1.target

    aux1['target0']=aux1.target0/aux1.target0.sum()

    aux1['target']=aux1.target/aux1.target.sum()

    auxt['ID_code']=auxt.ID_code/auxt.ID_code.sum()

    aux1['ID_code']=aux1.ID_code/aux1.ID_code.sum()

    

    plt.figure(figsize=(20,4))

    plt.plot(aux1['freq'],aux1.target0,alpha=0.5,label='0')

    plt.plot(aux1['freq'],aux1.target,color='red',alpha=0.5,label='1')

    plt.plot(auxt['freq'],auxt.ID_code,color='green',alpha=0.5,label='test',linestyle="--")

    plt.plot(aux1['freq'],aux1.ID_code,color='black',alpha=0.5,label='train',linestyle="--")

    plt.legend()

    plt.title(x+'target0/1')

    plt.show()

    del train['freq']
for x in features[80:100]:

    aux=train.groupby(x).agg({'ID_code':'count'})

    aux['freq']=aux.ID_code/aux.ID_code.sum()

    del aux['ID_code']

    train = train.merge(aux, on=x,how='left')

    

    aux1=train.groupby('freq').agg({'ID_code':'count','target':'sum'}).reset_index()

    aux1['target0']=aux1.ID_code-aux1.target

    aux1['target0']=aux1.target0/aux1.target0.sum()

    aux1['target']=aux1.target/aux1.target.sum()

    plt.figure(figsize=(20,4))

    plt.plot(aux1['freq'],aux1.target0,label='0')

    plt.plot(aux1['freq'],aux1.target,color='red',alpha=0.7,label='1')

    plt.legend()

    plt.title(x+'target0/1')

    plt.show()

    del train['freq']
for x in features[100:120]:

    aux=train.groupby(x).agg({'ID_code':'count'})

    aux['freq']=aux.ID_code/aux.ID_code.sum()

    del aux['ID_code']

    train = train.merge(aux, on=x,how='left')

    

    auxt=test.groupby(x).agg({'ID_code':'count'})

    auxt['freqt']=auxt.ID_code/auxt.ID_code.sum()

    del auxt['ID_code']

    test = train.merge(auxt, on=x,how='left')

    

    aux1=train.groupby('freq').agg({'ID_code':'count','target':'sum'}).reset_index()

    auxt=test.groupby('freq').agg({'ID_code':'count','target':'sum'}).reset_index()

    aux1['target0']=aux1.ID_code-aux1.target

    aux1['target0']=aux1.target0/aux1.target0.sum()

    aux1['target']=aux1.target/aux1.target.sum()

    auxt['ID_code']=auxt.ID_code/auxt.ID_code.sum()

    aux1['ID_code']=aux1.ID_code/aux1.ID_code.sum()

    

    plt.figure(figsize=(20,4))

    plt.plot(aux1['freq'],aux1.target0,alpha=0.5,label='0')

    plt.plot(aux1['freq'],aux1.target,color='red',alpha=0.5,label='1')

    plt.plot(auxt['freq'],auxt.ID_code,color='green',alpha=0.5,label='test',linestyle="--")

    plt.plot(aux1['freq'],aux1.ID_code,color='black',alpha=0.5,label='train',linestyle="--")

    plt.legend()

    plt.title(x+'target0/1')

    plt.show()

    del train['freq']
for x in features[120:140]:

    aux=train.groupby(x).agg({'ID_code':'count'})

    aux['freq']=aux.ID_code/aux.ID_code.sum()

    del aux['ID_code']

    train = train.merge(aux, on=x,how='left')

    

    auxt=test.groupby(x).agg({'ID_code':'count'})

    auxt['freqt']=auxt.ID_code/auxt.ID_code.sum()

    del auxt['ID_code']

    test = train.merge(auxt, on=x,how='left')

    

    aux1=train.groupby('freq').agg({'ID_code':'count','target':'sum'}).reset_index()

    auxt=test.groupby('freq').agg({'ID_code':'count','target':'sum'}).reset_index()

    aux1['target0']=aux1.ID_code-aux1.target

    aux1['target0']=aux1.target0/aux1.target0.sum()

    aux1['target']=aux1.target/aux1.target.sum()

    auxt['ID_code']=auxt.ID_code/auxt.ID_code.sum()

    aux1['ID_code']=aux1.ID_code/aux1.ID_code.sum()

    

    plt.figure(figsize=(20,4))

    plt.plot(aux1['freq'],aux1.target0,alpha=0.5,label='0')

    plt.plot(aux1['freq'],aux1.target,color='red',alpha=0.5,label='1')

    plt.plot(auxt['freq'],auxt.ID_code,color='green',alpha=0.5,label='test',linestyle="--")

    plt.plot(aux1['freq'],aux1.ID_code,color='black',alpha=0.5,label='train',linestyle="--")

    plt.legend()

    plt.title(x+'target0/1')

    plt.show()

    del train['freq']
for x in features[140:160]:

    aux=train.groupby(x).agg({'ID_code':'count'})

    aux['freq']=aux.ID_code/aux.ID_code.sum()

    del aux['ID_code']

    train = train.merge(aux, on=x,how='left')

    

    auxt=test.groupby(x).agg({'ID_code':'count'})

    auxt['freqt']=auxt.ID_code/auxt.ID_code.sum()

    del auxt['ID_code']

    test = train.merge(auxt, on=x,how='left')

    

    aux1=train.groupby('freq').agg({'ID_code':'count','target':'sum'}).reset_index()

    auxt=test.groupby('freq').agg({'ID_code':'count','target':'sum'}).reset_index()

    aux1['target0']=aux1.ID_code-aux1.target

    aux1['target0']=aux1.target0/aux1.target0.sum()

    aux1['target']=aux1.target/aux1.target.sum()

    auxt['ID_code']=auxt.ID_code/auxt.ID_code.sum()

    aux1['ID_code']=aux1.ID_code/aux1.ID_code.sum()

    

    plt.figure(figsize=(20,4))

    plt.plot(aux1['freq'],aux1.target0,alpha=0.5,label='0')

    plt.plot(aux1['freq'],aux1.target,color='red',alpha=0.5,label='1')

    plt.plot(auxt['freq'],auxt.ID_code,color='green',alpha=0.5,label='test',linestyle="--")

    plt.plot(aux1['freq'],aux1.ID_code,color='black',alpha=0.5,label='train',linestyle="--")

    plt.legend()

    plt.title(x+'target0/1')

    plt.show()

    del train['freq']
for x in features[160:180]:

    aux=train.groupby(x).agg({'ID_code':'count'})

    aux['freq']=aux.ID_code/aux.ID_code.sum()

    del aux['ID_code']

    train = train.merge(aux, on=x,how='left')

    

    auxt=test.groupby(x).agg({'ID_code':'count'})

    auxt['freqt']=auxt.ID_code/auxt.ID_code.sum()

    del auxt['ID_code']

    test = train.merge(auxt, on=x,how='left')

    

    aux1=train.groupby('freq').agg({'ID_code':'count','target':'sum'}).reset_index()

    auxt=test.groupby('freq').agg({'ID_code':'count','target':'sum'}).reset_index()

    aux1['target0']=aux1.ID_code-aux1.target

    aux1['target0']=aux1.target0/aux1.target0.sum()

    aux1['target']=aux1.target/aux1.target.sum()

    auxt['ID_code']=auxt.ID_code/auxt.ID_code.sum()

    aux1['ID_code']=aux1.ID_code/aux1.ID_code.sum()

    

    plt.figure(figsize=(20,4))

    plt.plot(aux1['freq'],aux1.target0,alpha=0.5,label='0')

    plt.plot(aux1['freq'],aux1.target,color='red',alpha=0.5,label='1')

    plt.plot(auxt['freq'],auxt.ID_code,color='green',alpha=0.5,label='test',linestyle="--")

    plt.plot(aux1['freq'],aux1.ID_code,color='black',alpha=0.5,label='train',linestyle="--")

    plt.legend()

    plt.title(x+'target0/1')

    plt.show()

    del train['freq']
for x in features[180:]:

    aux=train.groupby(x).agg({'ID_code':'count'})

    aux['freq']=aux.ID_code/aux.ID_code.sum()

    del aux['ID_code']

    train = train.merge(aux, on=x,how='left')

    

    auxt=test.groupby(x).agg({'ID_code':'count'})

    auxt['freqt']=auxt.ID_code/auxt.ID_code.sum()

    del auxt['ID_code']

    test = train.merge(auxt, on=x,how='left')

    

    aux1=train.groupby('freq').agg({'ID_code':'count','target':'sum'}).reset_index()

    auxt=test.groupby('freq').agg({'ID_code':'count','target':'sum'}).reset_index()

    aux1['target0']=aux1.ID_code-aux1.target

    aux1['target0']=aux1.target0/aux1.target0.sum()

    aux1['target']=aux1.target/aux1.target.sum()

    auxt['ID_code']=auxt.ID_code/auxt.ID_code.sum()

    aux1['ID_code']=aux1.ID_code/aux1.ID_code.sum()

    

    plt.figure(figsize=(20,4))

    plt.plot(aux1['freq'],aux1.target0,alpha=0.5,label='0')

    plt.plot(aux1['freq'],aux1.target,color='red',alpha=0.5,label='1')

    plt.plot(auxt['freq'],auxt.ID_code,color='green',alpha=0.5,label='test',linestyle="--")

    plt.plot(aux1['freq'],aux1.ID_code,color='black',alpha=0.5,label='train',linestyle="--")

    plt.legend()

    plt.title(x+'target0/1')

    plt.show()

    del train['freq']