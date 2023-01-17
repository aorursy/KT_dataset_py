import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))

from tqdm import tqdm_notebook as tqdm

import matplotlib.pyplot as plt
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
test=test.iloc[real_samples_indexes]

features = [c for c in train.columns if c not in ['ID_code', 'target']]
for x in features[0:10]:

    aux=train.groupby(x).agg({'ID_code':'count','target':'sum'}).reset_index()

    aux['target0']=aux.ID_code-aux.target

    auxt=test.groupby(x).agg({'ID_code':'count'}).reset_index()

    plt.figure(figsize=(20,4))

    plt.plot(auxt[x],auxt.ID_code,label='test')

    plt.plot(aux[x],aux.ID_code,color='red',alpha=0.4,label='train')

    plt.legend()

    plt.title(x)

    plt.show()

    plt.figure(figsize=(20,4))

    plt.plot(aux[x],aux.target0,label='0')

    plt.plot(aux[x],aux.target,color='red',alpha=0.7,label='1')

    plt.legend()

    plt.title(x+'target0/1')

    plt.show()
for x in features[10:20]:

    aux=train.groupby(x).agg({'ID_code':'count','target':'sum'}).reset_index()

    aux['target0']=aux.ID_code-aux.target

    auxt=test.groupby(x).agg({'ID_code':'count'}).reset_index()

    plt.figure(figsize=(20,4))

    plt.plot(auxt[x],auxt.ID_code,label='test')

    plt.plot(aux[x],aux.ID_code,color='red',alpha=0.4,label='train')

    plt.legend()

    plt.title(x)

    plt.show()

    plt.figure(figsize=(20,4))

    plt.plot(aux[x],aux.target0,label='0')

    plt.plot(aux[x],aux.target,color='red',alpha=0.7,label='1')

    plt.legend()

    plt.title(x+'target0/1')

    plt.show()
for x in features[20:30]:

    aux=train.groupby(x).agg({'ID_code':'count','target':'sum'}).reset_index()

    aux['target0']=aux.ID_code-aux.target

    auxt=test.groupby(x).agg({'ID_code':'count'}).reset_index()

    plt.figure(figsize=(20,4))

    plt.plot(auxt[x],auxt.ID_code,label='test')

    plt.plot(aux[x],aux.ID_code,color='red',alpha=0.4,label='train')

    plt.legend()

    plt.title(x)

    plt.show()

    plt.figure(figsize=(20,4))

    plt.plot(aux[x],aux.target0,label='0')

    plt.plot(aux[x],aux.target,color='red',alpha=0.7,label='1')

    plt.legend()

    plt.title(x+'target0/1')

    plt.show()
for x in features[30:40]:

    aux=train.groupby(x).agg({'ID_code':'count','target':'sum'}).reset_index()

    aux['target0']=aux.ID_code-aux.target

    auxt=test.groupby(x).agg({'ID_code':'count'}).reset_index()

    plt.figure(figsize=(20,4))

    plt.plot(auxt[x],auxt.ID_code,label='test')

    plt.plot(aux[x],aux.ID_code,color='red',alpha=0.4,label='train')

    plt.legend()

    plt.title(x)

    plt.show()

    plt.figure(figsize=(20,4))

    plt.plot(aux[x],aux.target0,label='0')

    plt.plot(aux[x],aux.target,color='red',alpha=0.7,label='1')

    plt.legend()

    plt.title(x+'target0/1')

    plt.show()
for x in features[40:50]:

    aux=train.groupby(x).agg({'ID_code':'count','target':'sum'}).reset_index()

    aux['target0']=aux.ID_code-aux.target

    auxt=test.groupby(x).agg({'ID_code':'count'}).reset_index()

    plt.figure(figsize=(20,4))

    plt.plot(auxt[x],auxt.ID_code,label='test')

    plt.plot(aux[x],aux.ID_code,color='red',alpha=0.4,label='train')

    plt.legend()

    plt.title(x)

    plt.show()

    plt.figure(figsize=(20,4))

    plt.plot(aux[x],aux.target0,label='0')

    plt.plot(aux[x],aux.target,color='red',alpha=0.7,label='1')

    plt.legend()

    plt.title(x+'target0/1')

    plt.show()
for x in features[50:60]:

    aux=train.groupby(x).agg({'ID_code':'count','target':'sum'}).reset_index()

    aux['target0']=aux.ID_code-aux.target

    auxt=test.groupby(x).agg({'ID_code':'count'}).reset_index()

    plt.figure(figsize=(20,4))

    plt.plot(auxt[x],auxt.ID_code,label='test')

    plt.plot(aux[x],aux.ID_code,color='red',alpha=0.4,label='train')

    plt.legend()

    plt.title(x)

    plt.show()

    plt.figure(figsize=(20,4))

    plt.plot(aux[x],aux.target0,label='0')

    plt.plot(aux[x],aux.target,color='red',alpha=0.7,label='1')

    plt.legend()

    plt.title(x+'target0/1')

    plt.show()
for x in features[60:70]:

    aux=train.groupby(x).agg({'ID_code':'count','target':'sum'}).reset_index()

    aux['target0']=aux.ID_code-aux.target

    auxt=test.groupby(x).agg({'ID_code':'count'}).reset_index()

    plt.figure(figsize=(20,4))

    plt.plot(auxt[x],auxt.ID_code,label='test')

    plt.plot(aux[x],aux.ID_code,color='red',alpha=0.4,label='train')

    plt.legend()

    plt.title(x)

    plt.show()

    plt.figure(figsize=(20,4))

    plt.plot(aux[x],aux.target0,label='0')

    plt.plot(aux[x],aux.target,color='red',alpha=0.7,label='1')

    plt.legend()

    plt.title(x+'target0/1')

    plt.show()
for x in features[70:80]:

    aux=train.groupby(x).agg({'ID_code':'count','target':'sum'}).reset_index()

    aux['target0']=aux.ID_code-aux.target

    auxt=test.groupby(x).agg({'ID_code':'count'}).reset_index()

    plt.figure(figsize=(20,4))

    plt.plot(auxt[x],auxt.ID_code,label='test')

    plt.plot(aux[x],aux.ID_code,color='red',alpha=0.4,label='train')

    plt.legend()

    plt.title(x)

    plt.show()

    plt.figure(figsize=(20,4))

    plt.plot(aux[x],aux.target0,label='0')

    plt.plot(aux[x],aux.target,color='red',alpha=0.7,label='1')

    plt.legend()

    plt.title(x+'target0/1')

    plt.show()
for x in features[80:90]:

    aux=train.groupby(x).agg({'ID_code':'count','target':'sum'}).reset_index()

    aux['target0']=aux.ID_code-aux.target

    auxt=test.groupby(x).agg({'ID_code':'count'}).reset_index()

    plt.figure(figsize=(20,4))

    plt.plot(auxt[x],auxt.ID_code,label='test')

    plt.plot(aux[x],aux.ID_code,color='red',alpha=0.4,label='train')

    plt.legend()

    plt.title(x)

    plt.show()

    plt.figure(figsize=(20,4))

    plt.plot(aux[x],aux.target0,label='0')

    plt.plot(aux[x],aux.target,color='red',alpha=0.7,label='1')

    plt.legend()

    plt.title(x+'target0/1')

    plt.show()
for x in features[90:100]:

    aux=train.groupby(x).agg({'ID_code':'count','target':'sum'}).reset_index()

    aux['target0']=aux.ID_code-aux.target

    auxt=test.groupby(x).agg({'ID_code':'count'}).reset_index()

    plt.figure(figsize=(20,4))

    plt.plot(auxt[x],auxt.ID_code,label='test')

    plt.plot(aux[x],aux.ID_code,color='red',alpha=0.4,label='train')

    plt.legend()

    plt.title(x)

    plt.show()

    plt.figure(figsize=(20,4))

    plt.plot(aux[x],aux.target0,label='0')

    plt.plot(aux[x],aux.target,color='red',alpha=0.7,label='1')

    plt.legend()

    plt.title(x+'target0/1')

    plt.show()
for x in features[100:110]:

    aux=train.groupby(x).agg({'ID_code':'count','target':'sum'}).reset_index()

    aux['target0']=aux.ID_code-aux.target

    auxt=test.groupby(x).agg({'ID_code':'count'}).reset_index()

    plt.figure(figsize=(20,4))

    plt.plot(auxt[x],auxt.ID_code,label='test')

    plt.plot(aux[x],aux.ID_code,color='red',alpha=0.4,label='train')

    plt.legend()

    plt.title(x)

    plt.show()

    plt.figure(figsize=(20,4))

    plt.plot(aux[x],aux.target0,label='0')

    plt.plot(aux[x],aux.target,color='red',alpha=0.7,label='1')

    plt.legend()

    plt.title(x+'target0/1')

    plt.show()
for x in features[110:120]:

    aux=train.groupby(x).agg({'ID_code':'count','target':'sum'}).reset_index()

    aux['target0']=aux.ID_code-aux.target

    auxt=test.groupby(x).agg({'ID_code':'count'}).reset_index()

    plt.figure(figsize=(20,4))

    plt.plot(auxt[x],auxt.ID_code,label='test')

    plt.plot(aux[x],aux.ID_code,color='red',alpha=0.4,label='train')

    plt.legend()

    plt.title(x)

    plt.show()

    plt.figure(figsize=(20,4))

    plt.plot(aux[x],aux.target0,label='0')

    plt.plot(aux[x],aux.target,color='red',alpha=0.7,label='1')

    plt.legend()

    plt.title(x+'target0/1')

    plt.show()
for x in features[120:130]:

    aux=train.groupby(x).agg({'ID_code':'count','target':'sum'}).reset_index()

    aux['target0']=aux.ID_code-aux.target

    auxt=test.groupby(x).agg({'ID_code':'count'}).reset_index()

    plt.figure(figsize=(20,4))

    plt.plot(auxt[x],auxt.ID_code,label='test')

    plt.plot(aux[x],aux.ID_code,color='red',alpha=0.4,label='train')

    plt.legend()

    plt.title(x)

    plt.show()

    plt.figure(figsize=(20,4))

    plt.plot(aux[x],aux.target0,label='0')

    plt.plot(aux[x],aux.target,color='red',alpha=0.7,label='1')

    plt.legend()

    plt.title(x+'target0/1')

    plt.show()
for x in features[130:140]:

    aux=train.groupby(x).agg({'ID_code':'count','target':'sum'}).reset_index()

    aux['target0']=aux.ID_code-aux.target

    auxt=test.groupby(x).agg({'ID_code':'count'}).reset_index()

    plt.figure(figsize=(20,4))

    plt.plot(auxt[x],auxt.ID_code,label='test')

    plt.plot(aux[x],aux.ID_code,color='red',alpha=0.4,label='train')

    plt.legend()

    plt.title(x)

    plt.show()

    plt.figure(figsize=(20,4))

    plt.plot(aux[x],aux.target0,label='0')

    plt.plot(aux[x],aux.target,color='red',alpha=0.7,label='1')

    plt.legend()

    plt.title(x+'target0/1')

    plt.show()
for x in features[140:150]:

    aux=train.groupby(x).agg({'ID_code':'count','target':'sum'}).reset_index()

    aux['target0']=aux.ID_code-aux.target

    auxt=test.groupby(x).agg({'ID_code':'count'}).reset_index()

    plt.figure(figsize=(20,4))

    plt.plot(auxt[x],auxt.ID_code,label='test')

    plt.plot(aux[x],aux.ID_code,color='red',alpha=0.4,label='train')

    plt.legend()

    plt.title(x)

    plt.show()

    plt.figure(figsize=(20,4))

    plt.plot(aux[x],aux.target0,label='0')

    plt.plot(aux[x],aux.target,color='red',alpha=0.7,label='1')

    plt.legend()

    plt.title(x+'target0/1')

    plt.show()
for x in features[150:160]:

    aux=train.groupby(x).agg({'ID_code':'count','target':'sum'}).reset_index()

    aux['target0']=aux.ID_code-aux.target

    auxt=test.groupby(x).agg({'ID_code':'count'}).reset_index()

    plt.figure(figsize=(20,4))

    plt.plot(auxt[x],auxt.ID_code,label='test')

    plt.plot(aux[x],aux.ID_code,color='red',alpha=0.4,label='train')

    plt.legend()

    plt.title(x)

    plt.show()

    plt.figure(figsize=(20,4))

    plt.plot(aux[x],aux.target0,label='0')

    plt.plot(aux[x],aux.target,color='red',alpha=0.7,label='1')

    plt.legend()

    plt.title(x+'target0/1')

    plt.show()
for x in features[160:170]:

    aux=train.groupby(x).agg({'ID_code':'count','target':'sum'}).reset_index()

    aux['target0']=aux.ID_code-aux.target

    auxt=test.groupby(x).agg({'ID_code':'count'}).reset_index()

    plt.figure(figsize=(20,4))

    plt.plot(auxt[x],auxt.ID_code,label='test')

    plt.plot(aux[x],aux.ID_code,color='red',alpha=0.4,label='train')

    plt.legend()

    plt.title(x)

    plt.show()

    plt.figure(figsize=(20,4))

    plt.plot(aux[x],aux.target0,label='0')

    plt.plot(aux[x],aux.target,color='red',alpha=0.7,label='1')

    plt.legend()

    plt.title(x+'target0/1')

    plt.show()
for x in features[170:180]:

    aux=train.groupby(x).agg({'ID_code':'count','target':'sum'}).reset_index()

    aux['target0']=aux.ID_code-aux.target

    auxt=test.groupby(x).agg({'ID_code':'count'}).reset_index()

    plt.figure(figsize=(20,4))

    plt.plot(auxt[x],auxt.ID_code,label='test')

    plt.plot(aux[x],aux.ID_code,color='red',alpha=0.4,label='train')

    plt.legend()

    plt.title(x)

    plt.show()

    plt.figure(figsize=(20,4))

    plt.plot(aux[x],aux.target0,label='0')

    plt.plot(aux[x],aux.target,color='red',alpha=0.7,label='1')

    plt.legend()

    plt.title(x+'target0/1')

    plt.show()
for x in features[180:190]:

    aux=train.groupby(x).agg({'ID_code':'count','target':'sum'}).reset_index()

    aux['target0']=aux.ID_code-aux.target

    auxt=test.groupby(x).agg({'ID_code':'count'}).reset_index()

    plt.figure(figsize=(20,4))

    plt.plot(auxt[x],auxt.ID_code,label='test')

    plt.plot(aux[x],aux.ID_code,color='red',alpha=0.4,label='train')

    plt.legend()

    plt.title(x)

    plt.show()

    plt.figure(figsize=(20,4))

    plt.plot(aux[x],aux.target0,label='0')

    plt.plot(aux[x],aux.target,color='red',alpha=0.7,label='1')

    plt.legend()

    plt.title(x+'target0/1')

    plt.show()
for x in features[190:]:

    aux=train.groupby(x).agg({'ID_code':'count','target':'sum'}).reset_index()

    aux['target0']=aux.ID_code-aux.target

    auxt=test.groupby(x).agg({'ID_code':'count'}).reset_index()

    plt.figure(figsize=(20,4))

    plt.plot(auxt[x],auxt.ID_code,label='test')

    plt.plot(aux[x],aux.ID_code,color='red',alpha=0.4,label='train')

    plt.legend()

    plt.title(x)

    plt.show()

    plt.figure(figsize=(20,4))

    plt.plot(aux[x],aux.target0,label='0')

    plt.plot(aux[x],aux.target,color='red',alpha=0.7,label='1')

    plt.legend()

    plt.title(x+'target0/1')

    plt.show()