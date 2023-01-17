import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
train_competition = pd.read_json('../input/stanford-covid-vaccine/train.json', lines=True)

train_competition.head()
reactivity = []

for n in train_competition['reactivity']:

    reactivity+=n

reactivity = pd.Series(reactivity)



reactivity.plot(kind='hist',bins=500)

plt.xlim(-0.2,5)

plt.show()
reactivity.to_frame().describe()
import pandas as pd

second_place = pd.read_csv('../input/covid-result-of-233-sequences/2nd-place-233-seq.csv')

second_place.set_index('id_seqpos',inplace=True)

second_place.columns = ['P2_'+n for n in second_place.columns]

second_place.sort_values(by='P2_reactivity').tail(5)
first_place = pd.read_csv('../input/ov-inference-233-new-seq/submission.csv')

first_place.set_index('id_seqpos',inplace=True)

first_place.columns = ['P1_'+n for n in first_place.columns]

first_place.sort_values(by='P1_reactivity').tail(5)
merge = pd.concat([second_place['P2_reactivity'],first_place['P1_reactivity']],axis=1)

merge.plot(kind='hist',histtype='step',bins=50,figsize=(12,4),density=1)



reactivity[(reactivity>-0.1)&(reactivity<5)].plot(kind='hist',histtype='step',bins=50,density=1,label='Competition train')

plt.xlim(-0.2,2)

plt.legend()

plt.show()
#scatter everything

merge.plot(kind='scatter',x='P2_reactivity',y='P1_reactivity',alpha=0.1,figsize=(12,6))

plt.show()
merge['base']=[int(n.split('_')[-1]) for n in merge.index.values]

#only positions less than 68 where trained

merge[merge['base']<68].plot(kind='scatter',x='P2_reactivity',y='P1_reactivity',alpha=0.1,figsize=(12,6))

plt.show()
merge['base']=[int(n.split('_')[-1]) for n in merge.index.values]

#only positions less than 68 where trained

merge[merge['base']>68].sample(n=merge[merge['base']<68].shape[0]).plot(kind='scatter',x='P2_reactivity',y='P1_reactivity',alpha=0.1,figsize=(12,6))

plt.show()
from sklearn.preprocessing import QuantileTransformer

transformer = QuantileTransformer(n_quantiles=100, random_state=0, output_distribution="normal")
merge[['P2_reactivity','P1_reactivity']] = transformer.fit_transform(merge[['P2_reactivity','P1_reactivity']] )

merge[merge['base']<68].plot(kind='scatter',x='P2_reactivity',y='P1_reactivity',alpha=0.1,figsize=(12,6)) 

plt.show()
merge[['P2_reactivity','P1_reactivity']] = transformer.fit_transform(merge[['P2_reactivity','P1_reactivity']] )

merge[merge['base']>68].sample(n=merge[merge['base']<68].shape[0]).plot(kind='scatter',x='P2_reactivity',y='P1_reactivity',alpha=0.1,figsize=(12,6)) 

plt.show()
def process_df(df,tag=''):

    df.set_index('id_seqpos',inplace=True)

    df.columns = [tag+n for n in df.columns]

    return df
df = pd.read_csv('../input/covid-19-mrna-4th-place-solution/submission_lstm_lstm.csv')

df_4o = process_df(df,tag='P4o_')



df  = pd.read_csv('../input/covid19-cnn-transformer/submission_cnn_0.22935.csv')

df_23o = process_df(df,tag='P23o_')
df_4o.head()
merge = pd.concat([df_4o['P4o_reactivity'],df_23o['P23o_reactivity']],axis=1)

merge.plot(kind='scatter',x='P4o_reactivity',y='P23o_reactivity',alpha=0.1)

merge.head()
merge['base']=[int(n.split('_')[-1]) for n in merge.index.values]

#only positions less tha 68 where trained

merge[merge['base']<68].plot(kind='scatter',x='P4o_reactivity',y='P23o_reactivity',alpha=0.1)