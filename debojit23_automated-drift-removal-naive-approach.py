# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import statsmodels.api as sm

from statsmodels.tsa.seasonal import seasonal_decompose

test = pd.read_csv('../input/liverpool-ion-switching/test.csv')

train = pd.read_csv('../input/liverpool-ion-switching/train.csv')



def get_offset(a,b):

  if (a)>0 and b<0:

    if b<0-a:

      offset=b

    else:

      offset=0-a

  elif a>0 and b>0:

    if b>a:

      offset=0-b

    else:

      offset=0-a

  elif a<0 and b>0:

    if b<0-a:

      offset=b

    else:

      offset=0-a

  elif a<0 and b<0:

    offset=a+b

  return offset







def get_samples(input_batch,single_batch_size,roll_period):

  test_drift=input_batch;roll_period=np.int(roll_period)

  result = seasonal_decompose(input_batch, model='additive',freq=1)

  trend_rolling=result.trend.rolling(roll_period).mean()

  trend_rolling=trend_rolling.fillna(trend_rolling.min())

  c=trend_rolling.values

  data_without_drift=pd.Series(c-np.array(input_batch))

  c=pd.Series(c)

  xx=data_without_drift.rolling(roll_period).mean().min()

  a=xx-c.min();b=input_batch.mean()

  return a,b,c,data_without_drift



##################

def remove_redundant(sub_index,batch_index,relax_rate=5):

    sub2=[];

    #print("sub_index:",sub_index)

    #print("sub2 :",sub2)

    for i in range(0,len(sub_index)-1):

        if sub_index[i+1]-sub_index[i]>relax_rate*num_record_per_sec:

            sub2.append(sub_index[i])

    sub_index=sub2;sub2=[]

    for i in range(0,len(sub_index)):

        c=0

        for j in range(len(batch_index)):

            if abs(sub_index[i]-batch_index[j])<relax_rate*num_record_per_sec:

                c+=1

        if c==0:

            sub2.append(sub_index[i])

    sub2.extend(batch_index)

    sub2 = list(dict.fromkeys(sub2))

    sub2=np.sort(sub2)

    return list(sub2)

def get_subindex2sample(df,istrain,batch_index,batch_index1,order=2,relax_rate=5):

  sub_index=[];count=0;

  while(order!=0):

    #print(order)

    for i in range(len(batch_index)-1):

      train_series=pd.Series(df.iloc[batch_index[i]:batch_index[i+1]].signal);idx=[]

      a,b,c,data_without_drift_init=get_samples(train_series,batch_index[i+1]-batch_index[i],np.round((num_record_per_sec/single_batch_size)*(batch_index[i+1]-batch_index[i])))

      offset=get_offset(a,b)



      data_without_drift_init=offset-data_without_drift_init

      f=data_without_drift_init.rolling(num_record_per_sec).mean()

      f=f.fillna(f.mean())

      g=np.array(c)

      idx = np.argwhere(np.diff(np.sign(f - g))).flatten()

      if count==0:

        idx=np.insert(idx,len(idx),single_batch_size);idx=np.insert(idx,0,0)

      idx=[val+batch_index[i] for val in idx]

      sub_index.extend(idx)

    sub_index=remove_redundant(sub_index,batch_index1,relax_rate)

    #sub_index=np.sort(sub_index)

    batch_index=sub_index

    order-=1;count+=1

  if sub_index[-1]!=df.shape[0]:

    sub_index[-1]=df.shape[0] 

  if sub_index[0]!=0:

    sub_index[0]=0 

  return sub_index

#vary relax_rate to get better results

single_batch_size=500000;num_record_per_sec=10000;relax_rate=5



for df in [train,test]:

  num_of_batch=np.int(df.shape[0]/single_batch_size)

  count=pd.Series()

  batch_index=[val*single_batch_size for val in range(num_of_batch+1)]

  sampling_idx= get_subindex2sample(df,True,batch_index,batch_index,order=2,relax_rate=relax_rate)

  print(sampling_idx[0])

  sampling_idx=np.sort(sampling_idx)

  for i in range(len(sampling_idx)-1):

    train_series=pd.Series(df.iloc[sampling_idx[i]:sampling_idx[i+1]].signal)

    a,b,c,data_without_drift_init=get_samples(train_series,sampling_idx[i+1]-sampling_idx[i],np.round((num_record_per_sec/single_batch_size)*(sampling_idx[i+1]-sampling_idx[i])))

    offset=get_offset(a,b)

    data_without_drift_init=offset-data_without_drift_init

    count=count.append(data_without_drift_init)

  count=list(count)

  print(len(count))

  df['new_sig']=count

plt.figure(figsize=(30,8))



train.signal.plot(color='r')

train.new_sig.plot()

plt.figure(figsize=(30,8))



test.signal.plot(color='r')

test.new_sig.plot()

train[train.signal>0]
test[test.signal<0]