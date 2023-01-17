!pip install ProgressBar


import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import glob 

from progressbar import ProgressBar
train_frag = glob.glob("../input/predict-volcanic-eruptions-ingv-oe/test/*")



train_means=[]



pbar = ProgressBar()

for i in pbar(train_frag):

    train_means = np.append(train_means,pd.read_csv(i).mean().max())
'''

train_frag = glob.glob("../input/predict-volcanic-eruptions-ingv-oe/train/*")



data_tot =pd.read_csv(train_frag[0])

data_tot.columns = [['s1_0','s2_0','s3_0','s4_0','s5_0','s6_0','s7_0','s8_0','s9_0','s10_0']]

pbar = ProgressBar()

j = 0

for i in pbar(train_frag[1:]):

    j+=1

    data = pd.read_csv(i)

    data.columns = [['s1_{}'.format(j), 's2_{}'.format(j), 's3_{}'.format(j), 's4_{}'.format(j), 's5_{}'.format(j), 's6_{}'.format(j),

       's7_{}'.format(j), 's8_{}'.format(j), 's9_{}'.format(j), 's10_{}'.format(j)]]

    data_tot = pd.concat([data_tot,data],axis=1)

'''
train = train_means.copy()

sig_tr=[]

for i in range(0,len(train_frag)):

    begin = train_frag[i].find('train/')+6

    end = train_frag[i].find('.csv', begin)

    sig_tr = np.append(sig_tr,train_frag[i][begin:end])
tomerge = pd.DataFrame({'mean': train , 'segment_id': sig_tr})

tomerge['segment_id'] = tomerge['segment_id'].astype(int) 
train_df = pd.read_csv('../input/predict-volcanic-eruptions-ingv-oe/train.csv')
train_df = pd.merge(train_df,tomerge, on = ['segment_id'])
y = train_df['time_to_eruption']

x = train_df['mean']
from sklearn import  linear_model



regr = linear_model.LinearRegression()

regr.fit(np.array(x).reshape(-1,1), y)
test_frag = glob.glob("../input/predict-volcanic-eruptions-ingv-oe/test/*")



test_means=[]



pbar = ProgressBar()

for i in pbar(test_frag):

    test_means = np.append(test_means,pd.read_csv(i).mean().max())
test = test_means.copy()

sig_ts=[]

for i in range(0,len(test_frag)):

    begin = test_frag[i].find('test/')+5

    end = test_frag[i].find('.csv', begin)

    sig_ts = np.append(sig_ts,test_frag[i][begin:end])
sample_submission=pd.read_csv('../input/predict-volcanic-eruptions-ingv-oe/sample_submission.csv')

tomerge = pd.DataFrame({'mean': test , 'segment_id': sig_ts})

tomerge['segment_id'] = tomerge['segment_id'].astype(int) 

submission = pd.merge(sample_submission,tomerge, on = ['segment_id'])
submission['time_to_eruption'] = regr.predict(np.array(submission['mean']).reshape(-1,1))
sample_submission = submission.drop(columns = ['mean'])

sample_submission.to_csv('submission.csv',index=False)