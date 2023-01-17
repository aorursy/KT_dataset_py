# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import os
import random
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Machine Learning
def OHEncode (df, cols):
    for x in cols:
        x_ohe = pd.get_dummies(df[x], prefix=x)
        df = pd.concat([df, x_ohe], axis=1)
        df = df.drop([x], axis=1)
    return df
# define columns we want to OHE
# cols = ['platform','geo_location','dayofweek','hour']
# cols = ['platform','geo_location','hour']
cols = ['platform']
testing=False
chunksize=50000
clf = LogisticRegression()
train = pd.read_csv("../input/train-featured/clicks_events_full.csv", iterator=True,chunksize=chunksize) #Load data
print('Chunks training')
for chunk in train:
    chunk = chunk.drop(['geo_location','dayofweek','hour'], axis=1)
    chunk = OHEncode(chunk,cols) # Perform OHE
    predictors=[x for x in chunk.columns if x not in ['display_id','clicked']] # Select columns for prediction
    chunk=chunk.fillna(0.0)
    clf.fit(chunk[predictors], chunk["clicked"]) #Fit classifier
    if testing:
        break
train='' #remove train
chunk.head()
test =  pd.read_csv("../input/test-featured/test_events_full.csv",iterator=True,chunksize=chunksize) #Load data
print('Testing')
predY=[]
for chunk in test:
    init_chunk_size=len(chunk)
    chunk = chunk.drop(['geo_location','dayofweek','hour'], axis=1)
    chunk = OHEncode(chunk,cols)
    chunk=chunk.fillna(0.0)
    chunk_pred=list(clf.predict_proba(chunk[predictors]).astype(float)[:,1])
    predY += chunk_pred
    if testing:
        break
print('Done Testing')
print('Preparing for Submission')
test='' #remove test
test= pd.read_csv('../input/outbrain-click-prediction/clicks_test.csv') #load full test
results=pd.concat((test,pd.DataFrame(predY)) ,axis=1,ignore_index=True) #Combine the predicted values with the ids
print(results.head(10))
results.columns = ['display_id','ad_id','clicked']#Rename the columns
print(results.head(10))
results = results.sort_values(by=['display_id','clicked'], ascending=[True, False])
results = results.reset_index(drop=True)
results2 = results.copy()
submission_data = results2.groupby('display_id').ad_id.apply(lambda x: " ".join(map(str,x))).reset_index()
submission_data.head()
submission_data.to_csv('submission_feature_removed.csv', index=False)
#results=results[['display_id','ad_id']].groupby('display_id')['ad_id'].agg(lambda col: ' '.join(map(str,col)))
#results.columns=[['display_id','ad_id']]
#results.to_csv('submission_final.csv')