import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))
# Read the data file
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
train_df.head()
test_df.head()
from surprise import SVD
from surprise import Reader, Dataset
#Set Rating Scale
reader = Reader(rating_scale=(-5,5))

#Load data with rating scale
train_data = Dataset.load_from_df(train_df[['Viewers_ID', 'Response_ID', 'Rating']], reader)

#Retrieve the trainset
trainset = train_data.build_full_trainset()
#Build an algorithm, and train it.
algo = SVD()
algo.fit(trainset)
#Making first recommendations
pred = [algo.predict(a,b)[3] for a,b in zip (test_df['Viewers_ID'], test_df['Response_ID'])]
def mergefit (pred):
    test = test_df
    train = train_df
    test['Rating'] = pred
    
    #merge test and train
    data = pd.concat([train, test])
    train_data = Dataset.load_from_df(data[['Viewers_ID', 'Response_ID', 'Rating']], reader)
    trainset = train_data.build_full_trainset()
    algo = SVD()
    algo.fit(trainset)
    pred = [algo.predict(a,b)[3] for a,b in zip (test_df['Viewers_ID'], test_df['Response_ID'])]
    return pred
for i in range(20):
    mergefit(pred)
submit = pd.DataFrame({'Response_ID':test_df['Response_ID'],'Rating':pred})
submit.to_csv('pred_20.csv',index=False)
