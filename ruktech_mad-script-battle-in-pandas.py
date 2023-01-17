#this is the as is Pandas version of the Mad Scripts Battle by ZFTurbo:
#https://www.kaggle.com/zfturbo/facebook-v-predicting-check-ins/mad-scripts-battle/code
#further improvements coming

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Any results you write to the current directory are saved as output.
trainDF = pd.read_csv('../input/train.csv')
testDF = pd.read_csv('../input/test.csv')
trainDF['x'] = np.floor(trainDF['x'].values * 500.0/10.0)
trainDF['y'] = np.floor(trainDF['y'].values * 1000.0/10.0)
trainDF['time'] = np.floor(((trainDF['time'].astype(float)+180.0)/360.0) % 4)

testDF['x'] = np.floor(testDF['x'].values * 500.0/10.0)
testDF['y'] = np.floor(testDF['y'].values * 1000.0/10.0)
testDF['time'] = np.floor(((testDF['time'].astype(float)+180.0)/360.0) % 4)
trainDF['identifier'] = trainDF['x'].astype(str) + ' ' + trainDF['y'].astype(str) + ' ' + trainDF['time'].astype(str) 
testDF['identifier']  = testDF['x'].astype(str)  + ' ' + testDF['y'].astype(str)  + ' ' + testDF['time'].astype(str) 
from collections import Counter
def threeMostCommon(listt):
    return (' ').join([str(val[0]) for val in Counter(listt).most_common(3)])
pivotedTrain = trainDF.pivot_table('place_id','identifier', aggfunc = {'place_id':threeMostCommon}).reset_index()

out = pd.merge(testDF[['row_id','identifier']], pivotedTrain, how = 'left')
out.sort('row_id', inplace = True)

out.drop(['identifier'],axis = 1).fillna(0).to_csv('output.csv',index = False)
