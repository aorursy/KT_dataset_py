# read data

import numpy as np

import pandas as pd

train=pd.read_csv('../input/train.csv')

test =pd.read_csv('../input/test.csv')
# select binary features

bin_cols = [col for col in train.columns if '_bin' in col]

# just to test with non binary features...

cat_cols = [col for col in train.columns if '_cat' in col]
import warnings

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin



class BinToCat(BaseEstimator, TransformerMixin):

    def __init__(self):

        pass

    def fit(self, X, y=None, **kwargs):

        cols=X.columns

        if(len(cols)>64):

            warnings.warn("Caution, more than 64 bin columns, 2**64 can overflow int64")

        for i in cols:

            unique_vals=X[i].unique()

            if(len(unique_vals)>2):

                raise Exception("Column "+i+" have more than 2 values, is it binary? values: "+str(unique_vals))

            if not (0 in unique_vals and 1 in unique_vals):

                raise Exception("Column "+i+" have values different from 0/1, is it binary? values: "+str(unique_vals))

        self.scale=np.array([1<<i for i in range(np.shape(X)[1])])

        

    def transform(self, X):

        return np.sum(self.scale*X,axis=1)

        

        
a=BinToCat()

a.fit(train[bin_cols])

t=train[0:3]

print('scale',a.scale)

print('bin    :',t[bin_cols])

print('bin2cat:',a.transform(t[bin_cols]))
train['bins']=a.transform(train[bin_cols])

test['bins'] =a.transform(test[bin_cols])

print('unique length: ',len(train['bins'].unique()))

print(train['bins'])
train.to_csv('train.withoutbin.csv',index=False)

test.to_csv('test.withoutbin.csv',index=False)
a=BinToCat()

a.fit(train[cat_cols])

a=BinToCat()

a.fit(train)
