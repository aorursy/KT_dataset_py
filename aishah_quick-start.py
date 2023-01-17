%pylab inline

%matplotlib inline



import numpy as np

import pylab as pl

import pandas as pd



# Change this to the path for your data directory

DATA_DIR='../input/'
# Training data

TR0 = pd.read_csv(DATA_DIR+'train.csv')

# Testing data

TS0 = pd.read_csv(DATA_DIR+'test.csv')
TR0.head()
# What's the size of the data set?

TS0.shape,TR0.shape
# What's the distribution of loss ? 

x=pl.hist(np.log10(TR0.loss),20)
# Data are categorical and ordinal: merge test & train then transform categorical attributes 



j1 = TR0.shape[0]



D0 = pd.concat((TR0.drop(['id','loss'],axis=1),TS0.drop(['id'],axis=1)))



# Categorical attributes all start with 'cat' 

C1 = [i for i in TS0.columns if i.startswith('cat')]



for c in C1:

 D0[c] = D0[c].astype('category').cat.codes
# Now reconstitute the training and testing data

TR1,TS1=D0.iloc[:j1],D0.iloc[j1:]

# Log-transform the loss 

TR1['log_loss']=np.log10(TR0['loss'])

TR1['id'] = TR0['id']

TS1['id'] = TS0['id']

TR1.set_index('id',inplace=True)

TS1.set_index('id',inplace=True)

TR1.shape,TS1.shape
import random

# If you want to subsample some of the data 

#I = np.random.randint(0,TS1.shape[0],100000)

#D1 = TR1.ix[TR1.index[I]]

#X1,Y1=D1.drop('log_loss',axis=1),D1['log_loss']



# Split the data into input,output

X1,Y1=TR1.drop('log_loss',axis=1),TR1['log_loss']

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_squared_error



# Train 

GBR0 = GradientBoostingRegressor(n_estimators=1000,learning_rate=0.05,max_depth=3,loss='lad')

X1,Y1=TR1.drop('log_loss',axis=1),TR1['log_loss']

GBR0.fit(X1,Y1)
# Test on the training data

Y1p = GBR0.predict(X1)
# What's the training performance (since log10 transformed, 10^y)?    

from sklearn.metrics import mean_absolute_error



mean_absolute_error(10**Y1,10**Y1p)

# Ok should really look at cross-validation performance
# Predict the loss for test cases

Y2 = GBR0.predict(TS1)

Y2p = pd.DataFrame(dict(id=TS1.index,loss=10**Y2))

Y2p.head()
# Dump out to csv file for submission

# Y2p.to_csv(DATA_DIR+'pred-v1a.csv',index=False)