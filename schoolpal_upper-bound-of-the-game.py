import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#Copied from Github, forgot the author  (sry!)

def rmsle(preds,actuals):

    return np.sqrt(np.sum((np.log(np.array(preds)+1) - np.log(np.array(actuals)+1))**2 / \

len(preds)))



train=pd.read_csv('../input/train.csv',parse_dates=['timestamp'])

# As suggested by radar, we have much fewer low prices entries in 2015, so we only use 2015's data.

train=train[train.timestamp.dt.year==2015]



gold=train.price_doc.copy()

# we make very good guess on the tax-purpose prices (0.2 percentile). 

# In practice, our model can easily precit prices on the 0.5 percentile

guessed_low_price=train[(train.product_type=='Investment') & (train.price_doc>1e6) & (train.price_doc!=2e6) & (train.price_doc!=3e6)].price_doc.quantile(0.2)

print('Somewhat optimal guess:',guessed_low_price)

predicted=train.price_doc.copy()

# we will be 100% correct on normal entries

predicted[(train.product_type=='Investment') & ((train.price_doc<=1e6) | (train.price_doc==2e6) | (train.price_doc==3e6))]=guessed_low_price

print('LB score:',rmsle(predicted.values,gold.values))


