import numpy as np 

import pandas as pd



train = pd.read_csv('../input/train.csv')



nulls = train.isnull().sum()



importance = train.shape[0]*[None]

for i in range(train.shape[0]):

    nulls = train.loc(i)[i].isnull().sum()

    imp_nulls = train.loc(i)[i].isnull()[: 12].sum()

    importance[i] = np.exp((1/(nulls+1)*(1/(imp_nulls+1))))/np.exp(1)



#imp_nulls are the nulls directly related to the house

#We decide to use the sqrt to reduce the amplitude. The model performed better in LB and local CV

importance = np.sqrt(importance)

train['price_per_sq'] = np.array(train.price_doc)/np.array(train.full_sq)

train = train[(train.price_per_sq > 60000) & (train.price_per_sq < 500000)]

#The values 60000 and 500000 were defined based on price_per_sq histogram.