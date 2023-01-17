# Disclaimer: I was just trying out the notebooks on Kaggle, and I ended up creating this undeletable public script. 
# There's not much value in here, sorry for this. 
%matplotlib inline

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from pylab import rcParams

rcParams['figure.figsize'] = 15, 7
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
labels = train['TARGET']
train.drop(['TARGET'], axis=1, inplace=True)
train['source']= 0
test['source'] = 1
data=pd.concat([train, test], ignore_index=True)
plt.plot(data.loc[(labels==0).values].loc[data.source == 0, 'saldo_var30'], label="Satisfied")
plt.plot(data.loc[(labels==1).values].loc[data.source == 0, 'saldo_var30'], label="Unsatisfied")
plt.legend()
plt.ylabel('saldo_var30')
plt.xlabel('Data points')

plt.plot(data.loc[(data.saldo_var30 < 3000) & (labels == 0)].loc[data.source == 0, 'saldo_var30'], label="Satisfied")
plt.plot(data.loc[(data.saldo_var30 < 3000) & (labels == 1)].loc[data.source == 0, 'saldo_var30'], label="Unsatisfied")
plt.legend()
plt.ylabel('saldo_var30')
plt.xlabel('Data points')