

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# TODO: code and runtime results



import pandas as pd

train_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')

train_transaction.head()

isFraud = train_transaction.loc[train_transaction['isFraud']==1]

isNotFraud = train_transaction.loc[train_transaction['isFraud']==0]
isFraud.head()
cols = [col for col in isFraud.columns if col in ['TransactionID', 'isFraud', 'TransactionAmt','TransactionDT','ProductCD','card4','card6','P_emaildomain','R_emaildomain','addr1','addr2','dist1','dist2']]

isFraud1 = isFraud[cols]

isFraud1.head()
df1 = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv')

df1.head()
cols = [col for col in df1.columns if col in ['DeviceType','DeviceInfo']]

df2 = df1[cols]

df2.head()
isFraud = pd.concat([df2,isFraud1], sort='False')

isFraud.tail()
import numpy as np

np.log(isFraud['TransactionAmt']).hist(bins=100)
import seaborn as sns

sns.countplot(x='ProductCD', data= isFraud)
sns.countplot(x='card4', data= isFraud)
sns.countplot(x='DeviceType', data= isFraud)
# TODO: code to generate the frequency graph

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(18,4))



#isFraud['TransactionDT'] = pd.to_numeric(isFraud['TransactionDT'], errors='coerce')

isFraud2 =isFraud 

isFraud2 = isFraud2.dropna(subset=['TransactionDT'])



time_val = isFraud2['TransactionDT'].values



sns.distplot(time_val, ax=ax[0], color='r')

ax[0].set_title('Distribution of TransactionDT', fontsize=14)

ax[1].set_xlim([min(time_val), max(time_val)])



sns.distplot(np.log(time_val), ax=ax[1], color='b')

ax[1].set_title('Distribution of LOG TransactionDT', fontsize=14)

ax[1].set_xlim([min(np.log(time_val)), max(np.log(time_val))])



plt.show()
isFraud['Time'] = np.round(isFraud['TransactionDT']/(60*60),0)

isFraud['Time'] = np.round(isFraud['Time']%24,0)

isFraud.loc[isFraud['Time']==10]

l3 = []

l3 = isFraud['Time'].tolist()

plt.bar(isFraud['Time'], isFraud['addr2'], align='center', alpha=0.5)



sns.countplot(x='DaysFromStart', data= isFraud)

isFraud['DaysFromStart'].unique()
# TODO: code to analyze prices for different product codes

isFraud1 = isFraud

isFraud1['Rank'] = isFraud1['TransactionAmt'].rank(ascending=0)

#isFraud1.sort_values('TransactionAmt', inplace=True)

isFraud1["group_rank"] = isFraud1.groupby("ProductCD")["TransactionAmt"].rank(ascending=0,method='dense')

isFraud1.head()
sns.countplot(x='ProductCD', data = isFraud)
df2 = isFraud.groupby(['ProductCD']).median()

df2.head()

df3 = df2['TransactionAmt']

l1= df3.index.tolist()

l2 = []

l2.append(df3[0]) 

l2.append(df3[1]) 

l2.append(df3[2]) 

l2.append(df3[3]) 

l2.append(df3[4])

plt.bar(l1, l2, align='center', alpha=0.5)



#plt.show()
# TODO: code to calculate correlation coefficient

isFraud4 = isFraud

isFraud4 = isFraud4.dropna(subset = ['TransactionDT'])

isFraud4 = isFraud4.dropna(subset = ['TransactionAmt'])

import scipy.stats as sp

spcor = sp.pearsonr(isFraud4['TransactionDT'], isFraud4['TransactionAmt'])

spcor1 = sp.spearmanr(isFraud4['TransactionDT'], isFraud4['TransactionAmt'])

cor = np.corrcoef(isFraud4['TransactionDT'], isFraud4['TransactionAmt'])

#cor

spcor
# TODO: code to generate the plot here.



import missingno as msno

msno.dendrogram(isFraud)
import pandas as pd

test_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv')

cols = [col for col in test_transaction.columns if col in ['TransactionID', 'TransactionAmt','TransactionDT','ProductCD','card4','card6','P_emaildomain','R_emaildomain','addr1','addr2','dist1','dist2']]

test_X = test_transaction[cols]

test_X.head()



from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

import pandas as pd

train_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')

cols = [col for col in test_transaction.columns if col in ['TransactionID', 'TransactionAmt','TransactionDT','ProductCD','card4','card6','P_emaildomain','R_emaildomain','addr1','addr2','dist1','dist2']]

X_train = train_transaction[cols]

X_train.head()
# TODO: code for your final model

import pandas as pd

train_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')

cols = [col for col in train_transaction.columns if col in ['isFraud']]

y_train = train_transaction[cols]

y_train.head()

X_train.fillna(-1, inplace=True)

y_train.fillna(-1, inplace=True)

test_X.fillna(-1, inplace=True)

X_train['card6'] = pd.factorize(X_train['card6'])[0]+1

X_train['card4'] = pd.factorize(X_train['card4'])[0]+1

X_train['P_emaildomain'] = pd.factorize(X_train['P_emaildomain'])[0]+1

X_train['R_emaildomain'] = pd.factorize(X_train['R_emaildomain'])[0]+1

X_train['ProductCD'] = pd.factorize(X_train['ProductCD'])[0]+1

#X_train

test_X['card6'] = pd.factorize(test_X['card6'])[0]+1

test_X['card4'] = pd.factorize(test_X['card4'])[0]+1

test_X['P_emaildomain'] = pd.factorize(test_X['P_emaildomain'])[0]+1

test_X['R_emaildomain'] = pd.factorize(test_X['R_emaildomain'])[0]+1

test_X['ProductCD'] = pd.factorize(test_X['ProductCD'])[0]+1

test_X
df4 = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv')
model.fit(X_train, y_train)

f = model.predict(test_X)
df8=df4['TransactionID']

d = pd.DataFrame(f)

#df8.append(f)

df8 = pd.concat([df8,d],axis=1)