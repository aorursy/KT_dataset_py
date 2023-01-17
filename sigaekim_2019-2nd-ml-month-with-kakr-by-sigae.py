import numpy as np 

import pandas as pd #

import os

from os.path import join

import matplotlib.pyplot as plt

import seaborn as sns

import xgboost

import lightgbm 
train_data = join('../input', 'train.csv')

test_data = join('../input', 'test.csv')
train = pd.read_csv(train_data)

test = pd.read_csv(test_data)



train.head()
train_len = len(train)

id_backup = train['id'][:train_len]

del train['id']

        



print(train_len)

tempchk = 0

for i in train.columns:

    for j in range(0, train_len):   

        tempcnt = 0

        if j == 0:

            tempcnt = train[i][j]

        elif tempcnt != train[i][j]:

            tempchk += 1

    if tempchk == 0:

        print(i)

        del train[i]
price_backup = train['price'][:train_len]

del train['price']
train['date'] = train['date'].apply(lambda x : str(x[:8])).astype(str)

train['date'].head()
train.head()
fig, ax = plt.subplots(figsize=(8, 6))

sns.kdeplot(price_backup[:train_len])
fig, ax = plt.subplots(figsize=(8, 6))

log_price = np.log1p(price_backup[:train_len])

sns.kdeplot(log_price)
fig, ax = plt.subplots(10, 2, figsize=(20, 60))



count = 0

columns = train.columns

for row in range(10):

    for col in range(2):

        sns.kdeplot(train[columns[count]], ax=ax[row][col])

        ax[row][col].set_title(columns[count], fontsize=15)

        count+=1

        if count == 19 :

            break


log_columns = ['bedrooms', 'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement']



for i in log_columns:

    train[i] = np.log1p(train[i])

    test[i] = np.log1p(test[i])



fig, ax = plt.subplots(3, 2, figsize=(10, 15))

cnt = 0

for row in range(3):

    for col in range(2):

        if cnt == 5:

            break

        sns.kdeplot(train[log_columns[cnt]], ax=ax[row][col])

        ax[row][col].set_title(log_columns[cnt], fontsize=15)

        cnt+=1
gbm = lightgbm.LGBMRegressor(random_state = 2019)

gbm.fit(train.values, price_backup)



test_len = len(test)

test_id_backup = test['id'][:test_len]

del test['id']

test['date'] = test['date'].apply(lambda x : str(x[:8])).astype(str)


sub = gbm.predict(test.values)

submit_data =  pd.DataFrame(data={'id':test_id_backup,'price':sub})



submit_data.to_csv('submission.csv', index=False)

print("complete!")