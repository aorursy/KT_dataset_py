# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

# print(check_output(["ls", "../input"]).decode("utf8"))

df = pd.read_csv("..//input//train.tsv", sep="\t")

df_test = pd.read_csv("..//input//test.tsv", sep="\t")

df_sample = pd.read_csv("..//input//sample_submission.csv", sep=",")

# Any results you write to the current directory are saved as output.



df.head()

df['main_category'], df['sub_category'], df['nested_category'] = df['category_name'].str.split('/', 2).str

df_test['main_category'], df_test['sub_category'], df_test['nested_category'] = df_test['category_name'].str.split('/', 2).str

df.head()
filtered = df[['price', 'item_condition_id', 'shipping', 'main_category', 'sub_category', 'nested_category', 'brand_name', 'name']]

filtered_test = df_test[['test_id','item_condition_id', 'shipping', 'main_category', 'sub_category', 'nested_category', 'brand_name', 'name']]

filtered.head()
filtered = filtered.apply(lambda x: x.astype(str).str.lower())

filtered_test = filtered_test.apply(lambda x: x.astype(str).str.lower())

filtered.head()
# sampled = filtered.sample(frac=0.3)

sampled = filtered

sampled.head()
encoded = pd.get_dummies(sampled, columns=['brand_name', 'main_category', 'sub_category', 'nested_category'])

encoded.head()
columns = ['main_category', 'brand_name', 'sub_category', 'nested_category', 'name']

factorized = sampled[columns].apply(lambda x: pd.factorize(x)[0])

factorized['price'] = sampled['price']

factorized['shipping'] = sampled['shipping']

factorized['item_condition_id'] = sampled['item_condition_id']





factorized_test = filtered_test[columns].apply(lambda x: pd.factorize(x)[0])

factorized_test['shipping'] = filtered_test['shipping']

factorized_test['item_condition_id'] = filtered_test['item_condition_id']

factorized_test['test_id'] = filtered_test['test_id']



factorized = factorized.apply(lambda x: x.astype('category'))

factorized_test = factorized_test.apply(lambda x: x.astype('category'))

factorized['price'] = factorized['price'].apply(pd.to_numeric)

factorized.head()

 
#Spliting the sampled data into training and testing

factorized['is_train'] = np.random.uniform(0, 1, len(factorized)) <= .75

# Create two new dataframes, one with the training rows, one with the test rows

train, test = factorized[factorized['is_train']==True], factorized[factorized['is_train']==False]



# Show the number of observations for the test and training dataframes

print('Number of observations in the training data:', len(train))

print('Number of observations in the test data:',len(test))
from sklearn.ensemble import RandomForestRegressor

from sklearn import linear_model



# regr = linear_model.Ridge (alpha = .9)

# regr = linear_model.LinearRegression()

regr = RandomForestRegressor(max_depth=30, random_state=0)

collist = train.columns.tolist()

collist.remove('price')

collist.remove('is_train')

# collist

regr.fit(train[collist], train['price'])

import math

from decimal import Decimal

def rmsle(y_pred, y):

    assert len(y) == len(y_pred)

    to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]

    return (sum(to_sum) * (1.0/len(y))) ** 0.5
# regr.score(test[collist], test['price'], sample_weight=None)

from sklearn.metrics import mean_squared_log_error







# mean_squared_log_error(test['price'], regr.predict(test[collist]))  

print(rmsle(list(regr.predict(test[collist])), list(test['price'])))

# factorized_test['price'] = regr.predict(factorized_test[collist])

# submission = factorized_test[['test_id', 'price']]

# submission.to_csv('final.csv', index=False)

# submission





# test.head()
import math

from decimal import Decimal

from keras import backend as K



def rmsle_k(y_pred, y):

#     assert len(y) == len(y_pred)

    to_sum = [(K.log(y_pred[i] + 1) - K.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]

    return (K.sum(to_sum) * (1.0/len(y))) ** 0.5
from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=4, mode='auto')



# train = (train - train.mean()) / (train.max() - train.min())

# test = (test - test.mean()) / (test.max() - test.min())



model = Sequential()

model.add(Dense(units=64, activation='relu', input_dim=len(collist)))

model.add(Dense(units=64, activation='relu'))

model.add(Dense(units=64, activation='relu'))

model.add(Dense(units=64, activation='relu'))

model.add(Dense(units=64, activation='relu'))

model.add(Dense(activation='relu', output_dim=1))



model.compile(optimizer='rmsprop',loss='mean_squared_logarithmic_error')

# model.fit(train[collist], train['price'],epochs=10, batch_size=128)

# print(rmsle(list(model.predict(test[collist])), list(test['price'])))

df['item_description']