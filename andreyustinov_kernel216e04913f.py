# This Python 3 environment comes with many helpful analytics libraries installed

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
import catboost

from catboost import CatBoostClassifier, Pool

import pandas as pd

import numpy as np

import datetime as dt
df = pd.read_csv('/kaggle/input/hsemath2020flights/flights_train.csv')

df['DATE'] = pd.to_datetime(df['DATE'])

df['dep_delayed_15min'] = pd.get_dummies(df['dep_delayed_15min'],drop_first = True)

df['date_month'] = df['DATE'].dt.month

df['dep_how'] = (df['DEPARTURE_TIME']//100).astype(int)

df['week_of_year'] = df['DATE'].dt.week

df['sum'] = df['DEPARTURE_TIME']+df['DISTANCE']

df['prod'] = df['DEPARTURE_TIME']*df['DISTANCE']

df['sq'] =  df['DEPARTURE_TIME']**2+df['DISTANCE']

df['gen'] = (df['sum']//200).astype(int)

df['DIST'] = df['DISTANCE']

df['DEPARTURE_TIME'] = df['DEPARTURE_TIME'].astype(int)

df['DATE'] = (df['DATE'].astype('str'))



q = pd.DataFrame(df.groupby('ORIGIN_AIRPORT')['DATE'].count())

q['AIRPORT'] = q.index

q = q.sort_values(by='DATE', ascending=False)

array_airport = q['DATE'][:20].index

def big_airport(x):

    if x in array_airport:

        return(x)

    else:

        return('No')



df['Big_air'] = df['ORIGIN_AIRPORT'].apply(big_airport)

df['half2'] = ((df['DEPARTURE_TIME']//100)*60+df['DEPARTURE_TIME']%100)//20

df['half4'] = ((df['DEPARTURE_TIME']//100)*60+df['DEPARTURE_TIME']%100)//17

df['half'] = ((df['DEPARTURE_TIME']//100)*60+df['DEPARTURE_TIME']%100)//15

df['half3'] = ((df['DEPARTURE_TIME']//100)*60+df['DEPARTURE_TIME']%100)//12

df['half0'] = ((df['DEPARTURE_TIME']//100)*60+df['DEPARTURE_TIME']%100)//30

Y = np.array(df['dep_delayed_15min'])

df = df.drop('dep_delayed_15min', axis = 1)

X = np.array(df)
data = pd.read_csv('/kaggle/input/hsemath2020flights/flights_test.csv')

data['DATE'] = pd.to_datetime(data['DATE'])

data['date_month'] = data['DATE'].dt.month

data['dep_how'] = (data['DEPARTURE_TIME']//100).astype(int)

data['week_of_year'] = data['DATE'].dt.week

data['sum'] = data['DEPARTURE_TIME']+data['DISTANCE']

data['prod'] = data['DEPARTURE_TIME']*data['DISTANCE']

data['sq'] =  data['DEPARTURE_TIME']**2+data['DISTANCE']

data['gen'] = (data['sum']//200).astype(int)

data['DIST'] = data['DISTANCE']

data['DEPARTURE_TIME'] = data['DEPARTURE_TIME'].astype(int)

data['DATE'] = (data['DATE'].astype('str'))

data['Big_air'] = data['ORIGIN_AIRPORT'].apply(big_airport)

data['half2'] = ((data['DEPARTURE_TIME']//100)*60+data['DEPARTURE_TIME']%100)//20

data['half4'] = ((data['DEPARTURE_TIME']//100)*60+data['DEPARTURE_TIME']%100)//17

data['half'] = ((data['DEPARTURE_TIME']//100)*60+data['DEPARTURE_TIME']%100)//15

data['half3'] = ((data['DEPARTURE_TIME']//100)*60+data['DEPARTURE_TIME']%100)//12

data['half0'] = ((data['DEPARTURE_TIME']//100)*60+data['DEPARTURE_TIME']%100)//30

X_test_true = np.array(data)
cat_features = [0,1,2,3,4,6,7,8,12,13,14,15,16,17,18,19]

model = CatBoostClassifier(iterations=1543, depth=8, loss_function = 'Logloss',learning_rate=0.1,l2_leaf_reg=6)

eval_dataset = Pool(data, cat_features=cat_features)

model.fit(X, Y, cat_features, eval_set=eval_dataset)

probs = model.predict_proba(X_test_true)
probabilities = pd.DataFrame(probs.T[1]).round(2)

probabilities.columns = ['dep_delayed_15min']

probabilities.to_csv('submit.csv', index_label='id')