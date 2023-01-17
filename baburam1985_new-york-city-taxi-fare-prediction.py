# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
TRAIN_PATH = '../input/train.csv'
# trainDF =  pd.read_csv('../input/train.csv', chunksize = 100000)
# trainDF = trainDF.get_chunk(100000)
%%time
# Assume we only know that the csv file is somehow large, but not the exact size
# we want to know the exact number of rows

# Method 1, using file.readlines. Takes about 20 seconds.
with open(TRAIN_PATH) as file:
    n_rows = len(file.readlines())

print ('Exact number of rows: '+str(n_rows))
chunksize = 5000000 # 5 million rows at one go. Or try 10 million
total_chunk = n_rows // chunksize + 1

print('Total chunks required:'+str(total_chunk))
traintypes = {'fare_amount': 'float32',
              'pickup_datetime': 'str', 
              'pickup_longitude': 'float32',
              'pickup_latitude': 'float32',
              'dropoff_longitude': 'float32',
              'dropoff_latitude': 'float32',
              'passenger_count': 'uint8'}

cols = list(traintypes.keys())
%%time
df_list = [] # list to hold the batch dataframe
i=0

for df_chunk in pd.read_csv(TRAIN_PATH, usecols=cols, dtype=traintypes, chunksize=chunksize):
    
    i = i+1
    # Each chunk is a corresponding dataframe
     
    
    # Neat trick from https://www.kaggle.com/btyuhas/bayesian-optimization-with-xgboost
    # Using parse_dates would be much slower!
    df_chunk['pickup_datetime'] = df_chunk['pickup_datetime'].str.slice(0, 16)
    df_chunk['pickup_datetime'] = pd.to_datetime(df_chunk['pickup_datetime'], utc=True, format='%Y-%m-%d %H:%M')
    
    # Can process each chunk of dataframe here
    # clean_data(), feature_engineer(),fit()
    
    # Alternatively, append the chunk to list and merge all
    df_list.append(df_chunk) 
trainDF = pd.concat(df_list)
trainDF=trainDF.dropna()
testDF =  pd.read_csv('../input/test.csv')
trainDF.head()
testDF.head()
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

y = trainDF["fare_amount"]
# y_test = testDF["fare_amount"]
y.head()
trainDF.columns
X = trainDF[[  'pickup_longitude',
       'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
       'passenger_count']]
X_pred = testDF[[  'pickup_longitude',
       'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
       'passenger_count']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X.head()

model = LogisticRegression()
model.fit(X,y)
my_model.fit(X_train, y_train, verbose=False)
my_model.score(X_test,y_test)
# from xgboost import XGBRegressor

# my_model = XGBRegressor()
# Add silent=True to avoid printing out updates with each cycle
# my_model.fit(X_train, y_train, verbose=False)
# my_model.score(X_test,y_test)
# testDF1 = testDF[[  'pickup_longitude',
#        'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
#        'passenger_count']]
y_pred = my_model.predict(X_pred)
y_pred.size
submission = pd.DataFrame({
        "key": testDF['key'],
        "fare_amount": y_pred.round(3)
})
submission.to_csv('submit.csv', index = False)

