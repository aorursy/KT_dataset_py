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
import pandas as pd

import numpy as np



from keras.models import Sequential, Model

from keras.layers import Embedding, Input, Dense, Activation, concatenate, Flatten, Reshape, Concatenate



from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.model_selection import train_test_split
train_data = pd.read_csv("../input/train.csv", sep=',',header=0)

state_data = pd.read_csv("../input/store_states.csv", sep=',', header=0)
print(f"\nTrain Data - Shape \n{train_data.shape}")

print(f"\nStore States Data - Shape \n{state_data.shape}")
print("---------------------------------------------------------------------\n")

print(f"Train Data - Top 5 Records\n\n{train_data.head()}\n")

print("---------------------------------------------------------------------\n")

print(f"\nStores States Data - Top 5 Records\n\n{state_data.head()}\n")

print("---------------------------------------------------------------------\n")
print(np.unique(train_data['Store']).size)

print(np.unique(state_data['Store']).size)
print(f"\nTrain Data - Summary \n\n{train_data.describe(include = 'all')}")

print(f"\n\nStore States Data - Summary \n\n{state_data.describe(include = 'all')}")
print("\nTrain Data - Data Types \n{}".format(train_data.dtypes))

print("\nStore States Data - Data Types \n{}".format(state_data.dtypes))
train_data['Date'] = pd.to_datetime(train_data['Date'], format='%Y-%m-%d')
print(f"\nTrain Data - Missing values \n{train_data.isnull().sum()}")

print(f"\nStore States Data - Missing value \n{state_data.isnull().sum()}")
train_data['year'] = train_data['Date'].dt.year

train_data['month'] = train_data['Date'].dt.month

train_data['day'] = train_data['Date'].dt.day
train_data.head(5)
print(train_data.shape)

train_data = train_data[train_data['Sales']!=0]

print(train_data.shape)
print(np.unique(train_data['Store']).size)

print(np.unique(state_data['Store']).size)
train_data = pd.merge(train_data, state_data, on='Store', how='inner')
train_data.head(5)
train_data.shape
# #cleansed_data = pd.read_csv('cleansed-train.csv')

# cat_cols = ["State"]

# train_data = pd.get_dummies(train_data,columns=cat_cols,drop_first=True,)
train_data_X = train_data[['Store','DayOfWeek','Promo','year', 'month', 'day', ]]
train_data_y = train_data['Sales']
print(f"The shape of train_data_X is {train_data_X.shape}")

print(f"The shape of train_data_y is {train_data_y.shape}")
for i in [ 'DayOfWeek', 'Promo', 'year', 'month', 'day', 'Store']:

    print("{} has : {} unique values".format(i, np.size(np.unique(train_data_X[i]))))
train_data_X.dtypes
for col in ['Store', 'DayOfWeek', 'Promo', 'year', 'month', 'day']:

    train_data_X[col] = train_data_X[col].astype('category')
train_data_X.dtypes
max_log_y = np.max(np.log(train_data_y))

max_log_y
temp = train_data_y[:1][0]

log_temp = np.log(temp)

tran_temp = log_temp/max_log_y

inv_tran_temp = tran_temp * max_log_y

org_temp = np.exp(inv_tran_temp)



print("Actual Sales values              :{}".format(temp))

print("Log of Actual Sales values       :{}".format(log_temp))

print("Transformed Sales values         :{}".format(tran_temp))

print("Inverse Transformed Sales values :{}".format(org_temp))
# Normalizing the sales by dividing with maximum of sales. Default base of log function is e.

def val_for_fit(val):

    val = np.log(val)/max_log_y

    return val



# Denormalizing the predicted values back to original scale by multiplying with max and taking exponential

def val_for_pred(val):

    return np.exp(val * max_log_y)
train_data_X.head()
train_data_X_LE = train_data_X.apply(LabelEncoder().fit_transform)
train_data_X_LE.head()
enc = OneHotEncoder(handle_unknown='ignore')
train_data_X_OHE = enc.fit_transform(train_data_X_LE)
train_data_X_OHE = enc.transform(train_data_X)
train_data_X_OHE.shape
X_train_CE, X_val_CE, X_train_OHE, X_val_OHE, y_train, y_val = train_test_split(train_data_X_LE.values, 

                                                                                train_data_X_OHE, 

                                                                                train_data_y.values, 

                                                                                test_size=0.1, 

                                                                                random_state=123)
type(X_train_CE)
model1 = Sequential()

model1.add(Dense(1000, kernel_initializer="uniform", input_dim=1182, activation='relu'))

model1.add(Dense(500, kernel_initializer="uniform", activation='relu'))

model1.add(Dense(1, activation='sigmoid'))



model1.compile(loss='mean_absolute_error', optimizer='adam')
model1.fit(X_train_OHE, val_for_fit(y_train), 

           validation_data=(X_val_OHE, val_for_fit(y_val)),

           epochs=100, batch_size=128)
y_pred_val = model1.predict(X_val_OHE).flatten()

y_pred_val 