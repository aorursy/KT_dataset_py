import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from pylab import rcParams
class color:  # Testing to make the heading look a liitle more impressive
   BOLD = '\033[1m'
from sklearn.metrics import mean_squared_error , mean_absolute_error
df = pd.read_csv("../input/demand-forecasting-kernels-only/train.csv")
df.head()
df['date'] =  pd.to_datetime(df['date'])
df = df.set_index('date')
df.head()
df_1_1 = df[(df.store==1) & (df.item==1)] 
Deep1_all = df_1_1.resample('D')['sales'].sum()
#Deep1_all = df.resample('D')['sales'].sum()  # this is of doing the forecast on the total dataset by day
Deep1_all.head()
Deep1_all_With_index = Deep1_all.copy()
Deep1_all_With_index =Deep1_all_With_index.reset_index()
Deep1_all_With_index.head()
Deep1_all_With_index.head()
Deep1 = Deep1_all_With_index.drop(['date'], axis = 1)
Deep1['sales'] = Deep1['sales'].astype('float32')
Deep1.info()
values = Deep1.values
print(values)
values = values.astype('float32')
Deep1.shape
train_size = int(len(Deep1) -376) # This is 366 days of the year + 10 days of extra data beforehand
test_size = len(Deep1) - train_size
train, test = Deep1.iloc[0:train_size], Deep1.iloc[train_size:len(Deep1)]
print(len(train), len(test))
print(test)
print(train)
train.head()
# One of the most difficult parts of Deep Learning modelling is to get the dataset in the right format 
# This function completes that proces

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
        #print(Xs[-1], ys[-1])  
    return np.array(Xs), np.array(ys)
test.shape
# These next few lines are about getting the data ready for modelling
time_steps = 10

# reshape to [samples, time_steps, n_features]
X_train, y_train = create_dataset(train, train.sales, time_steps)

#X_train_c, y_train_c = create_dataset(X_train_c_a, y_train_c_a, time_steps)
X_test, y_test = create_dataset(test, test.sales, time_steps)
print(X_train.shape, y_train.shape)
# Note the 3 dimensional shape
len(X_test)


deep_model = keras.Sequential()
deep_model.add(keras.layers.LSTM(
  units=128,
  input_shape=(X_train.shape[1], X_train.shape[2])
))
deep_model.add(keras.layers.Dense(units=2))
deep_model.add(keras.layers.Dense(units=1))

deep_model.compile(
  loss='mse',
  optimizer=keras.optimizers.Adam(0.001)) # was 0.001
history = deep_model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=16,
    validation_split=0.1,
    verbose=1,
    shuffle=False
)
print(y_train)
y_pred = deep_model.predict(X_test)
print(y_pred)
X_test.shape
print(X_test)
print(y_pred)
print(X_test)
Results =[]
Results = pd.DataFrame( columns=['sales','pred'])
Results['sales'] = test['sales']
Results.head()
y_pred_df = pd.DataFrame(y_pred, columns=['pred'])
Results = Results[10:]  # As the Deep Learning process added the forst 10 dates I had to drop the first 10 rows, so the first result was 1 Jan 2017
y_pred_df.head()
Results= Results.reset_index() 
Results.head(10)
Results ['pred'] = y_pred_df['pred']  
Results = Results.set_index('index')
Results =Results.drop (['sales'],axis=1)
New_Results = pd.concat([Results, Deep1_all_With_index], axis=1)
New_Results.head() ##### GOOD
New_Results.tail() ##### GOOD
Results_with_date_2017 = New_Results[(New_Results.date>'2016-12-31')]
Results_with_date_2017.head()
RMSE_Deep  = np.mean(np.sqrt((Results_with_date_2017['pred'] - Results_with_date_2017['sales']) ** 2)) 
print(RMSE_Deep)
# Note this compares to 4.009 from XG boost for the same data period
_ = Results_with_date_2017[['sales','pred']].plot(figsize=(15, 5))
Results_with_date_Jan_2017 =Results_with_date_2017[(New_Results.date<'2017-02-01')]
_ = Results_with_date_Jan_2017[['sales','pred']].plot(figsize=(15, 5))
