import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
%matplotlib inline
from plotly import tools
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
from sklearn import model_selection, preprocessing, metrics
import lightgbm as lgb
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from plotly import tools
init_notebook_mode(connected=True)
%%time
df = pd.read_csv('../input/ga-allsessions-sorted.csv', dtype={'fullVisitorId': 'str'})

numeric_cols = ["totals_hits", "totals_pageviews", "visitNumber", "visitStartTime", 'totals_bounces',  'totals_newVisits', 'totals_timeOnSite', 'totals_transactions', 'totals_transactionRevenue']    
for col in numeric_cols:
    df[col] = df[col].astype(float).fillna(0)
df["totals_transactionRevenue"] = df["totals_transactionRevenue"].astype('float') / 10**6
df['date'] = df['date'].astype(str)
df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
df["month"]   = df['date'].dt.month
df["day"]     = df['date'].dt.day
df["weekday"] = df['date'].dt.weekday
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils import resample

df_copy = df.copy()
cat_cols = ["channelGrouping", "device_browser", 
            "device_deviceCategory", "device_operatingSystem",
            "geoNetwork_country", "trafficSource_medium", "trafficSource_source"]
numeric_cols = ["totals_hits", "totals_pageviews", "visitNumber", "visitStartTime", 'totals_bounces',  'totals_newVisits', 'totals_timeOnSite']    

for col in cat_cols:
    lbl = LabelEncoder()
    lbl.fit(list(df_copy[col].values.astype('str')))
    df_copy[col] = lbl.transform(list(df_copy[col].values.astype('str')))
    
for col in numeric_cols:
    scaler = preprocessing.MinMaxScaler()
    df_copy[col] = scaler.fit_transform(df_copy[col].values.reshape(-1,1))
    


df_majority = df_copy[df.totals_transactionRevenue == 0]
df_minority = df_copy[df.totals_transactionRevenue > 0 ]
 
# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=len(df_majority)) # reproducible results
 
# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
 
x = df_upsampled[cat_cols + numeric_cols].fillna(0).values
y = df_upsampled['totals_transactionRevenue'].fillna(0).values
y = y > 0
y = y.astype(float)

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.33, random_state=42)
train_y.sum()
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
model = Sequential()
model.add(Dense(128, input_dim=train_x.shape[1], activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.summary()
history = model.fit(train_x, train_y,validation_split=0.2, verbose=1, epochs=15, batch_size=64)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
scores = model.evaluate(test_x, test_y)
print("Accuracy: %d%%" % (scores[1]*100))
