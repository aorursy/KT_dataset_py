import pandas as pd

import numpy as np

import datetime as dt



import matplotlib.pyplot as plt

import matplotlib.dates as pltdt

%matplotlib inline



import seaborn as sns

sns.set_style("darkgrid")
coun = pd.read_csv("../input/EMHIRESPV_TSh_CF_Country_19862015.csv")

coun.head(3)
coun.shape
t = pd.date_range('1/1/1986', periods = 262968, freq = 'H')
coun["Hour"] = t

coun.set_index("Hour", inplace = True, )

coun['2015-12-31'].plot()

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, ncol = 2,  borderaxespad=0.)
coun['2015-12'].plot()

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, ncol = 2,  borderaxespad=0.)
coun['Day']=coun.index.map(lambda x: x.strftime('%Y-%m-%d'))

c_group_day = coun.groupby('Day').mean()

c_group_day.plot()

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, ncol = 2,  borderaxespad=0.)
coun['Month']=coun.index.map(lambda x: x.strftime('%Y-%m'))

coun['Month_only']=coun.index.map(lambda x: x.strftime('%m'))

c_group_month = coun.groupby('Month').mean()

c_group_month.plot()

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, ncol = 2,  borderaxespad=0.)
coun['Year']=coun.index.map(lambda x: x.strftime('%Y'))

c_group_year = coun.groupby('Year').mean()

c_group_year.plot()

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, ncol = 2,  borderaxespad=0.)
pt_heatmap = coun.pivot_table(index = 'Month_only', columns = 'Year', values = 'PT')

pt_heatmap.sortlevel(level = 0, ascending = True, inplace = True)

sns.heatmap(pt_heatmap, vmin = 0.09, vmax = 0.29, cmap = 'inferno', linewidth = 0.5)
sns.clustermap(pt_heatmap, cmap = 'inferno', standard_scale = 1)
pt_ts = coun.filter(['Month','Year','PT'], axis = 1)

pt_ts.plot()
pt_ts_m = pt_ts.groupby('Month').mean()

pt_ts_m.plot()
pt_ts_y = pt_ts.groupby('Year').mean()

pt_ts_y.plot()
pt_nn = coun.filter(['Hour', 'PT'], axis = 1)



pt_nn = pt_nn.reset_index()

pt_nn['Hour'] = pd.to_datetime(pt_nn['Hour'])



start = pd.Timestamp('2015-12-01')

split = pd.Timestamp('2015-12-22')

pt_nn = pt_nn[pt_nn['Hour']>=start]



pt_nn = pt_nn.set_index('Hour')



pt_nn.plot()
train = pt_nn.loc[:split, ['PT']]

test = pt_nn.loc[split:, ['PT']]

tr_pl = train

te_pl = test

ax = tr_pl.plot()

te_pl.plot(ax=ax)
from sklearn.preprocessing import StandardScaler



sc = StandardScaler()

train_sc = sc.fit_transform(train)

test_sc = sc.transform(test)



X_train = train_sc[:-1]

y_train = X_train[1:]

X_train = X_train[:-1]            # in order for arrays to have same length



X_test = test_sc[:-1]

y_test = X_test[1:]

X_test = X_test[:-1]
train_df = pd.DataFrame(train_sc, columns = ['PT'], index = train.index )

test_df = pd.DataFrame(test_sc, columns = ['PT'], index = test.index )
for s in range(1, 25):

    train_df['shift {}'.format(s)] = train_df['PT'].shift(s, freq = 'H')

    test_df['shift {}'.format(s)] = test_df['PT'].shift(s, freq = 'H')



train_df.head(3)
X_train = train_df.dropna().drop('PT', axis = 1)

y_train = train_df.dropna()[['PT']]



X_test = test_df.dropna().drop('PT', axis = 1)

y_test = test_df.dropna()[['PT']]

X_train.head(3)
X_train.shape
# to np.array

X_train = X_train.values

y_train = y_train.values



X_test = X_test.values

y_test = y_test.values
# Needs to be re-dimensioned for LSTM layer

X_train_w = X_train.reshape(X_train.shape[0], 1, 24)

X_test_w = X_test.reshape(X_test.shape[0], 1, 24)

X_train_w.shape
from keras.models import Sequential

from keras.layers import Dense, LSTM, Flatten

from keras.optimizers import Adam

from keras.callbacks import EarlyStopping

import keras.backend as K
K.clear_session()



eps = 500

bs = 1



in_sh = (1, 24) 

hidden_1= 12

hidden_2= 12

outputs = 1



model = Sequential()

model.add(LSTM(hidden_1, input_shape = in_sh,))

model.add(Dense(hidden_2, activation ='relu'))

model.add(Dense(outputs))

model.compile(optimizer='adam', loss='mean_squared_error',)

model.summary()
early_stop = EarlyStopping(monitor = 'loss', patience = 1, verbose = 1)
model.fit(X_train_w, y_train, epochs = eps, batch_size = bs, verbose = 1 , callbacks = [early_stop])
y_pred = model.predict(X_test_w)
plt.plot(y_test)

plt.plot(y_pred)