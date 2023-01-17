# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import RobustScaler

from keras.models import Sequential

from keras.layers import Dropout, Activation, Dense, LSTM

from tensorflow.keras.callbacks import EarlyStopping

from keras.preprocessing.sequence import TimeseriesGenerator



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
conf_cases = pd.read_csv('/kaggle/input/ece657aw20asg4coronavirus/time_series_covid19_confirmed_global.csv')

conf_cases.head()
conf_cases_canada = conf_cases[conf_cases["Country/Region"] == "Canada"]

conf_cases_canada
conf_cases_ontario = conf_cases_canada[conf_cases_canada["Province/State"] == "Ontario"]

conf_cases_ontario
conf_cases_ontario_a = pd.DataFrame(conf_cases_ontario[conf_cases_ontario.columns[4:]].sum(),columns=["confirmed"])

conf_cases_ontario_a.index = pd.to_datetime(conf_cases_ontario_a.index,format='%m/%d/%y')

conf_cases_ontario_a
conf = conf_cases_ontario_a[["confirmed"]]

len(conf)

leng = len(conf) - 6

train_data=conf.iloc[:leng]

test_data = conf.iloc[leng:]

sc = RobustScaler()

sc.fit(train_data)

sc_train = sc.transform(train_data)#and divide every point by max value

sc_test = sc.transform(test_data)

test_val = np.append(sc_train[77],sc_test)

test_val = test_val.reshape(7,1)



i = 4

f = 1 

TSG = TimeseriesGenerator(sc_train,sc_train,length = i,batch_size=1)







md = Sequential()

md.add(LSTM(135,activation="relu",input_shape=(i,f)))

md.add(Dense(75, activation='relu'))

md.add(Dense(units=1))

md.compile(optimizer="adam",loss="mse")



val = TimeseriesGenerator(test_val,test_val,length=4,batch_size=1)



hault = EarlyStopping(monitor='val_loss',patience=20,restore_best_weights=True)

md.fit_generator(TSG,validation_data=val,epochs=108,callbacks=[hault],steps_per_epoch=8)
predicted = []

res1 = sc_train[-i:]

live = res1.reshape(1,i,f)



for a in range(len(test_data)+7):

    pre_now = md.predict(live)[0]

    predicted.append(pre_now)

    live = np.append(live[:,1:,:],[[pre_now]],axis=1)

    

actual = sc.inverse_transform(predicted)

actual[:,0]



tm_data = test_data.index

for b in range(0,7):

    tm_data = tm_data.append(tm_data[-1:] + pd.DateOffset(1))

tm_data



fut = pd.DataFrame(columns=["confirmed","predicted cases"],index=tm_data)

fut.loc[:,"predicted cases"] = actual[:,0]

fut.loc[:,"confirmed"] = test_data["confirmed"]

print(fut)



fut.plot(title="Future values (Predictions) for a seven day period")