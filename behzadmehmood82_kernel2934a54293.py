# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import numpy as np # linear algebra

from numpy.lib.stride_tricks import as_strided

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas import Series, DataFrame

from matplotlib import pyplot as plt

plt.style.use('fivethirtyeight')

plt.rcParams['figure.figsize'] = [12, 8]

from sklearn.preprocessing import MinMaxScaler

from datetime import timedelta

import datetime
a = pd.read_csv('../input/solar-measurementspakistanlahorewb-esmapqc.csv', index_col='time', parse_dates=['time'])
a['time']=a.index

#a["time"] =  pd.to_datetime(a["time"])

#a['date'] = pd.to_datetime(a['time']).dt.date

a['Month'] = pd.to_datetime(a['time']).dt.month

a['day'] = pd.to_datetime(a['time']).dt.day

a['hour'] = pd.to_datetime(a['time']).dt.hour

a['minute'] = pd.to_datetime(a['time']).dt.minute

x=7

y=19

b=a[a['hour'] >x]

b=b[b['hour'] <y]

c=b[(b['hour']==y-1) & (b['minute']>0)]

index=c.index

for i in index:

    b = b.drop([i], axis=0)

d=b[(b['hour']==x) & (b['minute']<10)]

j=d.index

for i in j:

    b = b.drop([i], axis=0)
del b['comments']

#del b['time']

#del b['sensor_cleaning']

#del b['']

#del b['']

#del b['']

#del b['']

#del b['']
b=b.interpolate(method='time', axis=0, limit=None, inplace=False, limit_direction='forward', limit_area=None, downcast=None)
b.isnull().T.any().T.sum()
def timeseries_to_supervised(data, lag=1):

    df = pd.DataFrame(data)

    columns = [df.shift(i) for i in range(1, lag+1)]

    columns.append(df)

    df = pd.concat(columns, axis=1)

    return df
c=b.iloc[35:,:]
b=c
c=b['ghi_pyr']

d=timeseries_to_supervised(c, lag=60)
d
t1=d.iloc[:,0:1]

t2=d.iloc[:,1:2]

t3=d.iloc[:,2:3]

t4=d.iloc[:,3:4]

t5=d.iloc[:,4:5]

t6=d.iloc[:,5:6]

t7=d.iloc[:,6:7]

t8=d.iloc[:,7:8]

t9=d.iloc[:,8:9]

t10=d.iloc[:,9:10]

t11=d.iloc[:,10:11]

t12=d.iloc[:,11:12]

t13=d.iloc[:,12:13]

t14=d.iloc[:,13:14]

t15=d.iloc[:,14:15]

t16=d.iloc[:,15:16]

t17=d.iloc[:,16:17]

t18=d.iloc[:,17:18]

t19=d.iloc[:,18:19]

t20=d.iloc[:,19:20]

t21=d.iloc[:,20:21]

t22=d.iloc[:,21:22]

t23=d.iloc[:,22:23]

t24=d.iloc[:,23:24]

t25=d.iloc[:,24:25]

t26=d.iloc[:,25:26]

t27=d.iloc[:,26:27]

t28=d.iloc[:,27:28]

t29=d.iloc[:,28:29]

t30=d.iloc[:,29:30]

t31=d.iloc[:,30:31]

t32=d.iloc[:,31:32]

t33=d.iloc[:,32:33]

t34=d.iloc[:,33:34]

t35=d.iloc[:,34:35]

t36=d.iloc[:,35:36]

t37=d.iloc[:,36:37]

t38=d.iloc[:,37:38]

t39=d.iloc[:,38:39]

t40=d.iloc[:,39:40]

t41=d.iloc[:,40:41]

t42=d.iloc[:,41:42]

t43=d.iloc[:,42:43]

t44=d.iloc[:,43:44]

t45=d.iloc[:,44:45]

t46=d.iloc[:,45:46]

t47=d.iloc[:,46:47]

t48=d.iloc[:,47:48]

t49=d.iloc[:,48:49]

t50=d.iloc[:,49:50]

t51=d.iloc[:,50:51]

t52=d.iloc[:,51:52]

t53=d.iloc[:,52:53]

t54=d.iloc[:,53:54]

t55=d.iloc[:,54:55]

t56=d.iloc[:,55:56]

t57=d.iloc[:,56:57]

t58=d.iloc[:,57:58]

t59=d.iloc[:,58:59]

t60=d.iloc[:,59:60]

bb=b['ghi_pyr']
t1.columns=['t-1']

t2.columns=['t-2']

t3.columns=['t-3']

t4.columns=['t-4']

t5.columns=['t-5']

t6.columns=['t-6']

t7.columns=['t-7']

t8.columns=['t-8']

t9.columns=['t-9']

t10.columns=['t-10']

t11.columns=['t-11']

t12.columns=['t-12']

t13.columns=['t-13']

t14.columns=['t-14']

t15.columns=['t-15']

t16.columns=['t-16']

t17.columns=['t-17']

t18.columns=['t-18']

t19.columns=['t-19']

t20.columns=['t-20']

t21.columns=['t-21']

t22.columns=['t-22']

t23.columns=['t-23']

t24.columns=['t-24']

t25.columns=['t-25']

t26.columns=['t-26']

t27.columns=['t-27']

t28.columns=['t-28']

t29.columns=['t-29']

t30.columns=['t-30']

t31.columns=['t-31']

t32.columns=['t-32']

t33.columns=['t-33']

t34.columns=['t-34']

t35.columns=['t-35']

t36.columns=['t-36']

t37.columns=['t-37']

t38.columns=['t-38']

t39.columns=['t-39']

t40.columns=['t-40']

t41.columns=['t-41']

t42.columns=['t-42']

t43.columns=['t-43']

t44.columns=['t-44']

t45.columns=['t-45']

t46.columns=['t-46']

t47.columns=['t-47']

t48.columns=['t-48']

t49.columns=['t-49']

t50.columns=['t-50']

t51.columns=['t-51']

t52.columns=['t-52']

t53.columns=['t-53']

t54.columns=['t-54']

t55.columns=['t-55']

t56.columns=['t-56']

t57.columns=['t-57']

t58.columns=['t-58']

t59.columns=['t-59']

t60.columns=['t-60']

bb.columns=['ghi_pyr']
aa=pd.concat([t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15,t16,t17,t18,t19,t20,t21,t22,t23,t24,t25,t26,t27,t28,t29,t30,t31,t32,t33,t34,t35,t36,t37,t38,t39,t40,t41,t42,t43,t44,t45,t46,t47,t48,t49,t50,t51,t52,t53,t54,t55,t56,t57,t58,t59,t60,bb], axis=1)
cc=aa.iloc[61:,:]
z=cc['ghi_pyr']
from sklearn.metrics import mutual_info_score

import math

y=cc['t-1']

m1=mutual_info_score(z, y, contingency=None)



y=cc['t-2']

m2=mutual_info_score(z, y, contingency=None)



y=cc['t-3']

m3=mutual_info_score(z, y, contingency=None)



y=cc['t-4']

m4=mutual_info_score(z, y, contingency=None)



y=cc['t-5']

m5=mutual_info_score(z, y, contingency=None)



y=cc['t-6']

m6=mutual_info_score(z, y, contingency=None)



y=cc['t-7']

m7=mutual_info_score(z, y, contingency=None)



y=cc['t-8']

m8=mutual_info_score(z, y, contingency=None)



y=cc['t-9']

m9=mutual_info_score(z, y, contingency=None)



y=cc['t-10']

m10=mutual_info_score(z, y, contingency=None)



y=cc['t-11']

m11=mutual_info_score(z, y, contingency=None)



y=cc['t-12']

m12=mutual_info_score(z, y, contingency=None)



y=cc['t-13']

m13=mutual_info_score(z, y, contingency=None)



y=cc['t-14']

m14=mutual_info_score(z, y, contingency=None)



y=cc['t-15']

m15=mutual_info_score(z, y, contingency=None)



y=cc['t-16']

m16=mutual_info_score(z, y, contingency=None)



y=cc['t-17']

m17=mutual_info_score(z, y, contingency=None)



y=cc['t-18']

m18=mutual_info_score(z, y, contingency=None)



y=cc['t-19']

m19=mutual_info_score(z, y, contingency=None)



y=cc['t-20']

m20=mutual_info_score(z, y, contingency=None)



y=cc['t-21']

m21=mutual_info_score(z, y, contingency=None)



y=cc['t-22']

m22=mutual_info_score(z, y, contingency=None)



y=cc['t-23']

m23=mutual_info_score(z, y, contingency=None)



y=cc['t-24']

m24=mutual_info_score(z, y, contingency=None)



y=cc['t-25']

m25=mutual_info_score(z, y, contingency=None)



y=cc['t-26']

m26=mutual_info_score(z, y, contingency=None)



y=cc['t-27']

m27=mutual_info_score(z, y, contingency=None)



y=cc['t-28']

m28=mutual_info_score(z, y, contingency=None)



y=cc['t-29']

m29=mutual_info_score(z, y, contingency=None)



y=cc['t-30']

m30=mutual_info_score(z, y, contingency=None)



y=cc['t-31']

m31=mutual_info_score(z, y, contingency=None)



y=cc['t-32']

m32=mutual_info_score(z, y, contingency=None)



y=cc['t-33']

m33=mutual_info_score(z, y, contingency=None)



y=cc['t-34']

m34=mutual_info_score(z, y, contingency=None)



y=cc['t-35']

m35=mutual_info_score(z, y, contingency=None)



y=cc['t-36']

m36=mutual_info_score(z, y, contingency=None)



y=cc['t-37']

m37=mutual_info_score(z, y, contingency=None)



y=cc['t-38']

m38=mutual_info_score(z, y, contingency=None)



y=cc['t-39']

m39=mutual_info_score(z, y, contingency=None)



y=cc['t-40']

m40=mutual_info_score(z, y, contingency=None)



y=cc['t-41']

m41=mutual_info_score(z, y, contingency=None)



y=cc['t-42']

m42=mutual_info_score(z, y, contingency=None)



y=cc['t-43']

m43=mutual_info_score(z, y, contingency=None)



y=cc['t-44']

m44=mutual_info_score(z, y, contingency=None)



y=cc['t-45']

m45=mutual_info_score(z, y, contingency=None)



y=cc['t-46']

m46=mutual_info_score(z, y, contingency=None)



y=cc['t-47']

m47=mutual_info_score(z, y, contingency=None)



y=cc['t-48']

m48=mutual_info_score(z, y, contingency=None)



y=cc['t-49']

m49=mutual_info_score(z, y, contingency=None)



y=cc['t-50']

m50=mutual_info_score(z, y, contingency=None)



y=cc['t-51']

m51=mutual_info_score(z, y, contingency=None)



y=cc['t-52']

m52=mutual_info_score(z, y, contingency=None)



y=cc['t-53']

m53=mutual_info_score(z, y, contingency=None)



y=cc['t-54']

m54=mutual_info_score(z, y, contingency=None)



y=cc['t-55']

m55=mutual_info_score(z, y, contingency=None)



y=cc['t-56']

m56=mutual_info_score(z, y, contingency=None)



y=cc['t-57']

m57=mutual_info_score(z, y, contingency=None)



y=cc['t-58']

m58=mutual_info_score(z, y, contingency=None)



y=cc['t-59']

m59=mutual_info_score(z, y, contingency=None)



y=cc['t-60']

m60=mutual_info_score(z, y, contingency=None)

ms1=m1/5

logm1=math.log(ms1,10)



ms2=m2/5

logm2=math.log(ms2,10)



ms3=m3/5

logm3=math.log(ms3,10)



ms4=m4/5

logm4=math.log(ms4,10)



ms5=m5/5

logm5=math.log(ms5,10)



ms6=m6/5

logm6=math.log(ms6,10)



ms7=m7/5

logm7=math.log(ms7,10)



ms8=m8/5

logm8=math.log(ms8,10)



ms9=m9/5

logm9=math.log(ms9,10)



ms10=m10/5

logm10=math.log(ms10,10)



ms11=m11/5

logm11=math.log(ms11,10)



ms12=m12/5

logm12=math.log(ms12,10)



ms13=m13/5

logm13=math.log(ms13,10)

ms14=m14/5

logm14=math.log(ms14,10)

ms15=m15/5

logm15=math.log(ms15,10)

ms16=m16/5

logm16=math.log(ms16,10)

ms17=m17/5

logm17=math.log(ms17,10)

ms18=m18/5

logm18=math.log(ms18,10)

ms19=m19/5

logm19=math.log(ms19,10)

ms20=m20/5

logm20=math.log(ms20,10)



ms21=m21/5

logm21=math.log(ms21,10)



ms22=m22/5

logm22=math.log(ms22,10)



ms23=m23/5

logm23=math.log(ms23,10)



ms24=m24/5

logm24=math.log(ms24,10)



ms25=m25/5

logm25=math.log(ms25,10)



ms26=m26/5

logm26=math.log(ms26,10)



ms27=m27/5

logm27=math.log(ms27,10)



ms28=m28/5

logm28=math.log(ms28,10)



ms29=m29/5

logm29=math.log(ms29,10)



ms30=m30/5

logm30=math.log(ms30,10)



ms31=m31/5

logm31=math.log(ms31,10)



ms32=m32/5

logm32=math.log(ms32,10)



ms33=m33/5

logm33=math.log(ms33,10)



ms34=m34/5

logm34=math.log(ms34,10)



ms35=m35/5

logm35=math.log(ms35,10)



ms36=m36/5

logm36=math.log(ms36,10)



ms37=m37/5

logm37=math.log(ms37,10)



ms38=m38/5

logm38=math.log(ms38,10)



ms39=m39/5

logm39=math.log(ms39,10)



ms40=m40/5

logm40=math.log(ms40,10)



ms41=m41/5

logm41=math.log(ms41,10)



ms42=m42/5

logm42=math.log(ms42,10)



ms43=m43/5

logm43=math.log(ms43,10)



ms44=m44/5

logm44=math.log(ms44,10)



ms45=m45/5

logm45=math.log(ms45,10)



ms46=m46/5

logm46=math.log(ms46,10)



ms47=m47/5

logm47=math.log(ms47,10)



ms48=m48/5

logm48=math.log(ms48,10)



ms49=m49/5

logm49=math.log(ms49,10)



ms50=m50/5

logm50=math.log(ms50,10)

ms51=m51/5

logm51=math.log(ms51,10)

ms52=m52/5

logm52=math.log(ms52,10)

ms53=m53/5

logm53=math.log(ms53,10)

ms54=m54/5

logm54=math.log(ms54,10)

ms55=m55/5

logm55=math.log(ms55,10)

ms56=m56/5

logm56=math.log(ms56,10)

ms57=m57/5

logm57=math.log(ms57,10)

ms58=m58/5

logm58=math.log(ms58,10)

ms59=m59/5

logm59=math.log(ms59,10)

ms60=m60/5

logm60=math.log(ms60,10)





x=np.array(['t1','t2','t3','t4','t5','t6','t7','t8','t9','t10','t11','t12','t13','t14','t15','t16','t17','t18','t19','t20','t21','t22','t23','t24','t25','t26','t27','t28','t29','t30','t31','t32','t33','t34','t35','t36','t37','t38','t39','t40','t41','t42','t43','t44','t45','t46','t47','t48','t49','t50','t51','t52','t53','t54','t55','t56','t57','t58','t59','t60'])

y=np.array([logm1,logm2,logm3,logm4,logm5,logm6,logm7,logm8,logm9,logm10,logm11,logm12,logm13,logm14,logm15,logm16,logm17,logm18,logm19,logm20,logm21,logm22,logm23,logm24,logm25,logm26,logm27,logm28,logm29,logm30,logm31,logm32,logm33,logm34,logm35,logm36,logm37,logm38,logm39,logm40,logm41,logm42,logm43,logm44,logm45,logm46,logm47,logm48,logm49,logm50,logm51,logm52,logm53,logm54,logm55,logm56,logm57,logm58,logm59,logm60])
print(logm1)

print(logm2)

print(logm3)

print(logm4)

print(logm5)

print(logm6)

print(logm7)

print(logm8)

print(logm53)

print(logm54)

print(logm55)

print(logm56)

print(logm57)

print(logm58)

print(logm59)

print(logm60)
import seaborn as sns

sns.barplot(x, y)
b
f=b.iloc[61:,:]

y=f['air_temperature']

AT=mutual_info_score(z, y, contingency=None)

#at=AT/5

xx=math.log(AT,10)

print('MI with air_temperature is', xx)

y=f['relative_humidity']

AT=mutual_info_score(z, y, contingency=None)

xx=math.log(AT,10)

print('MI with relative_humidity is', xx)

y=f['wind_speed']

AT=mutual_info_score(z, y, contingency=None)

xx=math.log(AT,10)

print('MI with wind_speed is', xx)



y=f['wind_speed_of_gust']

AT=mutual_info_score(z, y, contingency=None)

xx=math.log(AT,10)

print('MI with wind_speed_of_gust is', xx)



y=f['wind_from_direction_st_dev']

AT=mutual_info_score(z, y, contingency=None)

xx=math.log(AT,10)

print('MI with wind_from_direction_st_dev is', xx)



y=f['wind_from_direction']

AT=mutual_info_score(z, y, contingency=None)

xx=math.log(AT,10)

print('MI with wind_from_direction is', xx)



y=f['barometric_pressure']

AT=mutual_info_score(z, y, contingency=None)

xx=math.log(AT,10)

print('MI with barometric_pressure is', xx)

a=f

a
del a['ghi_rsi']

del a['dni']

del a['dhi']

del a['wind_speed']

del a['wind_from_direction']

del a['minute']

del a['sensor_cleaning']

del a['Month']

del a['day']

del a['hour']

del a['ghi_pyr']

del a['time']
a
aa=pd.concat([t9,t8,t7,t6,t5,t4,t3,bb], axis=1)

aa=aa.iloc[61:,:]
g=pd.concat([a,aa], axis=1)
g
X=g.iloc[:,0:12].values

Y=g.iloc[:,1:13].values
l,m=X.shape

import math

trl= math.floor(0.75*l)
X_train = X[:trl,:]

X_test = X[trl:,:]

Y_train = Y[:trl,:]

Y_test = Y[trl:,:]
import keras

from sklearn.preprocessing import MinMaxScaler

# Scaling

sc = MinMaxScaler(feature_range=(0,1))

train_set_scaled = sc.fit_transform(X_train[:,:])

dev_set_scaled = sc.fit_transform(X_test[:,:])

Y_train=sc.fit_transform(Y_train[:,:])

Y_test=sc.fit_transform(Y_test[:,:])

print(train_set_scaled.shape, dev_set_scaled.shape, Y_train.shape, Y_test.shape)
X_train = []

y_train = []

X_test = []

y_test = []



X_train, y_train = train_set_scaled[:,:], Y_train

X_test, y_test = dev_set_scaled[:,:], Y_test



print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
def shape_for_lstm(input_df, seq_length=3):



    raw_values = input_df

    raw_shape = raw_values.shape

    new_values = np.empty([raw_shape[0] - seq_length, seq_length, raw_shape[1]])

    cur_pos = 0

    for i in range(new_values.shape[0]):

        next_pos = cur_pos + seq_length

        new_values[i,:,:] = raw_values[cur_pos:next_pos,:]

        cur_pos = cur_pos + 1

    return new_values
x_train = shape_for_lstm(X_train, seq_length=3)

x_test = shape_for_lstm(X_test, seq_length=3)

Y_train = shape_for_lstm(y_train, seq_length=3)

Y_test = shape_for_lstm(y_test, seq_length=3)
import numpy as np # linear algebra

from numpy.lib.stride_tricks import as_strided

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas import Series, DataFrame

from matplotlib import pyplot as plt

plt.style.use('fivethirtyeight')

plt.rcParams['figure.figsize'] = [12, 8]

from keras.models import Sequential

from keras.layers import Dense, LSTM, Flatten

from keras.optimizers import Adam

import keras.backend as K

from sklearn.preprocessing import MinMaxScaler

from datetime import timedelta
model = Sequential()

model.add(LSTM(x_train.shape[2], input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True))

model.compile(optimizer=Adam(lr=0.001), loss='mean_squared_error', metrics=['mse'])

model.fit(x_train, Y_train, batch_size=4, epochs=5, verbose=1)
Y_pred = model.predict(x_test, batch_size=1)
Y_pred.shape
j=Y_pred[:,2:3,:]

k=Y_test[:,2:3,:]
k.shape
J=j.reshape(14027,12)

K=k.reshape(14027,12)












yhat_inverse = sc.inverse_transform(J)

testY_inverse = sc.inverse_transform(K)
yp=yhat_inverse[:,11:12]

yt=testY_inverse[:,11:12]
def rmse(predictions, targets):

    return np.sqrt(((predictions - targets) ** 2).mean())
rmse(yp,yt)