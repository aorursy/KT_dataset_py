# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.decomposition import PCA

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn import metrics

from keras.models import Sequential

from keras.layers import Dense, LSTM



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import warnings

warnings.filterwarnings("ignore")

# Any results you write to the current directory are saved as output.
df_temp = pd.read_csv('../input/electric-motor-temperature/pmsm_temperature_data.csv')
print(df_temp.head())

print(df_temp.info())

print(df_temp['profile_id'].unique())

print(df_temp['profile_id'].value_counts())



df_temp['Voltage'] = np.sqrt(df_temp['u_d']**2 + df_temp['u_q']**2)

df_temp['Current'] = np.sqrt(df_temp['i_q']**2 + df_temp['i_d']**2)



#sns.pairplot(data=df_temp, y_vars=['torque'], hue='profile_id', x_vars=['Voltage', 'Current', 'motor_speed', 'stator_yoke', 'ambient', 'coolant'])

#plt.show()
df_temps = df_temp[['pm', 'stator_yoke', 'stator_tooth', 'stator_winding']]

pca1 = PCA(n_components=1)

pca_data1 = pca1.fit_transform(df_temps)

df_temps_pca = pd.DataFrame(data = pca_data1, columns=['PrincipleComponentTemp'])



df_speed = df_temp[['motor_speed', 'i_d']]

pca2 = PCA(n_components=1)

pca_data2 = pca2.fit_transform(df_speed)

df_speed_pca = pd.DataFrame(data=pca_data2, columns=['SpeedComponent'])



df_torque = df_temp[['torque']]

df_temp.drop(['pm', 'stator_yoke', 'stator_tooth', 'stator_winding', 'i_q', 'u_d', 'torque', 'motor_speed', 'i_d', 'u_q'], axis=1, inplace=True)

df_temp['Temps'] = df_temps_pca['PrincipleComponentTemp']

df_temp['Speed'] = df_speed_pca['SpeedComponent']

df_profile_id  = df_temp[['profile_id']]

df_temp.drop(['profile_id'], axis=1, inplace=True)

print(df_temp.info())
scaler = MinMaxScaler(copy=True, feature_range=(-1,1))

df_final = scaler.fit_transform(df_temp)

df_torque = scaler.fit_transform(df_torque)



corr = df_temp.corr()

sns.heatmap(corr, annot=True, cbar=True)

plt.show()

# df_final = pd.concat([df_temp, df_profile_id])

# df_final = df_temp

print(df_final[0:5, :])

print(df_torque[0:5, :])

input = df_final

output = df_torque

print(input.shape, output.shape)



timestep = 10

input = input.reshape(int(input.shape[0]/timestep), timestep, 6)

output = output.reshape(int(output.shape[0]/timestep), timestep, 1)

print(input.shape, output.shape)



train_x, test_x, train_y, test_y = train_test_split(input, output, test_size=0.20, random_state=0)

print(train_x.shape)

print(test_x.shape)

print(train_y.shape)

print(test_y.shape)
model = Sequential()

model.add(LSTM(units=12, return_sequences=True, input_shape=(train_x.shape[1], train_x.shape[2])))

model.add(LSTM(units=3, return_sequences=True))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

result = model.fit(train_x, train_y, epochs=50,batch_size=100, verbose=1, validation_split=0.10)
plt.plot(result.history['loss'])

plt.plot(result.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()
y_pred = model.predict(test_x)

print(y_pred.shape, test_y.shape)

test_y = test_y.reshape(test_y.shape[0]*test_y.shape[1], 1)

y_pred = y_pred.reshape(y_pred.shape[0]*y_pred.shape[1], 1)

plt.plot(test_y)

plt.plot(y_pred)

plt.title('Comparison')

plt.show()
print("MAE = %f"%(metrics.mean_absolute_error(test_y, y_pred)))

print("MSE = %f"%(metrics.mean_squared_error(test_y, y_pred)))

print("R^2 error = %f"%(metrics.r2_score(test_y, y_pred)))