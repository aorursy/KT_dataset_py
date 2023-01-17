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
reservoir_levels = pd.read_csv('/kaggle/input/chennai-water-management/chennai_reservoir_levels.csv')

reservoir_rainfalls = pd.read_csv('/kaggle/input/chennai-water-management/chennai_reservoir_rainfall.csv')
print(reservoir_levels.head())

print(reservoir_rainfalls.head())
import matplotlib.pyplot as plt

plt.plot(range(len(reservoir_levels['POONDI'])),reservoir_levels['POONDI'])

plt.plot(range(len(reservoir_levels['CHOLAVARAM'])),reservoir_levels['CHOLAVARAM'])

plt.plot(range(len(reservoir_levels['REDHILLS'])),reservoir_levels['REDHILLS'])

plt.plot(range(len(reservoir_levels['CHEMBARAMBAKKAM'])),reservoir_levels['CHEMBARAMBAKKAM'])

plt.legend()
import matplotlib.pyplot as plt

plt.plot(range(len(reservoir_rainfalls['POONDI'])),reservoir_rainfalls['POONDI'])

plt.plot(range(len(reservoir_rainfalls['CHOLAVARAM'])),reservoir_rainfalls['CHOLAVARAM'])

plt.plot(range(len(reservoir_rainfalls['REDHILLS'])),reservoir_rainfalls['REDHILLS'])

plt.plot(range(len(reservoir_rainfalls['CHEMBARAMBAKKAM'])),reservoir_rainfalls['CHEMBARAMBAKKAM'])

plt.legend()
import seaborn as sns



rainfall_corr = reservoir_rainfalls.corr()

level_corr = reservoir_levels.corr()



fig, (ax1, ax2) = plt.subplots(1,2)

sns.heatmap(rainfall_corr, ax=ax1,annot=True)

sns.heatmap(level_corr, ax=ax2,annot=True)

plt.show()
# group by month

# predict by year
poondi = [[each] for each in reservoir_levels['POONDI'].values]

cholavaram = [[each] for each in reservoir_levels['CHOLAVARAM'].values]

redhills = [[each] for each in reservoir_levels['REDHILLS'].values]

chembarambakkam = [[each] for each in reservoir_levels['CHEMBARAMBAKKAM'].values]
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range = (0, 1))

y_poondi = scaler.fit_transform(poondi)

y_cholavaram = scaler.fit_transform(cholavaram)

y_redhills = scaler.fit_transform(redhills)

y_chembarambakkam = scaler.fit_transform(chembarambakkam)
def get_feature_set(train_scaled):

    features_set = []  

    labels = []

    for i in range(60, train_scaled.shape[0]):  

        features_set.append(train_scaled[i-60:i,0])

        labels.append(train_scaled[i,0])



    features_set, labels = np.array(features_set), np.array(labels)  

    features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))

    

    return features_set
l_train = int(y_poondi.shape[0]*0.7)

l_test = int(y_poondi.shape[0]*0.3)

train_poondi = y_poondi[:l_train]

test_poondi = y_poondi[-l_test:]

train_cholavaram = y_cholavaram[:l_train]

test_cholavaram = y_cholavaram[-l_test:]

train_redhills = y_redhills[:l_train]

test_redhills = y_redhills[-l_test:]

train_chembarambakkam = y_chembarambakkam[:l_train]

test_chembarambakkam = y_chembarambakkam[-l_test:]
feature_set_p = get_feature_set(train_poondi)

feature_set_c = get_feature_set(train_cholavaram)

feature_set_r = get_feature_set(train_redhills)

feature_set_ch = get_feature_set(train_chembarambakkam)
print(feature_set_p.shape, feature_set_c.shape, feature_set_r.shape, feature_set_ch.shape)
from keras.models import Sequential  

from keras.layers import Dense, LSTM, Dropout



def def_model(feature_set):

    model = Sequential()



    model.add(LSTM(units=50, return_sequences=True, input_shape=(feature_set.shape[1], 1)))  

    model.add(Dropout(0.2))  

    model.add(LSTM(units=50, return_sequences=True))  

    model.add(Dropout(0.2))



    model.add(LSTM(units=50, return_sequences=True))  

    model.add(Dropout(0.2))



    model.add(LSTM(units=50))  

    model.add(Dropout(0.2))  



    model.add(Dense(units = 1)) 

    model.compile(optimizer = 'adam', loss = 'mean_squared_error')

    return model
pmodel = def_model(feature_set_c)

cmodel = def_model(feature_set_c)

rmodel = def_model(feature_set_c)

chmodel = def_model(feature_set_c)

chmodel.summary()
pmodel.fit(feature_set_p, train_poondi[60:], epochs = 10, batch_size = 32)
feature_set_tp = get_feature_set(test_poondi)

predictions = pmodel.predict(feature_set_tp)
scaler.inverse_transform(predictions)
plt.figure(figsize=(10,6))  

plt.plot(test_poondi[60:], color='blue', label='Actual')  

plt.plot(predictions, color='red', label='Predicted')  

plt.legend()  

plt.show()
cmodel.fit(feature_set_c, train_cholavaram[60:], epochs = 10, batch_size = 32)

rmodel.fit(feature_set_r, train_redhills[60:], epochs = 10, batch_size = 32)

chmodel.fit(feature_set_ch, train_chembarambakkam[60:], epochs = 10, batch_size = 32)
feature_set_tc = get_feature_set(test_cholavaram)

feature_set_tr = get_feature_set(test_redhills)

feature_set_tch = get_feature_set(test_chembarambakkam)



predictions_tc = cmodel.predict(feature_set_tc)

predictions_tr = rmodel.predict(feature_set_tr)

predictions_tch = chmodel.predict(feature_set_tch)
from keras.models import load_model

pmodel.save('/kaggle/working/pmodel.h5')

cmodel.save('/kaggle/working/cmodel.h5')

rmodel.save('/kaggle/working/rmodel.h5')

chmodel.save('/kaggle/working/chmodel.h5')
crop_range_df = pd.read_csv('/kaggle/input/monthly-normalized-crop-ranges/ranges.csv')

crop_range_df
import random



months = crop_range_df.columns

df = []



for year in range(2004,2020):

    for month,each in enumerate(crop_range_df.iloc[0,:].values): # one year

        x = each.split('-')

        high = float(x[0])

        low = float(x[1])

        noise = random.uniform(-0.05,0.05)

        df.append([months[month], year, round(random.uniform(high,low)+noise,2)])



df_generated = pd.DataFrame(df,columns=['month','year','crop'])

df_generated.head()
# example for one field

import matplotlib.pyplot as plt



for i in range(0,len(df_generated),12):

    plt.plot(df_generated.iloc[i:i+12,2:3].values)

import random



months = crop_range_df.columns

df = []



for year in range(2004,2020):

    for month,each in enumerate(crop_range_df.iloc[0,:].values): # one year

        x = each.split('-')

        high = float(x[0])

        low = float(x[1])

        noise = random.uniform(-0.05,0.05)

        # df.append([months[month], year, round(random.uniform(high,low)+noise,2)])

        for field in range(0,20):

            df.append([months[month], year, round(random.uniform(high,low)+noise,2), field])



df_generated = pd.DataFrame(df,columns=['month','year','crop','field'])

df_generated.head()
columns = ['field_id','requirement','closest_reservoir','month','year']
'''

Assume there are 20 fields in a district and these 4 are the only reservoirs.

Dataset: field_id, its requirement (sampled from a function) in a month and the closest reservoir

Group by month, field and closest. 

Predict: requirement in next month (1 months at a time)

Make analysis before going to next month



Water going to field (available) = Prev month beginning - next month beginning in reservoir



Ex: Jan - F0 - 830 required - POONDI

    Jan - F1 - 650 required - POONDI ... predicted values in Jan

    if sum(requirements from POONDI) > available(POONDI) in Jan:

        # do something

'''
# say 20 fields, 4 reservoirs

#                0           1           2              3

reservoirs = ['POONDI','CHOLAVARAM','REDHILLS','CHEMBARAMBAKKAM']

closest_reservoirs = [random.randint(0,3) for i in range(20)]

print(closest_reservoirs)
import matplotlib.pyplot as plt

plt.hist(closest_reservoirs)
for field in range(0,20):

    df_generated.loc[df_generated['field']==field,'closest_reservoir'] = closest_reservoirs[field]

df_generated.head()
len(df_generated)
# Groupby month, year and closest_reservoir

df_generated_copy = df_generated.copy()

df_generated_copy = df_generated_copy.drop(['field'], axis=1)

df_generated_copy = df_generated_copy.groupby(['closest_reservoir','year','month'],sort=False).agg({'crop':'sum'})

# df_generated_copy.head(12)
# normalize crop column

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range = (0, 1))

df_generated_copy['crop'] = scaler.fit_transform([[crop] for crop in df_generated_copy['crop'].values])

# df_generated_copy.head()

df_generated_copy.loc[1.0,:].head(12)
closest_r0 = df_generated_copy.loc[0.0,:].values

closest_r1 = df_generated_copy.loc[1.0,:].values

closest_r2 = df_generated_copy.loc[2.0,:].values

closest_r3 = df_generated_copy.loc[3.0,:].values

plt.plot(closest_r_0,label="0")

plt.plot(closest_r_1,label="1")

plt.plot(closest_r_2,label="2")

plt.plot(closest_r_3,label="3")

plt.legend()
train_closest_r2 = closest_r2[:-80]

test_closest_r2 = closest_r2[-80:]

plt.plot(train_closest_r2)
# increase data points.. i.e. add values per day basis

feature_set_r2 = get_feature_set(train_closest_r2)

r2model = def_model(feature_set=feature_set_r2)

r0model.fit(feature_set_r2, train_closest_r2[60:], epochs = 10, batch_size = 32)
len(train_closest_r2)
feature_set_tr2 = get_feature_set(test_closest_r2)

predictions_r2 = r0model.predict(feature_set_tr2)
plt.figure(figsize=(10,6))  

plt.plot(test_closest_r2[60:], color='blue', label='Actual')  

plt.plot(predictions_r2, color='red', label='Predicted')  

plt.legend()  

plt.show()