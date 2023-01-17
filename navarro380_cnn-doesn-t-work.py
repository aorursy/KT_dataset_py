import numpy as np 

import pandas as pd
data=pd.read_csv("../input/X_train.csv")

data_pred=pd.read_csv("../input/X_test.csv")

label=pd.read_csv("../input/y_train.csv")
'''NOTE: This code cames from "Nanashi Student at University of Valladolid" Thanks to Nanashi '''







def feat_eng(data):

    

    df = pd.DataFrame()

    data['totl_anglr_vel'] = (data['angular_velocity_X']**2 + data['angular_velocity_Y']**2 + data['angular_velocity_Z']**2)** 0.5

    data['totl_linr_acc'] = (data['linear_acceleration_X']**2 + data['linear_acceleration_Y']**2 + data['linear_acceleration_Z']**2)**0.5

    data['totl_xyz'] = (data['orientation_X']**2 + data['orientation_Y']**2 + data['orientation_Z']**2)**0.5

    data['acc_vs_vel'] = data['totl_linr_acc'] / data['totl_anglr_vel']

    

    def mean_change_of_abs_change(x):

        return np.mean(np.diff(np.abs(np.diff(x))))

    

    for col in data.columns:

        if col in ['row_id','measurement_number']:

            continue

        df[col + '_mean'] = data.groupby(['series_id'])[col].mean()

        df[col + '_median'] = data.groupby(['series_id'])[col].median()

        df[col + '_max'] = data.groupby(['series_id'])[col].max()

        df[col + '_min'] = data.groupby(['series_id'])[col].min()

        df[col + '_std'] = data.groupby(['series_id'])[col].std()

        df[col + '_range'] = df[col + '_max'] - df[col + '_min']

        df[col + '_maxtoMin'] = df[col + '_max'] / df[col + '_min']

        df[col + '_mean_abs_chg'] = data.groupby(['series_id'])[col].apply(lambda x: np.mean(np.abs(np.diff(x))))

        df[col + '_mean_change_of_abs_change'] = data.groupby('series_id')[col].apply(mean_change_of_abs_change)

        df[col + '_abs_max'] = data.groupby(['series_id'])[col].apply(lambda x: np.max(np.abs(x)))

        df[col + '_abs_min'] = data.groupby(['series_id'])[col].apply(lambda x: np.min(np.abs(x)))

        df[col + '_abs_avg'] = (df[col + '_abs_min'] + df[col + '_abs_max'])/2

    return df

    
df=feat_eng(data)

df.head()
df1=df.copy()
df1.isnull().any().any()

label.head()
df3=pd.merge(df1, label, left_on='series_id', right_index=True, how='left', sort=False)
print('Lenght data--->',len(data))

print('Lenght label-->',len(label))

print('Lenght df----->',len(df))
df3.columns
df4=df3.drop(['series_id_mean', 'series_id_median', 'series_id_max', 'series_id_min',

       'series_id_std', 'series_id_range', 'series_id_maxtoMin',

       'series_id_mean_abs_chg', 'series_id_mean_change_of_abs_change',

       'series_id_abs_max','series_id_abs_min', 'series_id_abs_avg','series_id', 'group_id'], axis=1)

df4.columns

from sklearn.model_selection import train_test_split as split

train, test=split(df4, test_size=0.3)
x=train.drop(['surface'], axis=1)

y=train.loc[:,['surface']]

y['surface_fact']=pd.factorize(y.surface)[0]

x.shape

xt=test.drop(['surface'], axis=1)

yt=test.loc[:,['surface']]

yt['surface_fact']=pd.factorize(yt.surface)[0]

x.head()
from sklearn.preprocessing import minmax_scale

x_arr=minmax_scale(x, axis=0)

xt_arr=minmax_scale(xt, axis=0)
x_arr.shape
print(yt.shape)

pd.value_counts(yt['surface'])
pd.value_counts(y['surface_fact'])
import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers

from tensorflow.keras import optimizers

from tensorflow.keras.utils import to_categorical

from tensorflow.keras import backend as k

from tensorflow.keras.models import model_from_json



yc=to_categorical(y['surface_fact'])

yct=to_categorical(yt['surface_fact'])

yc
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Conv1D, Flatten
#create model

model = Sequential()

k.clear_session()

#add model layers

model.add(Conv1D(168, kernel_size=3, activation='relu', input_shape=(168,1,)))

model.add(Conv1D(100, kernel_size=3, activation='relu'))

model.add(Conv1D(50, kernel_size=3, activation='relu'))

model.add(Conv1D(25, kernel_size=3, activation='relu'))

model.add(Flatten())

model.add(Dense(9, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(x_arr.shape)

print(xt_arr.shape)
x_a1=x_arr.reshape(2667, 168,1,)

xt_a1=xt_arr.reshape(1143, 168,1,)
model.fit(x_a1, yc, epochs=50, batch_size=512)
model.evaluate(xt_a1,yct, batch_size=20)