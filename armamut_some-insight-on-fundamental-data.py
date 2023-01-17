import pandas as pd

import numpy as np

import plotly.offline as py

py.init_notebook_mode(connected=True)



# Pandas configuration

pd.set_option('display.max_columns', None)
# Usual stuff...

with pd.HDFStore("../input/train.h5", "r") as train:

    # Note that the "train" dataframe is the only dataframe in the file

    df = train.get("train")

df.head()
asset = df[ df['id'] == 12 ] # some random asset. you can try out with different ids



y1 = asset['fundamental_1'] # some fundamental indicators. you can try the other indicators

y2 = asset['fundamental_3'] # they all show the same phenomenon.

y3 = asset['fundamental_6']

y4 = asset['fundamental_19']



py.iplot({'data': [{'y': y1}, {'y': y2}, {'y': y3}, {'y': y4}]})
y1d = y1 - y1.shift(1)

y2d = y2 - y2.shift(1)

y3d = y3 - y3.shift(1)

y4d = y4 - y4.shift(1)



py.iplot({'data': [{'y': y1d}, {'y': y2d}, {'y': y3d}, {'y': y4d}]})
# http://stackoverflow.com/questions/643699/how-can-i-use-numpy-correlate-to-do-autocorrelation

def autocorr(x):

    result = np.correlate(x, x, mode='full')

    return result[int(result.size/2):]



acc = abs(autocorr( y1d.fillna(method='bfill') ))



py.iplot({'data': [{'y': acc}, {'y':y1d}]})
fundamentals = ['fundamental_0','fundamental_1','fundamental_2',

                'fundamental_3','fundamental_5','fundamental_6',

                'fundamental_7','fundamental_8','fundamental_9',

                'fundamental_10','fundamental_11','fundamental_12',

                'fundamental_13','fundamental_14','fundamental_15',

                'fundamental_16','fundamental_17','fundamental_18',

                'fundamental_19','fundamental_20','fundamental_21',

                'fundamental_22','fundamental_23','fundamental_24',

                'fundamental_25','fundamental_26','fundamental_27',

                'fundamental_28','fundamental_29','fundamental_30',

                'fundamental_31','fundamental_32','fundamental_33',

                'fundamental_34','fundamental_35','fundamental_36',

                'fundamental_37','fundamental_38','fundamental_39',

                'fundamental_40','fundamental_41','fundamental_42',

                'fundamental_43','fundamental_44','fundamental_45',

                'fundamental_46','fundamental_47','fundamental_48',

                'fundamental_49','fundamental_50','fundamental_51',

                'fundamental_52','fundamental_53','fundamental_54',

                'fundamental_55','fundamental_56','fundamental_57',

                'fundamental_58','fundamental_59','fundamental_60',

                'fundamental_61','fundamental_62','fundamental_63']



asset = df[ df['id'] == 12 ]

datas = []

for f in fundamentals:

    y1 = asset[f]

    y1d = y1 - y1.shift(1)

    acc = abs(autocorr( y1d.fillna(method='bfill') ))

    pp = (acc[20:300]).argmax() + 20

    datas.append((f, pp))



datas
y1 = asset['fundamental_6']

y1 = y1.fillna(method='pad')

alpha = 0.87055

yy = (y1 - alpha * y1.shift(1)) / (1-alpha)



py.iplot({'data': [{'y': y1}, {'y': yy}]})
y1 = asset['fundamental_6']

y1 = y1.fillna(method='pad')

alpha = 0.87055

yy = (y1 - alpha * y1.shift(1)) / (1-alpha)

yy = yy.rolling(7).median(center=True)



py.iplot({'data': [{'y': y1}, {'y': yy}]})