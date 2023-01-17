# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# import numpy as np # linear algebra

# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np
AEP = pd.read_csv('../input/AEP_hourly.csv', index_col=[0], parse_dates=[0])
mau = ["#F8766D", "#D39200", "#93AA00", "#00BA38", "#00C19F", "#00B9E3", "#619CFF", "#DB72FB"]

bieudo = AEP.plot(style='.',figsize=(15,5), color=mau[0], title='AEP')

#Data transformation

def create_features(df, label=None):

    # Chọn các feature đáng chú ý của dữ liệu

    df = df.copy()

    df['date'] = df.index

    df['hour'] = df['date'].dt.hour

    df['dayofweek'] = df['date'].dt.dayofweek

    df['quarter'] = df['date'].dt.quarter

    df['month'] = df['date'].dt.month

    df['year'] = df['date'].dt.year

    df['dayofyear'] = df['date'].dt.dayofyear

    df['dayofmonth'] = df['date'].dt.day

    df['weekofyear'] = df['date'].dt.weekofyear



    X = df[['hour', 'dayofweek', 'quarter', 'month', 'year',

            'dayofyear', 'dayofmonth', 'weekofyear']]

    if label:

        y = df[label]

        return X, y

    return X





X, y = create_features(AEP, label='AEP_MW')

features_and_target = pd.concat([X, y], axis=1)

print(features_and_target)

plt.show()



plt.figure(figsize=(15,6))

data_csv = AEP.dropna()

dataset = data_csv.values

dataset = dataset.astype('float32')

max_value = np.max(dataset)

min_value = np.min(dataset)

scalar = max_value - min_value

dataset = list(map(lambda x: (x-min_value) / scalar, dataset))

plt.plot(dataset)

print(max_value, min_value)