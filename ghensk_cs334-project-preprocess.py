import numpy as np

import pandas as pd

from sklearn import preprocessing

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
renfe = pd.read_csv("../input/spanish-high-speed-rail-system-ticket-pricing/renfe.csv")

print(renfe.shape)
renfe = renfe.head(120000)
for i in ['insert_date','start_date','end_date']:

    renfe[i] = pd.to_datetime(renfe[i])
# Check for null value

renfe.isnull().mean()*100
print(renfe.info)

renfe.isnull().any()

renfe = renfe.dropna()
# Changing the origin and destination to the routes

renfe['route'] = renfe['origin'] + ' to ' + renfe['destination']
# Extracting features from 'start_date' and 'end_date'

renfe['month'] = renfe['start_date'].apply(lambda d:d.month)

renfe['day_name'] = renfe['start_date'].apply(lambda d: d.weekday_name)

renfe['quarter'] = renfe['start_date'].apply(lambda d: d.quarter)

renfe['travel_time'] = (renfe['end_date']-renfe['start_date'])/np.timedelta64(1, 'm')

renfe['start_hour'] = renfe['start_date'].apply(lambda d: d.hour)

renfe['end_hour'] = renfe['end_date'].apply(lambda d: d.hour)

print(renfe.info)
print(renfe.columns)

renfe.drop(['origin', 'destination', 'insert_date', 'start_date', 'end_date'], axis=1, inplace=True)

print(renfe.columns)
x_categ = renfe[['train_type', 'train_class', 'fare', 'day_name', 'quarter', 'route']]

x_oh = renfe[['price', 'month', 'travel_time', 'start_hour', 'end_hour']]

oh = preprocessing.OneHotEncoder()

oh_categ = oh.fit_transform(x_categ)

oh_feat = oh.get_feature_names(['train_type', 'train_class', 'fare', 'day_name', 'quarter', 'route'])

print(len(oh_feat))

print(len(np.transpose(oh_categ.toarray())))

i = 0

for c in oh_feat:

    x_oh[c] = np.transpose(oh_categ.toarray())[i]

    i += 1

print(x_oh)
#from sklearn.model_selection import train_test_split

y = x_oh['price']

xTrain_oh, xTest_oh, yTrain_oh, yTest_oh = train_test_split(x_oh.drop(columns=['price']), y, test_size=0.3, random_state=334)
xTrain_oh.to_csv("xTrain_renfe_oh.csv", index=False)

xTest_oh.to_csv("xTest_renfe_oh.csv", index=False)

yTrain_oh.to_csv("yTrain_renfe_oh.csv", header='label', index=False)

yTest_oh.to_csv("yTest_renfe_oh.csv", header='label', index=False)