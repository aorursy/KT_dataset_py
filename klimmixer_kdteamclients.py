# Гипотезы и этапы обработки данных KDTeam задача с обнаружением потенциальных клиентов



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn



# Input data files are available in the "../input/" directory.

import os

print(os.listdir("../input"))
data = pd.read_csv('../input/input_data.csv')

data['url'] = data['url'].fillna('https://www.mic-bunino.ru/')
data = data.drop(['_id', 'timestamp', 'ipxRealIp', 'QueryString', '_index', '_type', '_score', 'event_type', 'event_value', 'ipRemoteAddr', 'ipxForwardedFor', 'type', 'site_id'], axis=1)
columns = ['user_id', 'isClient', 'isIOS', 'isAndroid', 'otherOS', 'isPhone', 'isTablet', 'otherDeviceType']

for i in pd.get_dummies(data['url'], prefix='url'):

    columns.append(i)

dataTrain = pd.DataFrame(columns=columns)
def preprocess(row):

    global dataTrain

    print(row)

    if row['user_id'] not in dataTrain['user_id'].unique():

        temp = {'user_id':row['user_id']}

        dataTrain = dataTrain.append(temp, ignore_index=True)

    if row['isIOS']:

        dataTrain['isIOS'][row['user_id'] == dataTrain['user_id']] = 1

    elif row['isAndroidOS']:

        dataTrain['isAndroid'][row['user_id'] == dataTrain['user_id']] = 1

    else:

        dataTrain['otherOS'][row['user_id'] == dataTrain['user_id']] = 1

    if row['isPhone']:

        dataTrain['isPhone'][row['user_id'] == dataTrain['user_id']] = 1

    elif row['isTablet']:

        dataTrain['isTablet'][row['user_id'] == dataTrain['user_id']] = 1

    else:

        dataTrain['otherDeviceType'][row['user_id'] == dataTrain['user_id']] = 1

    dataTrain['url_'+str(row['url'])][row['user_id'] == dataTrain['user_id']] = 1

    return row
data.apply(preprocess, axis=1)
dataTrain = dataTrain.fillna(0)

dataTrain
import random



for i in range(100):

    dataTrain['isClient'][dataTrain['user_id'] == random.choice(dataTrain['user_id'].unique())] = 1
x_train = dataTrain[[x for x in dataTrain if 'user_id' not in x and 'isClient' not in x]].as_matrix()

y_train = dataTrain['isClient'].as_matrix()
tree =  sklearn.tree.DecisionTreeClassifier()