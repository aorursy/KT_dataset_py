import numpy as np

import pandas as pd
events = pd.read_csv('../input/events.csv', usecols=['display_id', 'uuid'])
train = pd.read_csv('../input/clicks_train.csv')

train_clicked = train[train.clicked==1]

del train
train_clicked = pd.merge(train_clicked, events, how='left', on='display_id')

# combinding uuid to specify each user
train_clicked.head()

uuid_grouped = train_clicked.groupby('uuid')['ad_id']
count = 0

for uuid, ads in uuid_grouped:

    if len(ads) != 1:

        if len(ads) != len(ads.unique()):

            count += 1
num = count / len(train_clicked.uuid.unique()) * 100

print('{}%'.format(round(num, 5)))