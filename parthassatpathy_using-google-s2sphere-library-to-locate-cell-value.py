import numpy as np

import pandas as pd

import s2sphere
train_df = pd.read_json('../input/train.json')

test_df = pd.read_json('../input/test.json')
lat = train_df.latitude.values.tolist()

lon = train_df.longitude.values.tolist()



cellId1 = []

cellId2 = []

cellId3 = []

for i in range(0,len(lat)):

    p1 = s2sphere.LatLng.from_degrees(lat[i], lon[i])

    cell = s2sphere.CellId.from_lat_lng(p1)

    cid = str(cell.id())

    #print(cid)

    ##cid is a 19 digit number so python storing it as Object, not number

    ##So I am converting it into 3 numbers

    cellId1.append(int(cid[:6]))

    cellId2.append(int(cid[6:12]))

    cellId3.append(int(cid[12:19]))

    



se = pd.Series(cellId1)

train_df['cellId1'] = se.values



se = pd.Series(cellId2)

train_df['cellId2'] = se.values



se = pd.Series(cellId3)

train_df['cellId3'] = se.values



lat = test_df.latitude.values.tolist()

lon = test_df.longitude.values.tolist()



cellId1 = []

cellId2 = []

cellId3 = []

for i in range(0,len(lat)):

    p1 = s2sphere.LatLng.from_degrees(lat[i], lon[i])

    cell = s2sphere.CellId.from_lat_lng(p1)

    cid = str(cell.id())

    #print(cid)

    cellId1.append(int(cid[:6]))

    cellId2.append(int(cid[6:12]))

    cellId3.append(int(cid[12:19]))

    



se = pd.Series(cellId1)

test_df['cellId1'] = se.values



se = pd.Series(cellId2)

test_df['cellId2'] = se.values



se = pd.Series(cellId1)

test_df['cellId3'] = se.values

#selectedFeatures.extend(['cellId1','cellId2','cellId3'])
train_df.head()
test_df.head()