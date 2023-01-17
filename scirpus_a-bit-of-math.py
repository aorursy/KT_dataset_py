import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
X_train = pd.read_json("../input/train.json")

X_train = X_train.reset_index(drop=True)

X_train['created'] = pd.to_datetime(X_train["created"])

X_train.sort_values(by='created',inplace=True)

interest_level_map = {'low': 0, 'medium': 1, 'high': 2}

X_train['interest_level'] = X_train['interest_level'].apply(lambda x: interest_level_map[x])

X_train['initial_listing_id'] = X_train['photos'].apply(lambda x: x[0] if len(x)> 0 else '').str.extract('.com\/\d\/(\d{1,7})',expand=False).fillna('-1').astype(int)

X_train["num_photos"] = X_train["photos"].apply(len)
colors = ['red','green','blue']

flt = (X_train.num_photos>0)&(X_train.interest_level==2)

plt.figure(figsize=(10,10))

plt.scatter(X_train[flt]['initial_listing_id'],X_train[flt].listing_id,color=[colors[x] for x in X_train[flt].interest_level])
flt = (X_train.num_photos>0)&(X_train.interest_level==1)

plt.figure(figsize=(10,10))

plt.scatter(X_train[flt]['initial_listing_id'],X_train[flt].listing_id,color=[colors[x] for x in X_train[flt].interest_level])
flt = (X_train.num_photos>0)&(X_train.interest_level==0)

plt.figure(figsize=(10,10))

plt.scatter(X_train[flt]['initial_listing_id'],X_train[flt].listing_id,color=[colors[x] for x in X_train[flt].interest_level])
flt = (X_train.num_photos>0)&(X_train.interest_level==2)

plt.figure(figsize=(10,10))

p1 = np.array([0.0,0.0]) # Origin

p2 = np.array([1.0, 1.0]) # Approximate to the line

ps = np.zeros((X_train[flt].shape[0],2))

ps[:,0] = X_train[flt].initial_listing_id

ps[:,1] = X_train[flt].listing_id

dist = []

for i in range(X_train[flt].shape[0]):

    dist.append(np.linalg.norm(np.cross(p2-p1, p1-ps[i,:]))/np.linalg.norm(p2-p1))

plt.plot(dist)

flt = (X_train.num_photos>0)&(X_train.interest_level==1)

plt.figure(figsize=(10,10))

p1 = np.array([0.0,0.0]) # Origin

p2 = np.array([1.0, 1.0]) # Approximate to the line

ps = np.zeros((X_train[flt].shape[0],2))

ps[:,0] = X_train[flt].initial_listing_id

ps[:,1] = X_train[flt].listing_id

dist = []

for i in range(X_train[flt].shape[0]):

    dist.append(np.linalg.norm(np.cross(p2-p1, p1-ps[i,:]))/np.linalg.norm(p2-p1))

plt.plot(dist)
flt = (X_train.num_photos>0)&(X_train.interest_level==0)

plt.figure(figsize=(10,10))

p1 = np.array([0.0,0.0]) # Origin

p2 = np.array([1.0, 1.0]) # Approximate to the line

ps = np.zeros((X_train[flt].shape[0],2))

ps[:,0] = X_train[flt].initial_listing_id

ps[:,1] = X_train[flt].listing_id

dist = []

for i in range(X_train[flt].shape[0]):

    dist.append(np.linalg.norm(np.cross(p2-p1, p1-ps[i,:]))/np.linalg.norm(p2-p1))

plt.plot(dist)