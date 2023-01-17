train['display_address'] = train['display_address'].apply(lambda x: x.strip("."))

train['display_address']  = train['display_address'].apply(lambda x: x.lower())

ga = train.groupby(['display_address'])['display_address'].count().fillna(0)

ga = pd.DataFrame(ga)

ga.columns = ['display_count']

ga['display_address'] = ga.index

ga.loc[ga['display_address'] == '','display_count'] = 0

pd.DataFrame(ga)
null
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import log_loss

from sklearn.model_selection import train_test_split





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_json('../input/train.json')

### and test if everything OK

train.head()
train['num_photos'] = train['photos'].apply(len)

train['num_features'] = train['features'].apply(len)

train['num_description_words'] = train['description'].apply(lambda x: len(x.split(' ')))

train['rooms'] = train['bathrooms'] + train['bedrooms']



ulimit = np.percentile(train.price.values, 99)

train['price'].loc[train['price']>ulimit] = ulimit

train['rooms_per_price'] = train['rooms']/train['price']

train = train[train['bedrooms'] > 0]

train['bath_per_beds'] = train['bathrooms']/train['bedrooms']

train.loc[train['bath_per_beds'] > 999999999999,'bath_per_beds'] = 0
train['display_address'] = train['display_address'].apply(lambda x: x.strip("."))

train['display_address']  = train['display_address'].apply(lambda x: x.lower())

ga = train.groupby(['display_address'])['display_address'].count().fillna(0)

ga = pd.DataFrame(ga)

ga.columns = ['display_count']

ga['display_address'] = ga.index

ga.loc[ga['display_address'] == '','display_count'] = 0

pd.DataFrame(ga)
grouped_building = train.groupby(

                           ['building_id']

                          )['building_id'].count().fillna(0)

grouped_building = pd.DataFrame(grouped_building)

grouped_building.columns = ['building_count']

grouped_building['building_id'] = grouped_building.index

grouped_building.loc[grouped_building['building_id'] == '0','building_count'] = 0

train = pd.merge(train,grouped_building,on='building_id')



gm = train.groupby(

                           ['manager_id']

                          )['manager_id'].count().fillna(0)

gm = pd.DataFrame(gm)

gm.columns = ['manager_count']

gm['manager_id'] = gm.index

gm.loc[gm['manager_id'] == '0','manager_count'] = 0

train = pd.merge(train,gm,on='manager_id')



train['display_address'] = train['display_address'].apply(lambda x: x.strip("."))

train['display_address']  = train['display_address'].apply(lambda x: x.lower())

ga = train.groupby(['display_address'])['display_address'].count().fillna(0)

ga = pd.DataFrame(ga)

ga.columns = ['display_count']

ga['display_address'] = ga.index

ga.loc[ga['display_address'] == '','display_count'] = 0



train = pd.merge(train,ga,on='display_address')
labels = train['interest_level']

target_num_map = {'high':0, 'medium':1, 'low':2}

labels = np.array(labels.apply(lambda x: target_num_map[x]))

train['labels'] = labels
X  = train[['bathrooms','bedrooms','price','num_photos',

            'num_features','num_description_words','rooms','rooms_per_price',

            'bath_per_beds','latitude','longitude','building_count','manager_count','display_count']]

y = train['labels']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 

                                                    random_state=7)
gbc = GradientBoostingClassifier(loss='deviance', learning_rate=0.05, n_estimators=600, 

                                  subsample=1.0, criterion='friedman_mse', min_samples_split=2, 

                                  min_samples_leaf=1, min_weight_fraction_leaf=0.0, 

                                  max_depth=5, init=None, random_state=None, 

                                  max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, 

                                  presort='auto')

gbc.fit(X,y)

out = gbc.predict_proba(X_test)
print(log_loss(y_test,out))
test = pd.read_json('../input/test.json')

test['num_photos'] = test['photos'].apply(len)

test['num_features'] = test['features'].apply(len)

test['num_description_words'] = test['description'].apply(lambda x: len(x.split(' ')))

test['price'].loc[test['price']>ulimit] = ulimit

test['rooms'] = test['bathrooms'] + test['bedrooms']

test['room_per_price'] = test['rooms']/test['price']

test['bath_per_beds'] = train['bathrooms']/train['bedrooms']

test.loc[test['bath_per_beds'] > 999999999999,'bath_per_beds'] = 0



grouped_building = test.groupby(

                           ['building_id']

                          )['building_id'].count().fillna(0)

grouped_building = pd.DataFrame(grouped_building)

grouped_building.columns = ['building_count']

grouped_building['building_id'] = grouped_building.index

grouped_building.loc[grouped_building['building_id'] == '0','building_count'] = 0

test = pd.merge(test,grouped_building,on='building_id')



gm = test.groupby(

                           ['manager_id']

                          )['manager_id'].count().fillna(0)

gm = pd.DataFrame(gm)

gm.columns = ['manager_count']

gm['manager_id'] = gm.index

gm.loc[gm['manager_id'] == '0','manager_count'] = 0

test = pd.merge(test,gm,on='manager_id')



test['display_address'] = test['display_address'].apply(lambda x: x.strip("."))

test['display_address'] = test['display_address'].apply(lambda x: x.lower())

ga = test.groupby(['display_address'])['display_address'].count().fillna(0)

ga = pd.DataFrame(ga)

ga.columns = ['display_count']

ga['display_address'] = ga.index

ga.loc[ga['display_address'] == '','display_count'] = 0

test = pd.merge(test,ga,on='display_address')



X  = test[['bathrooms','bedrooms','price','num_photos',

            'num_features','num_description_words','rooms','room_per_price',

            'bath_per_beds','latitude','longitude','building_count','manager_count','display_count']]



lists = test['listing_id']

test = test.drop(['listing_id'],axis=1)


test.loc[test['bedrooms'] == 0,'bath_per_beds'] = 0

X  = test[['bathrooms','bedrooms','price','num_photos',

            'num_features','num_description_words','rooms','room_per_price',

            'bath_per_beds','latitude','longitude','building_count','manager_count','display_count']]

X = X.fillna(0)
out = gbc.predict_proba(X)
res = []

ls = np.array(lists, dtype=pd.Series)

for i,row in enumerate(out):

    res.append(np.insert(row,0,int(ls[i])))
out_df = pd.DataFrame(res)

out_df.columns = ["listing_id", "high", "medium", "low"]

out_df['listing_id'] = out_df['listing_id'].astype("int")

out_df.to_csv("gbc_start.csv", index=False)