import numpy as np

import pandas as pd



import os

print(os.listdir("../input"))
date_format = '%Y-%m-%dT%H:%M:%S.%fZ'



# TODO: read timestamp as pandas.Timestamp object



# Load Datasets

clicks_df=pd.read_csv('../input/yoochoose-data/yoochoose-clicks.dat',

                      names=['session_id','timestamp','item_id','category'],

                      dtype={'category': str})

clicks_df['timestamp'] = pd.to_datetime(clicks_df['timestamp'])



display("Clicks Data",)

display(clicks_df.head())



buys_df = pd.read_csv('../input/yoochoose-data/yoochoose-buys.dat', 

                      names=['session_id', 'timestamp', 'item_id', 'price', 'quantity'],

                      )

buys_df['timestamp'] = pd.to_datetime(buys_df['timestamp'])



display("Buys Data",)

display(buys_df.head())



buys_df.info()
# Category – the context of the click. 

# The value "S" indicates a special offer,

# "0" indicates  a missing value, 

# a number between 1 to 12 indicates a real category identifier,

# any other number indicates a brand.

#   - if an item has been clicked in the context of a special offer then the value will be "S", 

#   - if the context was a brand 

#              eg. BOSCH, then the value will be an 8-10 digits number.

#   - If the item has been clicked under regular category, 

#              eg. sport, then the value will be a number between 1 to 12. 
# Clicks data has some missing values for category

# clicks_df[clicks_df.category==0].head()

print("Number of items with missing category info:", 

      len(np.unique(clicks_df[clicks_df.category=="0"].item_id)))



# Buys data has some missing values for price AND quantity

# buys_df[buys_df.price==0].head()

print("Number of missing price and qty entries in buys data:", 

      len(buys_df[buys_df.price==0]), len(buys_df[buys_df.quantity==0]))

print("The training dataset has", len(clicks_df), "clicks", 

      "from", len(np.unique(clicks_df.session_id)), "sessions")

print("There are ",len(buys_df), "purchases", 

      "from", len(np.unique(buys_df.session_id)), "sessions\n",

      "involving", len(np.unique(buys_df.item_id)), "unique items",

      "out of", len(np.unique(clicks_df.item_id)), "items in the whole training set.")



print("\nThis means that the sessions with/without purchases are highly imbalanced.")

print("Number of sessions with purchases", len(np.unique(buys_df.session_id)))

print("Number of sessions without purchases", len(np.unique(clicks_df.session_id)) - len(np.unique(buys_df.session_id)))

print("Difference:", len(np.unique(clicks_df.session_id)) - 2 * len(np.unique(buys_df.session_id)))
from sklearn.ensemble import RandomForestClassifier
# TODO: Handle class imbalance by downsampling non buy data from clicks_df

# How to do downsampling:-

# Take all entries from buys data

# Take same number of samples from clicks data but make sure that they belong to non buy category





# FIXME: This is a JUGAAD to prevent memory error

clicks_df = clicks_df.head(1000)

buys_df = buys_df.head(1000)
# temporary

clicks_df['buy'] = 0

buys_df['buy'] = 1



union_df = pd.concat([clicks_df, buys_df], ignore_index=True, sort=True).sort_values(by=['session_id','item_id'])

union_df['category'] = union_df['category'].fillna(method='ffill')

union_df.head()

union_df.info()
from sklearn.model_selection import train_test_split



# TODO: one-hot encode categorical features

# for col in union_df.dtypes[union_df.dtypes == 'object'].index:

#     for_dummy = union_df.pop(col)

#     union_df = pd.concat([union_df, pd.get_dummies(for_dummy, prefix=col)], axis=1)



# FIXME: using sparse=TRUE prevents MEMORY ERROR. find RCA???

one_hot = pd.get_dummies(union_df['category'], sparse=True)

union_df = union_df.drop('category', axis=1)

union_df = union_df.join(one_hot)



one_hot = pd.get_dummies(union_df['timestamp'], sparse=True)

union_df = union_df.drop('timestamp', axis=1)

union_df = union_df.join(one_hot)
union_df = union_df.drop(['price', 'quantity'], axis=1)



X = union_df.drop(['buy'], axis=1)

y = union_df[['buy']]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

x_train.shape, x_test.shape, y_train.shape, y_test.shape
union_df.describe()
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

rf.fit(x_train, y_train.values.ravel())



y_pred = rf.predict(x_test)

from sklearn.metrics import roc_curve, auc, accuracy_score

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)

roc_auc = auc(false_positive_rate, true_positive_rate)

roc_auc
accuracy_score(y_pred,y_test)