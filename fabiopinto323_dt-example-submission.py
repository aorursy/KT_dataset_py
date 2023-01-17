import numpy as np

import pandas as pd



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train_v2.csv")

test = pd.read_csv("../input/test_v2.csv")

train.head()
def profile_feature(train, test, id_feature, datetime_feature, feature_to_groupby, feature_to_agg, window, agg_function):

    

    # create index based on id

    train.index = train[id_feature]

    test.index = test[id_feature]

    

    # concat train and test set in order to create profile

    data = pd.concat([train, test], axis=0)

    data = data.sort_values(datetime_feature)

    

    # create mask to split again in the end

    train_idx = np.in1d(data.index, train.index)

    test_idx = np.in1d(data.index, test.index)

    

    # do profile computation

    data.index = pd.to_datetime(data[datetime_feature], unit='ms')

    feature = data.groupby(feature_to_groupby)[feature_to_agg].rolling(window=window).apply(agg_function)

    feature = feature.reset_index(level=[0,1]).sort_values(datetime_feature)[feature_to_agg].values

    

    return feature[train_idx], feature[test_idx]
# feature_train, feature_test = profile_feature(

#    train=train, 

#    test=test,

#    id_feature='id',

#    datetime_feature='timestamp', 

#    feature_to_groupby='card_id', 

#    feature_to_agg='amount',

#    window='6H',

#    agg_function=np.mean

# )
from sklearn.tree import DecisionTreeClassifier



clf = DecisionTreeClassifier()

clf.fit(X=train[['C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'amount']], y=train['isfraud'])
preds = clf.predict_proba(test[['C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'amount']])

preds = [ x[1] for x in preds ]

preds = pd.concat([test['id'],pd.Series(preds)], axis=1)

preds.columns = [['id', 'isfraud']]
preds.to_csv("submission_dt_baseline_v2.csv", index=None)