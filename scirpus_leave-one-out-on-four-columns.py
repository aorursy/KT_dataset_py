import numpy as np

import pandas as pd

import xgboost as xgb

from sklearn.preprocessing import LabelEncoder
def LeaveOneOut(data1, data2, columnName, useLOO=False, addNoise=False):

    grpOutcomes = data1.groupby(columnName)['interest_level'].mean().reset_index()

    grpCount = data1.groupby(columnName)['interest_level'].count().reset_index()

    grpOutcomes['cnt'] = grpCount.interest_level

    if(useLOO):

        grpOutcomes = grpOutcomes[grpOutcomes.cnt > 4]

    grpOutcomes.drop('cnt', inplace=True, axis=1)

    outcomes = data2['interest_level'].values

    x = pd.merge(data2[[columnName, 'interest_level']], grpOutcomes,

                 suffixes=('x_', ''),

                 how='left',

                 on=columnName,

                 left_index=True)['interest_level']

    if(useLOO):

        x = ((x*x.shape[0])-outcomes)/(x.shape[0]-1)

        if(addNoise):

            x = x + np.random.normal(0, .25, x.shape[0])

    return x.fillna(x.mean())



def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=0, num_rounds=1000):

    param = {}

    param['objective'] = 'multi:softprob'

    param['eta'] = 0.03

    param['max_depth'] = 4

    param['silent'] = 1

    param['num_class'] = 3

    param['eval_metric'] = "mlogloss"

    param['min_child_weight'] = 1

    param['subsample'] = 0.7

    param['colsample_bytree'] = 0.6

    param['seed'] = seed_val

    param['seed'] = seed_val

    num_rounds = num_rounds



    plst = list(param.items())

    xgtrain = xgb.DMatrix(train_X, label=train_y)



    if test_y is not None:

        xgtest = xgb.DMatrix(test_X, label=test_y)

        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]

        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=20)

    else:

        xgtest = xgb.DMatrix(test_X)

        model = xgb.train(plst, xgtrain, num_rounds)



    pred_test_y = model.predict(xgtest)

    return pred_test_y, model
labels = ['building_id','display_address','street_address','manager_id']

train = pd.read_json('../input/train.json')

test = pd.read_json('../input/test.json')

train.loc[train.interest_level=='low','interest_level'] = 0

train.loc[train.interest_level=='medium','interest_level'] = 1

train.loc[train.interest_level=='high','interest_level'] = 2

train.interest_level = train.interest_level.astype(float)

test['interest_level'] = -1
for f in labels:

    lbl = LabelEncoder()

    lbl.fit(list(train[f].values) + list(test[f].values))

    train[f] = lbl.transform(train[f].values).astype(int)

    test[f] = lbl.transform(test[f].values).astype(int)
actualcolumns = []

for col in labels: 

    for lvl in range(3):

        kftrain = train.copy()

        kftrain.interest_level = (kftrain.interest_level==lvl).astype(int)

        train['loo_'+col+'_'+str(lvl)] = LeaveOneOut(kftrain.copy(),

                                                     train.copy(),

                                                     col, True, True).values

        test['loo_'+col+'_'+str(lvl)] = LeaveOneOut(kftrain.copy(),

                                                    test.copy(),

                                                    col, True, False).values

        actualcolumns.extend(['loo_'+col+'_'+str(lvl)])
preds, model = runXGB(train[actualcolumns],

                      train.interest_level,

                      test[actualcolumns],

                      None,

                      feature_names=None,

                      seed_val=0,

                      num_rounds=500)
out_df = pd.DataFrame(preds)

out_df.columns = ["low", "medium", "high" ]

out_df["listing_id"] = test.listing_id.values

out_df = out_df[['high', 'medium', 'low','listing_id']]

out_df.to_csv("loo_xgb_starter.csv", index=False)
print(out_df[['low','medium','high']].mean())