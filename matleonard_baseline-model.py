

import pandas as pd

ks = pd.read_csv('../input/kickstarter-projects/ks-projects-201801.csv',

                 parse_dates=['deadline', 'launched'])

ks.head(6)
print('Unique values in `state` column:', list(ks.state.unique()))
# Drop live projects

ks = ks.query('state != "live"')



# Add outcome column, "successful" == 1, others are 0

ks = ks.assign(outcome=(ks['state'] == 'successful').astype(int))
ks = ks.assign(hour=ks.launched.dt.hour,

               day=ks.launched.dt.day,

               month=ks.launched.dt.month,

               year=ks.launched.dt.year)
from sklearn.preprocessing import LabelEncoder



cat_features = ['category', 'currency', 'country']

encoder = LabelEncoder()



# Apply the label encoder to each column

encoded = ks[cat_features].apply(encoder.fit_transform)
# Since ks and encoded have the same index and I can easily join them

data = ks[['goal', 'hour', 'day', 'month', 'year', 'outcome']].join(encoded)

data.head()
valid_fraction = 0.1

valid_size = int(len(data) * valid_fraction)



train = data[:-2 * valid_size]

valid = data[-2 * valid_size:-valid_size]

test = data[-valid_size:]
import lightgbm as lgb



feature_cols = train.columns.drop('outcome')



dtrain = lgb.Dataset(train[feature_cols], label=train['outcome'])

dvalid = lgb.Dataset(valid[feature_cols], label=valid['outcome'])



param = {'num_leaves': 64, 'objective': 'binary'}

param['metric'] = 'auc'

num_round = 1000

bst = lgb.train(param, dtrain, num_round, valid_sets=[dvalid], early_stopping_rounds=10, verbose_eval=False)
from sklearn import metrics

ypred = bst.predict(test[feature_cols])

score = metrics.roc_auc_score(test['outcome'], ypred)



print(f"Test AUC score: {score}")