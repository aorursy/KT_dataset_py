import pandas as pd



from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb

from sklearn.metrics import accuracy_score, roc_auc_score
ks = pd.read_csv('../input/kickstarter-projects/ks-projects-201801.csv', parse_dates = ['deadline', 'launched'])

ks.head()
# array of unique states

pd.unique(ks.state)
# group and count states

ks.groupby('state')['ID'].count()
# select only the projects that are not live

ks = ks.query('state != "live"')
# target as successful or not

ks = ks.assign(outcome = (ks['state'] == 'successful').astype(int))
# converting time stamp

ks = ks.assign(hour = ks.launched.dt.hour, 

               day=ks.launched.dt.day,

               month=ks.launched.dt.month,

               year=ks.launched.dt.year)

ks.head()
cat_features = ['category', 'currency', 'country']

ks[cat_features].head()
# label encoding



encoder = LabelEncoder()



encoded = ks[cat_features].apply(encoder.fit_transform)

encoded.head()
data = ks[['goal', 'hour', 'day', 'month', 'year', 'outcome']].join(encoded)

data.head()
valid_fraction = 0.1

valid_size = int(len(data) * valid_fraction)



train = data[:-2*valid_size]

valid = data[-2*valid_size:-valid_size]

test = data[-valid_size:]
for i in [train, valid, test]:

    print(f"Outcome fraction = {i.outcome.mean():.4f}")
feature_cols = train.columns.drop('outcome')



dtrain = lgb.Dataset(train[feature_cols], label=train['outcome'])

dvalid = lgb.Dataset(valid[feature_cols], label=valid['outcome'])



param = {'num_leaves':64, 'objective':'binary', 'metric':'auc'}

num_round=1000



bst = lgb.train(param, dtrain, num_round, valid_sets=[dvalid], early_stopping_rounds=10, verbose_eval=False)
ypred = bst.predict(test[feature_cols])

score = roc_auc_score(test['outcome'], ypred)



print(score)