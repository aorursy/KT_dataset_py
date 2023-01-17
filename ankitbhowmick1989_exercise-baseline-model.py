import pandas as pd



ks = pd.read_csv('../input/kickstarter-projects/ks-projects-201801.csv',parse_dates = ['deadline','launched'])



with pd.option_context('display.max_rows',None,'display.max_columns',None):

        print(ks.head(10))

ks.info(verbose=True)
pd.unique(ks['state'])
ks.groupby('state')['ID'].count()
# Drop Live projects



ks = ks.query('state != "live"')

## ks.groupby('state')['ID'].count()



ks = ks.assign(outcome = (ks['state'] == 'successful').astype(int))

ks.head(5)
ks = ks.assign(hour=ks.launched.dt.hour,

               day=ks.launched.dt.day,

               month=ks.launched.dt.month,

               year=ks.launched.dt.year)



ks.head()
from sklearn.preprocessing import LabelEncoder



cat_features = ['category','currency','country']

encoder = LabelEncoder()



#Apply the label encoder to each features

encoded = ks[cat_features].apply(encoder.fit_transform)

encoded.head(10)
# Since ks and encoded have the same index and I can easily join them

data = ks[['goal', 'hour', 'day', 'month', 'year', 'outcome']].join(encoded)

print(data.head())

data.shape
valid_fraction = 0.1

valid_size = int(len(data) * valid_fraction)

print(valid_size)



train = data[:-2 * valid_size]

print(train.head(5))

print(train.shape)

valid = data[-2 * valid_size:-valid_size]

print(valid.head(5))

print(valid.shape)

test = data[-valid_size:]

print(test.head(5))

print(test.shape)
# list = [1,2,3,4,5,6,8]

# print(list[:-2 * 1])

# print(list[:-2 * 2:-2])

# print(list[-2:])
for each in [train, valid, test]:

    print(f"Outcome fraction = {each.outcome.mean():.4f}")
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