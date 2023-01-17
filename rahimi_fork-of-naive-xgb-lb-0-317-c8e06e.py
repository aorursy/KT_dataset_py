import numpy as np

import pandas as pd

import xgboost as xgb

import matplotlib.pyplot as plt
df_train = pd.read_csv("../input/train.csv", parse_dates=['timestamp'])

df_test = pd.read_csv("../input/test.csv", parse_dates=['timestamp'])

df_macro = pd.read_csv("../input/macro.csv", parse_dates=['timestamp'])



df_train.head()
df_train['price_doc'].hist(bins=50)
y_train = df_train['price_doc'].values

id_test = df_test['id']



df_train.drop(['id', 'price_doc'], axis=1, inplace=True)

df_test.drop(['id'], axis=1, inplace=True)



# Build df_all = (df_train+df_test).join(df_macro)

num_train = len(df_train)

df_all = pd.concat([df_train, df_test])

df_all = df_all.join(df_macro, on='timestamp', rsuffix='_macro')

print(df_all.shape)



# Add month-year

month_year = (df_all.timestamp.dt.month + df_all.timestamp.dt.year * 100)

month_year_cnt_map = month_year.value_counts().to_dict()

df_all['month_year_cnt'] = month_year.map(month_year_cnt_map)



# Add week-year count

week_year = (df_all.timestamp.dt.weekofyear + df_all.timestamp.dt.year * 100)

week_year_cnt_map = week_year.value_counts().to_dict()

df_all['week_year_cnt'] = week_year.map(week_year_cnt_map)



# Add month and day-of-week

df_all['month'] = df_all.timestamp.dt.month

df_all['dow'] = df_all.timestamp.dt.dayofweek



# Other feature engineering

df_all['rel_floor'] = df_all['floor'] / df_all['max_floor'].astype(float)

df_all['rel_kitch_sq'] = df_all['kitch_sq'] / df_all['full_sq'].astype(float)



# Add apartment id (as suggested in https://www.kaggle.com/c/sberbank-russian-housing-market/discussion/33269)

# and replace it with its count and count by month

df_all['building_name'] = pd.factorize(df_all.sub_area + df_all['metro_km_avto'].astype(str))[0]



def add_time_features(col):

    col_month_year = pd.Series(pd.factorize(df_all[col].astype(str) + month_year.astype(str))[0])

    df_all[col + '_month_year_cnt'] = col_month_year.map(col_month_year.value_counts())



    col_week_year = pd.Series(pd.factorize(df_all[col].astype(str) + week_year.astype(str))[0])

    df_all[col + '_week_year_cnt'] = col_week_year.map(col_week_year.value_counts())

    

add_time_features('building_name')

add_time_features('sub_area')



# Remove timestamp column (may overfit the model in train)

df_all.drop(['timestamp', 'timestamp_macro'], axis=1, inplace=True)
# Deal with categorical values

df_numeric = df_all.select_dtypes(exclude=['object'])

df_obj = df_all.select_dtypes(include=['object']).copy()



for c in df_obj:

    df_obj[c] = pd.factorize(df_obj[c])[0]



df_values = pd.concat([df_numeric, df_obj], axis=1)
# Convert to numpy values

X_all = df_values.values

print(X_all.shape)



X_train = X_all[:num_train]

X_test = X_all[num_train:]



df_columns = df_values.columns
dtrain = xgb.DMatrix(X_train, y_train, feature_names=df_columns)

dtest = xgb.DMatrix(X_test, feature_names=df_columns)
from collections import Counter



class XGBCVHolder:

    """

    This is a hack to XGBoost, which does not provide an API to access 

    the models trained over xgb.cv

    """

    def __init__(self, features_names):

        self.models = []

        self.dtests = []

        self.called = False

        self.features_names = features_names



    def __call__(self, env):

        if not self.called:

            self.called = True

            for cvpack in env.cvfolds:

                self.models.append(cvpack.bst)

                self.dtests.append(cvpack.dtest)



    def predict_oof(self, ntree_limit=0):

        y = []

        y_hat = []

        for model, dtest in zip(self.models, self.dtests):

            y.extend(dtest.get_label())

            y_hat.extend(model.predict(dtest, ntree_limit=ntree_limit))



        return np.array(y), np.array(y_hat)



    def get_fscore(self):

        total = Counter()



        for m in self.models:

            fscore = m.get_fscore()

            fixed_features_names = { self.features_names[int(k[1:])] : v for k, v in fscore.iteritems() }

            total.update(fixed_features_names)



        return total
xgb_params = {

    'eta': 0.1,

    'max_depth': 5,

    'subsample': 0.7,

    'colsample_bytree': 0.7,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'silent': 1

}
# Uncomment to tune XGB `num_boost_rounds`

xgb_model = XGBCVHolder(df_columns)

cv_result = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,

    verbose_eval=10, show_stdv=False, callbacks=[xgb_model], seed=42)

cv_result[['train-rmse-mean', 'test-rmse-mean']].plot()

num_boost_rounds = len(cv_result)

print("num_boost_rounds:", num_boost_rounds)



#num_boost_round = 395
total = Counter()

fscore.items()



fig, axs = plt.subplots(1, len(xgb_model.models), figsize=(12, 8))



for m, ax in zip(xgb_model.models, axs):

    fscore = m.get_fscore()

    fixed_features_names = Counter({df_columns[int(k[1:])] : v for k, v in fscore.items() })

    total.update(fixed_features_names)



    r = dict(fixed_features_names.most_common(30))

    ax.barh(range(len(r)), r.values(), align='center')

    ax.set_yticks(range(len(r)))

    ax.set_yticklabels(r.keys())
fig, ax = plt.subplots(1, 1, figsize=(8, 16))

r = dict(fixed_features_names.most_common(30))

ax.barh(range(len(r)), r.values(), align='center')

ax.set_yticks(range(len(r)))

ax.set_yticklabels(r.keys())
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=250)
fig, ax = plt.subplots(1, 1, figsize=(8, 16))

xgb.plot_importance(model, max_num_features=50, height=0.5, ax=ax)
y_pred = model.predict(dtest)



df_sub = pd.DataFrame({'id': id_test, 'price_doc': y_pred})



df_sub.to_csv('sub.csv', index=False)