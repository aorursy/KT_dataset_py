import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn import model_selection, preprocessing
import xgboost as xgb
import datetime
from sklearn.model_selection import train_test_split

# df_train = pd.read_csv('../input/train.csv', parse_dates=['timestamp'])
# df_test = pd.read_csv('../input/test.csv', parse_dates=['timestamp'])

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head(10)
print(train.describe())
train.timestamp.describe()
train.info()
train.sub_area.unique()
train.product_type.unique()
train.groupby('sub_area').sub_area.count()
train.groupby('ecology').sub_area.count()
train.groupby('state').sub_area.count()
Data_stat=train.groupby('state').sub_area.count().sort_values()
Data_stat
type(Data_stat)
train.dtypes
train.price_doc.dtype
print(train.columns)
train=train.fillna(-999)
pd.isnull(train).any()
pd.isnull(test).any()
test=test.fillna(-999)
y_train = train["price_doc"]
x_train = train.drop(["id", "price_doc","timestamp" ], axis=1)
x_test = test.drop(["id","timestamp"], axis=1)
id_test = test['id']
# pd.concat([___, ____])
#split
numeric_columns= x_train.columns[x_train.dtypes!='object']
numeric_columns
x_train.dtypes.value_counts()
int64_columns= x_train.columns[x_train.dtypes=='int64']
int64_columns
x_train['sub_area']
for c in x_train.columns:
    if x_train[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_train[c].values)) 
        x_train[c] = lbl.transform(list(x_train[c].values))
        #x_train.drop(c,axis=1,inplace=True)
x_train['sub_area']
for c in x_test.columns:
    if x_test[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_test[c].values)) 
        x_test[c] = lbl.transform(list(x_test[c].values))
        #x_test.drop(c,axis=1,inplace=True) 
xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)

# fig, ax = plt.subplots(1, 1, figsize=(8, 13))
# xgb.plot_importance(model, max_num_features=50, height=0.5, ax=ax)
# cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,
#     verbose_eval=50, show_stdv=False)
# cv_output[['train-rmse-mean', 'test-rmse-mean']].plot()
# num_boost_rounds = len(cv_output)
# print( num_boost_rounds )
num_boost_rounds = 384
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round= num_boost_rounds)
y_predict = model.predict(dtest)
output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})
output.head()


output.to_csv('xgbSb.csv', index=False)

pd.read_csv('../output/xgbSb.csv')