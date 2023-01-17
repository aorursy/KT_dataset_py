import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas import DataFrame



pd.set_option('display.max_rows', 100)

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"



from xgboost import XGBRegressor, plot_importance

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV

from sklearn.model_selection import cross_val_score, GridSearchCV



%matplotlib inline
from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



df_train_raw = pd.read_csv('../input/train.csv')

df_test_raw = pd.read_csv('../input/test.csv')

df_ss = pd.read_csv('../input/sample_submission.csv')

print(df_train_raw.shape)

print(df_test_raw.shape)
Y_train = df_train_raw.SalePrice

df_train = df_train_raw.drop(['Id', 'SalePrice'], axis=1)

df_test = df_test_raw.drop(['Id'], axis=1)

cnt = df_train.count()

dtypes = df_train.dtypes

fields = DataFrame({'cnt': cnt, 'dtype': dtypes})



# drop sparse fields

sparse_cols = fields[fields.cnt < 500].index

df_train = df_train.drop(sparse_cols, axis=1)



fields = fields.drop(sparse_cols)
# obj

fields_obj = fields[fields.dtype=='object']

df_train_obj = df_train[fields_obj.index]

df_test_obj = df_test[fields_obj.index]

fields_obj['nunique'] = df_train_obj.apply(lambda s: s.nunique())



# convert to categorical

for f in fields_obj.index:

    categories = df_train_obj[f].unique()

    df_train_obj[f] = df_train_obj[f].astype('category', categories=categories)

    df_test_obj[f] = df_test_obj[f].astype('category', categories=categories)
# num

fields_num = fields[fields.dtype != 'object']

df_train_num = df_train[fields_num.index]

df_test_num = df_test[fields_num.index]

fields_num = fields_num.join(df_train_num.describe().T)



#df_train_num = df_train_num.fillna(df_train_num.mean())
X_train = pd.get_dummies(df_train_obj,  dummy_na=False)

X_test = pd.get_dummies(df_test_obj,  dummy_na=False)



X_train = X_train.join(df_train_num)

X_test = X_test.join(df_test_num)
model = XGBRegressor()

#model = Ridge()

cv = 5



scores = cross_val_score(model, X_train, Y_train, cv=cv)

print("scores: %s" % scores)

print("Avg score: %f" % np.mean(scores))
model = XGBRegressor()

params = {

    'max_depth': [3],

    'n_estimators': [500, 200],

}



gs = GridSearchCV(model, params, cv=5)

gs.fit(X_train, Y_train)
gs.best_params_
model = XGBRegressor(max_depth=3, n_estimators=500)

model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)
df_re = DataFrame({

    "Id": df_test_raw["Id"],

    "SalePrice": Y_pred

})

df_re.to_csv('result.csv', index=False)



df_re.head()