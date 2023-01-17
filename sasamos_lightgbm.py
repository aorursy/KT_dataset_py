import numpy as np

import pandas as pd

import lightgbm as lgb

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import LabelEncoder
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
df = pd.concat([train, test])

df.head()
for col in df.columns:

    df[col] = df[col].fillna(df[col].mode()[0])
le = LabelEncoder()



for col in (df.dtypes == object).index:

    if col != 'SalePrice':

        df[col] = le.fit_transform(df[col])
y_train = df.SalePrice[:train.shape[0]]

df.drop(['SalePrice', 'Id'], axis=1)



df_train = df.iloc[:train.shape[0], :]

df_test = df.iloc[train.shape[0]:, :]
scores = []



for i in range(10):

    print(i)

    lgb_model = lgb.LGBMRegressor(n_estimators=1000, 

                                  max_depth=15,

                                  learning_rate=.05, 

                                  random_state=i)

    X_tra, X_val, y_tra, y_val = train_test_split(df_train,

                                                  y_train,

                                                  test_size=0.33,

                                                  random_state=i)

    lgb_model.fit(

        X_tra, y_tra,

        eval_set=[(X_val, y_val)],

        early_stopping_rounds=200,

        verbose=200)

    scores.append(np.sqrt(mean_squared_error(np.nan_to_num(np.log(lgb_model.predict(X_val))),

                                             np.nan_to_num(np.log(y_val)))))
np_scores = np.array(scores)
sns.distplot(scores);
from matplotlib import pyplot

fig, ax = pyplot.subplots(figsize=(2, 10))

sns.barplot(lgb_model.feature_importances_, X_tra.columns);
lgb_model = lgb.LGBMRegressor(n_estimators=500, 

                              max_depth=15,

                              learning_rate=.05)

lgb_model.fit(df_train, y_train)
predicted_prices = lgb_model.predict(df_test)

my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})

my_submission.to_csv('submission.csv', index=False)