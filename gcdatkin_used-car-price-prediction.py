import numpy as np

import pandas as pd



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



from sklearn.linear_model import LinearRegression

import lightgbm as lgb



from sklearn.metrics import mean_squared_error
data = pd.read_csv('../input/craigslist-carstrucks-data/vehicles.csv')
data
data.info()
data.isna().sum()
null_columns = data.columns[data.isna().mean() > 0.25]



data = data.drop(null_columns, axis=1)
data
unneeded_columns = ['id', 'url', 'region_url', 'image_url', 'description']



data = data.drop(unneeded_columns, axis=1)
data
{column: len(data[column].unique()) for column in data.columns if data.dtypes[column] == 'object'}
data = data.drop('model', axis=1)
def onehot_encode(df, columns, prefixes):

    df = df.copy()

    for column, prefix in zip(columns, prefixes):

        dummies = pd.get_dummies(df[column], prefix=prefix)

        df = pd.concat([df, dummies], axis=1)

        df = df.drop(column, axis=1)

    return df
data = onehot_encode(

    data,

    ['region', 'fuel', 'title_status', 'transmission', 'state'],

    ['reg', 'fuel', 'title', 'trans', 'state']

)
data
for column in data.columns:

    data[column] = data[column].fillna(data[column].mean())
data.isna().sum().sum()
y = data.loc[:, 'price']

X = data.drop('price', axis=1)
scaler = StandardScaler()



X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=34)
lin_model = LinearRegression()



lin_model.fit(X_train, y_train)



lin_y_preds = lin_model.predict(X_test)
lgb_model = lgb.LGBMRegressor(

    boosting_type='gbdt',

    num_leaves=31,

    n_estimators=100,

    reg_lambda=1.0

)



lgb_model.fit(X_train, y_train)



lgb_y_preds = lgb_model.predict(X_test)
lin_loss = np.sqrt(mean_squared_error(y_test, lin_y_preds))

lgb_loss = np.sqrt(mean_squared_error(y_test, lgb_y_preds))
print("Linear Regression RMSE:", lin_loss)

print("Gradient Boosted RMSE:", lgb_loss)
print("Linear Regression R^2 Score:", lin_model.score(X_test, y_test))

print("Gradient Boosted R^2 Score:", lgb_model.score(X_test, y_test))