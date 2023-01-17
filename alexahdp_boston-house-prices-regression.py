import numpy as np

import pandas as pd



from sklearn import datasets, linear_model

from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler
ds = datasets.load_boston()
data = ds.data

newdata = np.hstack((data, ds.target[:,None]))

df = pd.DataFrame(newdata, columns=[*ds.feature_names, 'target'])
df
# StandardScale ??
X_train, X_test, Y_train, Y_test = train_test_split(

    df[ds.feature_names],

    ds['target'],

    test_size=0.2,

    random_state=42

)
scaler = StandardScaler()

scaler.fit(X_train)



X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)
lr = linear_model.LinearRegression()

reg = lr.fit(X_train, Y_train)

Y_predict = reg.predict(X_test)
mse = mean_squared_error(Y_test, Y_predict)

R2 = r2_score(Y_test, Y_predict)



print(f'mse: {mse}, R2: {R2}')
regr = RandomForestRegressor(max_depth=7, random_state=42, n_estimators=200)
reg2 = regr.fit(X_train, Y_train)

Y2_predict = reg2.predict(X_test)
mse = mean_squared_error(Y_test, Y_predict)

R2 = r2_score(Y_test, Y_predict)



print(f'mse: {mse}, R2: {R2}')