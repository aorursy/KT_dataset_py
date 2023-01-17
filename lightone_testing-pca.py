import numpy as np 

import pandas as pd 

with pd.HDFStore("../input/train.h5", "r") as train:

    odf = train.get("train")

print("Train shape: {}".format(df.shape))

df = odf
from sklearn.model_selection import train_test_split

# Values from top public kernel https://www.kaggle.com/bguberfain/two-sigma-financial-modeling/univariate-model-with-clip/run/482189

low_y_cut = -0.086092

high_y_cut = 0.093496



print("Preparing data for model...")

excl = ["id", "timestamp"]

col = [c for c in df.columns if c not in excl]

df = df.sample(frac=0.1)

d_mean= df.median(axis=0)



df = df[col]

n = df.isnull().sum(axis=1)

for c in df.columns:

    df[c + '_nan_'] = pd.isnull(df[c])

    d_mean[c + '_nan_'] = 0

df.fillna(d_mean, inplace=True)

df['znull'] = n

n = []



y_is_within_cut = ((df['y'] > low_y_cut) & (df['y'] < high_y_cut))

train_X = df.loc[y_is_within_cut, df.columns[2:-1]]

train_y = df.loc[y_is_within_cut, 'y'].values.reshape(-1, 1)

X_tr, X_val, y_tr, y_val = train_test_split(train_X, train_y, random_state = 3)

print("Data for model: X={}, y={}".format(train_X.shape, train_y.shape))
from sklearn.decomposition import PCA

model_pca = PCA(whiten=True)

print("Fitting...")

model_pca.fit(X_tr)

print("Fitting done")

print(model_pca.explained_variance_ratio_)

variance = pd.DataFrame(model_pca.explained_variance_ratio_)

np.cumsum(model_pca.explained_variance_ratio_)



model_pca = PCA(n_components=3,whiten=True)

model_pca.fit(X_tr)

X_tr_pca = model_pca.transform(X_tr)

X_val_pca = model_pca.transform(X_val)

print(X_tr_pca)
y_tr_pca = model_pca.transform(y_tr)

y_val_pca = model_pca.transform(y_val)
import xgboost as xgb

model_xgb = xgb.XGBRegressor()

print("Fitting...")

model_xgb.fit(X_tr, y_tr)

print("Fitting done")
model_xgb_pca = xgb.XGBRegressor()

print("Fitting...")

model_xgb_pca.fit(X_tr, y_tr)

print("Fitting done")
from sklearn.metrics import r2_score

y_pred = model_xgb.predict(X_val)

predictions = [round(value) for value in y_pred]

# evaluate predictions

r2 = r2_score(y_val, predictions)

print("XGBoost with raw dataset " + str(mae))
y_pred = model_xgb_pca.predict(X_val_pca)

predictions = [round(value) for value in y_pred]

# evaluate predictions

r2 = r2_score(y_val, predictions)

print("XGBoost with PCA dataset " + str(r2))
odf[col].fillna(d_mean).loc[y_is_within_cut, 'technical_20'].values
from sklearn.ensemble import ExtraTreesRegressor

rfr = ExtraTreesRegressor(n_estimators=100, max_depth=4, n_jobs=-1, random_state=17, verbose=0)

model1 = rfr.fit(X_tr, y_tr)

y_pred = model1.predict(X_val)

prediction = [round(value) for value in y_pred]

# evaluate predictions

r2 = r2_score(y_val, prediction)

print("ExtraTreeRegressor model " + str(r2))
from sklearn.linear_model import LinearRegression

model2 = LinearRegression(n_jobs=-1)

model2.fit(X_tr, y_tr)
y_pred = model2.predict(X_val)

# evaluate predictions

r2 = r2_score(y_val, y_pred)

print("Linear model " + str(r2))