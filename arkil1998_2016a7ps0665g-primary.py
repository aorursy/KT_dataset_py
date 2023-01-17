import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import random

%matplotlib inline



from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import KFold

from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error
train_df, test_df = pd.read_csv("../input/train.csv"), pd.read_csv("../input/test.csv")

orig_test = pd.read_csv("../input/test.csv")
train_df.info()
test_df.info()
# Compute the correlation matrix

corr = train_df.corr()



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, cmap=cmap, vmax=1, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)



plt.show()
train_df_1 = train_df.drop(['type', 'year','XLarge Bags', 'Total Bags', 'Total Volume'],1)

test_df_1 = test_df.drop(['type', 'year', 'XLarge Bags', 'Total Bags', 'Total Volume'],1)
X = train_df_1.drop(['AveragePrice'],1)

y = train_df_1['AveragePrice']
xgb = XGBRegressor(n_estimators=5000, learning_rate=0.01, gamma=0, subsample=0.5, reg_lambda = 2,

                           colsample_bytree=1, max_depth=19, n_jobs=4, random_state=42)



kf = KFold(n_splits=5, shuffle=False, random_state=42)



error_sum = 0



for train, test in kf.split(X):

    X_train, X_val = X.loc[train], X.loc[test]

    y_train, y_val = y[train], y[test]

    

    scaler1 = MinMaxScaler()

    scaled_X_train = scaler1.fit_transform(X_train)

    scaled_X_train = pd.DataFrame(scaled_X_train)

    scaled_X_val = scaler1.transform(X_val)

    scaled_X_val = pd.DataFrame(scaled_X_val)

    

    xgb.fit(scaled_X_train, y_train)

    pred = xgb.predict(scaled_X_val)

    error_sum += mean_squared_error(pred, y_val)

    

print(error_sum/5)
scaler2 = MinMaxScaler()

scaled_X_train_full = scaler2.fit_transform(X)

scaled_X_train_full = pd.DataFrame(scaled_X_train_full)

scaled_X_test = scaler2.transform(test_df_1)

scaled_X_test = pd.DataFrame(scaled_X_test)



xgb.fit(scaled_X_train_full, y)

pred = xgb.predict(scaled_X_test)
res1 = pd.DataFrame(pred)

final = pd.concat([orig_test["id"], res1], axis=1).reindex()

final = final.rename(columns={0: "AveragePrice"})

final.to_csv('sub11.csv', index = False)