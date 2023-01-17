import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LinearRegression, Ridge, Lasso, HuberRegressor

from sklearn import metrics

from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import SelectKBest, chi2

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import explained_variance_score

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from tensorflow.keras.callbacks import EarlyStopping
import xgboost
from xgboost import XGBRegressor, train
from sklearn.pipeline import Pipeline

from sklearn.feature_selection import SelectKBest
df = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
df
df.drop(['id','host_id', 'host_name', 'last_review'], axis=1, inplace=True)
for column in df.columns:

    print(f"{column[0].upper()}{column[1:]}: {len(df[column].unique())}")
df.drop(['name'], axis=1, inplace=True)
print(f"Number of Features: {len(df.columns)}")
plt.figure(figsize=(16,6))

sns.heatmap(df.isnull(), cbar=False, yticklabels=False)
df2 = df   #Saving original DataFrame for future use
df2 = pd.get_dummies(df, drop_first=True)

df2.head(5)
df2 = df2.fillna(0)
columns = ['latitude', 

           'longitude',

           'minimum_nights',

           'number_of_reviews',

           'reviews_per_month', 

           'calculated_host_listings_count',

           'availability_365']
scaler = StandardScaler()

scaler.fit(df2[columns])

df2[columns]=pd.DataFrame(scaler.transform(df2[columns]), columns=columns)
plt.figure(figsize=(30, 6))

sns.boxplot(df['price'])
plt.figure(figsize=(30, 6))

sns.boxplot(df[df['price']<400]['price'])
sns.boxplot(df['reviews_per_month'])
sns.boxplot(df[df['reviews_per_month']<5]['reviews_per_month'])
df2
X = df2[df2['price']<400]

X = X[X['reviews_per_month']<5]

y = X['price']

X.drop(['price'], inplace=True, axis=1)
sns.jointplot(x='longitude', y='price', data=df[df['price']<400], kind='kde')
sns.jointplot(x='latitude', y='price', data=df[df['price']<400], color='red', kind='kde')
plt.figure(figsize=(12, 6))

sns.countplot(x='neighbourhood_group', hue='room_type', data=df, alpha=0.7, palette='Blues_d')
plt.figure(figsize=(12, 6))

sns.boxplot(x='neighbourhood_group', y='price', hue='room_type', data=df[df['price']<400])
plt.figure(figsize=(12, 6))

sns.set_context('paper')

sns.scatterplot(df['longitude'], df['latitude'], hue=df['neighbourhood_group'])
plt.figure(figsize=(12, 6))

sns.heatmap(df.corr(), annot=True)
X_Train, X_CV, y_train, y_cv = train_test_split(X, y, test_size=0.4)

X_CV, X_Test, y_cv, y_test = train_test_split(X_CV, y_cv, test_size=0.5)
linear_reg = LinearRegression()

linear_reg.fit(X_Train, y_train)
linear_reg_pred = linear_reg.predict(X_Train)
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_train, linear_reg_pred)))

print('Variance Score: ', explained_variance_score(y_train, linear_reg_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, linear_reg.predict(X_Test))))

print('Variance Score: ', explained_variance_score(y_test, linear_reg.predict(X_Test)))
ann = Sequential()

ann.add(Dense(233, activation='relu'))

ann.add(Dense(115, activation='relu'))

ann.add(Dense(60, activation='relu'))

ann.add(Dense(30, activation='relu'))

ann.add(Dense(15, activation='relu'))

ann.add(Dense(1))

ann.compile(optimizer='Adam', loss='mse')
early_stop2=EarlyStopping(monitor='val_loss', mode='min', patience=10)
ann.fit(x=np.array(X_Train), 

          y=np.array(y_train), 

          epochs=500,

          verbose=1,

          validation_data=(np.array(X_CV), np.array(y_cv)), 

          callbacks=[early_stop2])
error_ann = pd.DataFrame(ann.history.history)
error_ann.plot()
print('*'*10, 'Performance on Training Set', '*'*10)

print('\n')

print(f'Explained Variance Score: {explained_variance_score(y_train, ann.predict(X_Train))}')

print(f'RMSE: {np.sqrt(metrics.mean_squared_error(y_train, ann.predict(X_Train)))}')

print(f'R2 Score: {metrics.r2_score(y_train, ann.predict(X_Train))}')

print('\n')

print('*'*10, 'Performance on Cross-Validation Set', '*'*10)

print('\n')

print(f'Explained Variance Score: {explained_variance_score(y_cv, ann.predict(X_CV))}')

print(f'RMSE: {np.sqrt(metrics.mean_squared_error(y_cv, ann.predict(X_CV)))}')

print(f'R2 Score: {metrics.r2_score(y_cv, ann.predict(X_CV))}')
print(f'Explained Variance Score: {explained_variance_score(y_test, ann.predict(X_Test))}')

print(f'RMSE: {np.sqrt(metrics.mean_squared_error(y_test, ann.predict(X_Test)))}')

print(f'R2 Score: {metrics.r2_score(y_test, ann.predict(X_Test))}')
gs_rr = GridSearchCV(Ridge(), 

                    param_grid={

                        'alpha':[1, 2, 3, 4, 5, 6]

                    }, 

                    verbose=3)
gs_rr.fit(X_Train, y_train)
rr = gs_rr.best_estimator_
print(f'Explained Variance Score: {explained_variance_score(y_train, rr.predict(X_Train))}')

print(f'RMSE: {np.sqrt(metrics.mean_squared_error(y_train, rr.predict(X_Train)))}')

print(f'R2 Score: {metrics.r2_score(y_train, rr.predict(X_Train))}')
print(f'Explained Variance Score: {explained_variance_score(y_test, rr.predict(X_Test))}')

print(f'RMSE: {np.sqrt(metrics.mean_squared_error(y_test, rr.predict(X_Test)))}')

print(f'R2 Score: {metrics.r2_score(y_test, rr.predict(X_Test))}')
gs_lr = GridSearchCV(Lasso(), 

                    param_grid={

                        'alpha':[0.1, 0.3, 1, 2, 3, 4, 5, 6]

                    }, 

                    verbose=3)
gs_lr.fit(X_Train, y_train)
lasso = gs_lr.best_estimator_
print(f'Explained Variance Score: {explained_variance_score(y_train, lasso.predict(X_Train))}')

print(f'RMSE: {np.sqrt(metrics.mean_squared_error(y_train, lasso.predict(X_Train)))}')

print(f'R2 Score: {metrics.r2_score(y_train, lasso.predict(X_Train))}')
print(f'Explained Variance Score: {explained_variance_score(y_test, lasso.predict(X_Test))}')

print(f'RMSE: {np.sqrt(metrics.mean_squared_error(y_test, lasso.predict(X_Test)))}')

print(f'R2 Score: {metrics.r2_score(y_test, lasso.predict(X_Test))}')
#Run this code for finding optimal parameters.



#gs_xgb = GridSearchCV(XGBRegressor(booster='gbtree', subsample=0.75),

#                     param_grid={

#                         'min_child_weight': [4, 6, 8],

#                         'max_depth': [8, 10, 12],

#                         'eta': [0.3, 0.03], 

#                         'learning_rate': [0.01, 0.1],

#                         'reg_alpha': [0.1, 1, 3],

#                         'reg_lambda': [0.1, 1, 2, 3]

#                     }, 

#                     verbose=3, 

#                     cv=3)

#gs_xgb.fit(X_Train, y_train)
xgb = XGBRegressor(eta=0.3,

                    learning_rate=0.1,

                    max_depth=8,

                    min_child_weight=6,

                    reg_alpha=3,

                    reg_lambda=3

                   )



xgb.fit(X_Train, y_train)
print(f'Explained Variance Score: {explained_variance_score(y_train, xgb.predict(X_Train))}')

print(f'RMSE: {np.sqrt(metrics.mean_squared_error(y_train, xgb.predict(X_Train)))}')

print(f'R2 Score: {metrics.r2_score(y_train, xgb.predict(X_Train))}')
print(f'Explained Variance Score: {explained_variance_score(y_test, xgb.predict(X_Test))}')

print(f'RMSE: {np.sqrt(metrics.mean_squared_error(y_test, xgb.predict(X_Test)))}')

print(f'R2 Score: {metrics.r2_score(y_test, xgb.predict(X_Test))}')
gs_hr = GridSearchCV(HuberRegressor(max_iter=1500),

                    param_grid={

                        'epsilon': [1, 1.25, 1.5, 1.75, 2]

                    }, 

                    verbose=3)
gs_hr.fit(X_Train, y_train)
huber = gs_hr.best_estimator_
print(f'Explained Variance Score: {explained_variance_score(y_train, huber.predict(X_Train))}')

print(f'RMSE: {np.sqrt(metrics.mean_squared_error(y_train, huber.predict(X_Train)))}')

print(f'R2 Score: {metrics.r2_score(y_train, huber.predict(X_Train))}')
print(f'Explained Variance Score: {explained_variance_score(y_test, huber.predict(X_Test))}')

print(f'RMSE: {np.sqrt(metrics.mean_squared_error(y_test, huber.predict(X_Test)))}')

print(f'R2 Score: {metrics.r2_score(y_test, huber.predict(X_Test))}')
gs_rfr = GridSearchCV(RandomForestRegressor(),

                      param_grid={

                          'n_estimators': [100, 125, 150, 175, 200, 225, 250],

                          'min_samples_leaf': [2, 4, 6, 8],

                          'min_samples_leaf': [1, 2, 4, 6],

                          'max_depth': [int(x) for x in np.linspace(10, 150, 8)],

                          'max_features': ['auto', 'sqrt']

                      }, 

                      verbose=3,

                      scoring=metrics.make_scorer(metrics.r2_score),

                      cv=3

                     )



gs_rfr.fit(X_Train, y_train)

gs_rfr.best_params_