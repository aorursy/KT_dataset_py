import numpy as np 
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, HuberRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from xgboost import XGBRegressor, train
df = pd.read_csv('/kaggle/input/life-expectancy-who/Life Expectancy Data.csv')
df.head(5)
print("\t     Unique Values\n\n\n")
print(df.nunique())
df = df.drop(['Country'], axis=1)
df.describe()
print(df.info())
plt.figure(figsize=(12, 6))
plt.tight_layout()
sns.heatmap(df.isnull())
df.columns
df.isnull().sum()
df = df.dropna(subset=['Life expectancy ', 'Adult Mortality', 'Alcohol', ' BMI ', 'Diphtheria ', 
                       ' thinness  1-19 years', ' thinness 5-9 years', 'Polio'])
df.isnull().sum()
given_values = df.dropna()
missing_values_index = list(set(df.index) - set(given_values.index))
missing_values = df.loc[missing_values_index]
df.info()
imputer = KNNImputer(n_neighbors=2)
imputed_values = pd.DataFrame(imputer.fit_transform(df.drop(['Status'], axis=1)), columns=df.drop(['Status'], axis=1).columns)
imputed_values['Status'] = df['Status']
plt.figure(figsize=(20, 10))
sns.heatmap(imputed_values.corr(), annot=True)
df2 = pd.DataFrame(pd.get_dummies(data=imputed_values, columns=['Status']))
plt.figure(figsize=(12, 6))
sns.set_style('darkgrid')
sns.scatterplot(data=df, y='Life expectancy ', x='Adult Mortality', hue='Status')
plt.figure(figsize=(15, 8))
sns.boxplot(df['Year'], df['Life expectancy '], hue=df['Status'], palette="coolwarm")
fig, axes = plt.subplots(4, 1, figsize=(20, 40))
sns.scatterplot(data=df, y='Life expectancy ', x='Alcohol', hue='Status', palette="mako_r", alpha=0.5, ax=axes[0])
sns.scatterplot(data=df, y='Life expectancy ', x=' HIV/AIDS', hue='Status', palette="mako_r", alpha=0.5, ax=axes[1])
sns.scatterplot(data=df, y='Life expectancy ', x='Polio', hue='Status', palette="mako_r", alpha=0.5, ax=axes[2])
sns.scatterplot(data=df, y='Life expectancy ', x='Diphtheria ', hue='Status', palette="mako_r", alpha=0.5, ax=axes[3])
fig, axes = plt.subplots(1, 2, figsize=(30, 10))
sns.scatterplot(data=df, y='Life expectancy ', x='Schooling', hue='Status', palette="OrRd_r", alpha=0.5, ax=axes[0])
sns.scatterplot(data=df, y='Life expectancy ', x='Income composition of resources', hue='Status', palette="OrRd_r", alpha=0.5, ax=axes[1])
df2.corr()['Life expectancy ']
columns_to_drop = []
for col in df.drop(['Status'], axis=1).columns:
    temp = df.corr()[col].loc['Life expectancy ']
    if temp < 0.35 and temp > -0.2:
        columns_to_drop.append(col)
df2 = df2.drop(columns_to_drop, axis=1)
y = df2['Life expectancy ']
X = df2.drop(['Life expectancy '], axis=1)
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_Train, X_CV, y_train, y_cv = train_test_split(X, y, test_size=0.4)
X_Test, X_CV, y_test, y_cv = train_test_split(X_CV, y_cv, test_size=0.5)
lr = LinearRegression()
lr.fit(X_Train, y_train)
print(f"Explained Variance Score: {explained_variance_score(y_pred=lr.predict(X_Train), y_true=y_train)}")
print(f"RMSE: {np.sqrt(metrics.mean_squared_error(y_pred=lr.predict(X_Train), y_true=y_train))}")
print(f"Explained Variance Score: {explained_variance_score(y_pred=lr.predict(X_CV), y_true=y_cv)}")
print(f"RMSE: {np.sqrt(metrics.mean_squared_error(y_pred=lr.predict(X_CV), y_true=y_cv))}")
gs_rr = GridSearchCV(Ridge(),
                    param_grid={
                        'alpha':[0.1, 0.3, 1, 3, 6, 8, 10]
                    }, verbose=1)

gs_rr.fit(X_Train, y_train)
rr=gs_rr.best_estimator_
print(f"Explained Variance Score: {explained_variance_score(y_pred=rr.predict(X_Train), y_true=y_train)}")
print(f"RMSE: {np.sqrt(metrics.mean_squared_error(y_pred=rr.predict(X_Train), y_true=y_train))}")
print(f"Explained Variance Score: {explained_variance_score(y_pred=rr.predict(X_CV), y_true=y_cv)}")
print(f"RMSE: {np.sqrt(metrics.mean_squared_error(y_pred=rr.predict(X_CV), y_true=y_cv))}")
gs_lr = GridSearchCV(Lasso(),
                    param_grid={
                        'alpha': [0.1, 0.3, 1, 3, 6, 8, 10]
                    }, verbose=1)

gs_lr.fit(X_Train, y_train)
lasso = gs_lr.best_estimator_
print(f"Explained Variance Score: {explained_variance_score(y_pred=lasso.predict(X_Train), y_true=y_train)}")
print(f"RMSE: {np.sqrt(metrics.mean_squared_error(y_pred=lasso.predict(X_Train), y_true=y_train))}")
print(f"Explained Variance Score: {explained_variance_score(y_pred=lasso.predict(X_CV), y_true=y_cv)}")
print(f"RMSE: {np.sqrt(metrics.mean_squared_error(y_pred=lasso.predict(X_CV), y_true=y_cv))}")
ann = Sequential()
ann.add(Dense(15, activation='relu'))
ann.add(Dense(10, activation='relu'))
ann.add(Dense(10, activation='relu'))
ann.add(Dense(5, activation='relu'))
ann.add(Dense(1))
ann.compile(optimizer='Adam', loss='mse')
early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=10)
ann.fit(x=np.array(X_Train),
       y=np.array(y_train),
       epochs=500,
       verbose=1,
       validation_data=(np.array(X_CV), np.array(y_cv)),
       callbacks=[early_stop])
error_ann = pd.DataFrame(ann.history.history)
error_ann.plot()
print(f"Explained Variance Score: {explained_variance_score(y_pred=ann.predict(X_Train), y_true=y_train)}")
print(f"RMSE: {np.sqrt(metrics.mean_squared_error(y_pred=ann.predict(X_Train), y_true=y_train))}")
print(f"Explained Variance Score: {explained_variance_score(y_pred=ann.predict(X_CV), y_true=y_cv)}")
print(f"RMSE: {np.sqrt(metrics.mean_squared_error(y_pred=ann.predict(X_CV), y_true=y_cv))}")
gs_xgb = GridSearchCV(XGBRegressor(booster='gbtree', subsample=0.75),
                     param_grid={
                         'min_child_weight': [4, 6, 8],
                         'max_depth': [8, 10, 12],
                         'eta': [0.3, 0.03], 
                         'learning_rate': [0.01, 0.1],
                         'reg_alpha': [0.1, 1, 3],
                         'reg_lambda': [0.1, 1, 2, 3]
                     }, 
                     verbose=3, 
                     cv=3)
gs_xgb.fit(X_Train, y_train)
xgb = gs_xgb.best_estimator_
print(f"Explained Variance Score: {explained_variance_score(y_pred=xgb.predict(X_Train), y_true=y_train)}")
print(f"RMSE: {np.sqrt(metrics.mean_squared_error(y_pred=xgb.predict(X_Train), y_true=y_train))}")
print(f"Explained Variance Score: {explained_variance_score(y_pred=xgb.predict(X_CV), y_true=y_cv)}")
print(f"RMSE: {np.sqrt(metrics.mean_squared_error(y_pred=xgb.predict(X_CV), y_true=y_cv))}")
print(f"Explained Variance Score: {explained_variance_score(y_pred=xgb.predict(X_Test), y_true=y_test)}")
print(f"RMSE: {np.sqrt(metrics.mean_squared_error(y_pred=xgb.predict(X_Test), y_true=y_test))}")