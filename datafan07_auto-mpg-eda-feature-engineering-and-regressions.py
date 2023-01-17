import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



#



from sklearn.preprocessing import LabelEncoder, StandardScaler
train_df = pd.read_csv('/kaggle/input/autompg-dataset/auto-mpg.csv')
print(f'Training Shape: {train_df.shape}')
print(train_df.head())

print(train_df.sample(5))
display(train_df.info())
train_df['horsepower'] = train_df['horsepower'].replace('?', np.NaN).astype('float64')
train_df_corr = train_df.corr().abs().unstack().sort_values(kind='quicksort', ascending=False).reset_index()

train_df_corr.rename(columns={"level_0": "Feature A", 

                             "level_1": "Feature B", 0: 'Correlation Coefficient'}, inplace=True)

train_df_corr[train_df_corr['Feature A'] == 'horsepower'].style.background_gradient(cmap='summer_r')
train_df['horsepower'] = train_df.groupby(['displacement'], sort=False)['horsepower'].apply(lambda x: x.fillna(x.mean()))

train_df['horsepower'] = train_df.groupby(['cylinders'], sort=False)['horsepower'].apply(lambda x: x.fillna(x.mean()))
sns.set()

plt.subplots(figsize=(10, 6))

sns.distplot(train_df['horsepower'],bins=25)

plt.show()
print(f'There are some missing values: {train_df.isna().any().any()}')
g = sns.PairGrid(train_df.drop('car name',axis=1), hue='origin')

g = g.map_diag(plt.hist, alpha=0.4)

g = g.map_upper(sns.scatterplot)

g = g.map_lower(sns.regplot)
org=train_df.copy()

org['origin']=train_df.origin.map({1: 'US', 2: 'Asian',3:'European'})

org['origin'].value_counts(normalize=True)
sns.set()

fig, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x='origin', y="mpg", data=org)

plt.axhline(org.mpg.mean(),color='r',linestyle='dashed',linewidth=2)

plt.show()
pd.crosstab(train_df['car name'],train_df['origin'])
train_df['Brand'] = train_df['car name'].str.extract('([A-Za-z]+)\s', expand=False)
train_df['Brand']= train_df['Brand'].replace(np.NaN, 'subaru')

train_df['Brand']= train_df['Brand'].replace('chevroelt', 'chevrolet')

train_df['Brand']= train_df['Brand'].replace('vw', 'volkswagen')

train_df['Brand']= train_df['Brand'].replace('toyouta', 'toyota')

train_df['Brand']= train_df['Brand'].replace('vokswagen', 'volkswagen')

train_df['Brand']= train_df['Brand'].replace('maxda', 'mazda')

train_df['Brand']= train_df['Brand'].replace('mazada', 'mazda')

train_df['Brand']= train_df['Brand'].replace('chevy', 'chevrolet')
train_df['Brand'].value_counts(normalize=True)
fig, ax = plt.subplots(figsize=(16, 8))

sns.countplot(train_df['Brand'])

plt.xticks(rotation=60)

plt.show()
le = LabelEncoder()

train_df['Brand'] = le.fit_transform(train_df['Brand'])

train_df.drop('car name', axis=1, inplace=True)

train_df.sample(5)
features=train_df.columns.tolist()

for feature in features:

    print(f'{feature} Skewness: {train_df[feature].skew():.2f}, Kurtosis: {train_df[feature].kurtosis():.2f}')
skew_cols=['cylinders','displacement','horsepower','weight']

train_df[skew_cols]=np.log1p(train_df[skew_cols])

for feature in features:

    print(f'{feature} skewness: {train_df[feature].skew():.2f}, Kurtosis: {train_df[feature].kurtosis():.2f}')
from sklearn.model_selection import train_test_split, KFold, cross_val_score

from sklearn.linear_model import LinearRegression, Ridge

from sklearn.metrics import r2_score, mean_squared_error

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, BaggingRegressor

import xgboost as xgb

import lightgbm as lgb
n_folds = 5



def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train_df.drop('mpg', axis=1))

    rmse= np.sqrt(np.abs(cross_val_score(model, train_df.drop('mpg', axis=1).values, train_df['mpg'], scoring="neg_mean_squared_error", cv = kf, n_jobs=-1)))

    return(rmse)



def rtw_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train_df.drop('mpg', axis=1))

    rtw= cross_val_score(model, train_df.drop('mpg', axis=1).values, train_df['mpg'], scoring="r2", cv = kf, n_jobs=-1)

    return(rtw)
mods = [LinearRegression(),Ridge(),GradientBoostingRegressor(),

      RandomForestRegressor(),BaggingRegressor(),

      xgb.XGBRegressor(), lgb.LGBMRegressor()]



model_df = pd.DataFrame({

    'Model': [type(i).__name__ for i in mods],

    'RMSE': [np.mean(rmsle_cv(i)) for i in mods],    

    'Rmse Std': [np.std(rmsle_cv(i)) for i in mods],

    'R2': [np.mean(rtw_cv(i)) for i in mods],

    'R2 Std': [np.std(rmsle_cv(i)) for i in mods]})

display(model_df.sort_values(by='RMSE', ascending=True).reset_index(drop=True).style.background_gradient(cmap='summer_r'))
X_train, X_test, y_train, y_test = train_test_split(train_df.drop('mpg', axis=1),train_df['mpg'], test_size= 0.33, random_state=42)
linreg=LinearRegression()

linreg.fit(X_train,y_train)

y_pred=linreg.predict(X_test)



rmse=np.sqrt(mean_squared_error(y_test,y_pred))

print(rmse)

print(r2_score(y_test, y_pred))
ridge = Ridge()

ridge.fit(X_train,y_train)

y_pred=ridge.predict(X_test)



rmse=np.sqrt(mean_squared_error(y_test,y_pred))

print(rmse)

print(r2_score(y_test, y_pred))
bag_regressor = BaggingRegressor()

bag_regressor.fit(X_train,y_train)

y_predict = bag_regressor.predict(X_test)

rmse_bgr = np.sqrt(mean_squared_error(y_test,y_predict))



rmse=np.sqrt(mean_squared_error(y_test,y_predict))

print(rmse)

print(r2_score(y_test, y_predict))
gb_regressor = GradientBoostingRegressor()

gb_regressor.fit(X_train,y_train)

y_predict = gb_regressor.predict(X_test)

rmse_bgr = np.sqrt(mean_squared_error(y_test,y_predict))



rmse=np.sqrt(mean_squared_error(y_test,y_predict))

print(rmse)

print(r2_score(y_test, y_predict))





feature_imp = pd.DataFrame(sorted(zip(gb_regressor.feature_importances_,train_df.drop('mpg', axis=1).columns)), columns=['Value','Feature'])



plt.figure(figsize=(12, 6))

sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))

plt.title('Features')

plt.tight_layout()

plt.show()
rf_regressor = RandomForestRegressor()

rf_regressor.fit(X_train,y_train)

y_predict = rf_regressor.predict(X_test)

rmse_bgr = np.sqrt(mean_squared_error(y_test,y_predict))



rmse=np.sqrt(mean_squared_error(y_test,y_predict))

print(rmse)

print(r2_score(y_test, y_predict))





feature_imp = pd.DataFrame(sorted(zip(rf_regressor.feature_importances_,train_df.drop('mpg', axis=1).columns)), columns=['Value','Feature'])



plt.figure(figsize=(12, 6))

sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))

plt.title('Features')

plt.tight_layout()

plt.show()
xg_reg=xgb.XGBRegressor(booster='gbtree', objective='reg:squarederror')



xg_reg.fit(X_train,y_train)

xg_y_pred=xg_reg.predict(X_test)

xg_rmse=np.sqrt(mean_squared_error(y_test,xg_y_pred))

print(xg_rmse)

print(r2_score(y_test,xg_y_pred))



feature_imp = pd.DataFrame(sorted(zip(xg_reg.feature_importances_,train_df.drop('mpg', axis=1).columns)), columns=['Value','Feature'])



plt.figure(figsize=(12, 6))

sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))

plt.title('Features')

plt.tight_layout()

plt.show()
lgb_reg = lgb.LGBMRegressor()



lgb_reg.fit(X_train,y_train)

y_predict=lgb_reg.predict(X_test)

rmse=np.sqrt(mean_squared_error(y_test,y_predict))

print(rmse)

print(r2_score(y_test,y_predict))



feature_imp = pd.DataFrame(sorted(zip(lgb_reg.feature_importances_,train_df.drop('mpg', axis=1).columns)), columns=['Value','Feature'])



plt.figure(figsize=(12, 6))

sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))

plt.title('Features')

plt.tight_layout()

plt.show()