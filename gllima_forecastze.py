import pandas as pd

import numpy as np

from pandas_profiling import ProfileReport

from matplotlib import pyplot as plt

from matplotlib.gridspec import GridSpec

import seaborn as sns

import warnings

warnings.filterwarnings("ignore") # ignoring annoying warnings



from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error

from tpot import TPOTRegressor

import xgboost as xgb



enable_pandas_profilling = False

enable_tpot = False
df_features = pd.read_csv('../input/walmart-recruiting-store-sales-forecasting/features.csv.zip')

df_train = pd.read_csv('../input/walmart-recruiting-store-sales-forecasting/train.csv.zip')

df_stores = pd.read_csv('../input/walmart-recruiting-store-sales-forecasting/stores.csv')

df_test = pd.read_csv('../input/walmart-recruiting-store-sales-forecasting/test.csv.zip')

df_sample_submission = pd.read_csv('../input/walmart-recruiting-store-sales-forecasting/sampleSubmission.csv.zip')
df_feat_stores = df_features.merge(df_stores, how='inner', on='Store')
pd.DataFrame({'Type_Feat_Store': df_feat_stores.dtypes,'Type_Train': df_train.dtypes, 'Type_Test': df_test.dtypes})
df_feat_stores.Date = pd.to_datetime(df_feat_stores.Date)

df_train.Date = pd.to_datetime(df_train.Date)

df_test.Date = pd.to_datetime(df_test.Date)
len(df_feat_stores.Date.unique())/52
df_feat_stores['Week'] = df_feat_stores.Date.dt.week 

df_feat_stores['Year'] = df_feat_stores.Date.dt.year
df_train_feats = df_train.merge(df_feat_stores, how='inner', on=['Store','Date','IsHoliday']).sort_values(by=['Store','Dept','Date']).reset_index(drop=True)

df_test_feats = df_test.merge(df_feat_stores, how='inner', on=['Store','Date','IsHoliday']).sort_values(by=['Store','Dept','Date']).reset_index(drop=True)
del df_feat_stores, df_train, df_test
#Generate the report. We would use the mpg dataset as sample, title parameter for naming our report, and explorative parameter set to True for Deeper exploration.

if enable_pandas_profilling:

    profile_train = ProfileReport(df_train_feats, title='Training Dataset Report', explorative = True)

    profile_train.to_file(output_file='profile_train.html')

    profile_train
df_train_feats = df_train_feats.drop(columns=['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5'])

df_test_feats = df_test_feats.drop(columns=['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5'])
sns.set(style="white")

corr = df_train_feats.corr()

mask = np.triu(np.ones_like(corr, dtype=np.bool))

f, ax = plt.subplots(figsize=(20, 15))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

plt.title('Correlation Matrix', fontsize=18)

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)



plt.show()
corr['Weekly_Sales'].sort_values()
df_train_feats = df_train_feats.drop(columns=['Fuel_Price', 'Temperature','Date', 'CPI', 'Unemployment'])

df_test_feats = df_test_feats.drop(columns=['Fuel_Price', 'Temperature','Date', 'CPI', 'Unemployment'])
sns.set(style="white")

corr = df_train_feats.corr()

mask = np.triu(np.ones_like(corr, dtype=np.bool))

f, ax = plt.subplots(figsize=(20, 15))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

plt.title('Correlation Matrix', fontsize=18)

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)



plt.show()
corr['Weekly_Sales'].sort_values()
df_sales_2010_weekly = df_train_feats[df_train_feats.Year==2010]['Weekly_Sales'].groupby(df_train_feats['Week']).sum()

df_sales_2011_weekly = df_train_feats[df_train_feats.Year==2011]['Weekly_Sales'].groupby(df_train_feats['Week']).sum()

df_sales_2012_weekly = df_train_feats[df_train_feats.Year==2012]['Weekly_Sales'].groupby(df_train_feats['Week']).sum()
plt.figure(figsize=(20,8))

sns.lineplot(df_sales_2010_weekly.index, df_sales_2010_weekly.values)

sns.lineplot(df_sales_2011_weekly.index, df_sales_2011_weekly.values)

sns.lineplot(df_sales_2012_weekly.index, df_sales_2012_weekly.values)

plt.grid()

plt.xticks(np.arange(1, 53, step=1))

plt.legend(['2010', '2011', '2012'], loc='best', fontsize=16)

plt.title('Overall Weekly Sales - Per Year', fontsize=18)

plt.ylabel('Sales', fontsize=16)

plt.xlabel('Week', fontsize=16)

plt.show()
df_train_feats[df_train_feats.IsHoliday].Week.unique()
df_train_feats.loc[(df_train_feats.Year==2010) & (df_train_feats.Week==13), 'IsHoliday'] = True

df_train_feats.loc[(df_train_feats.Year==2011) & (df_train_feats.Week==16), 'IsHoliday'] = True

df_train_feats.loc[(df_train_feats.Year==2012) & (df_train_feats.Week==14), 'IsHoliday'] = True

df_test_feats.loc[(df_test_feats.Year==2013) & (df_test_feats.Week==13), 'IsHoliday'] = True
weekly_sales_mean = df_train_feats['Weekly_Sales'].groupby(df_train_feats['Week']).mean()

plt.figure(figsize=(20,8))

sns.lineplot(weekly_sales_mean.index, weekly_sales_mean.values)

plt.grid()

plt.xticks(np.arange(1, 53, step=1))

plt.legend(['Mean'], loc='best', fontsize=16)

plt.title('Mean Weekly Sales', fontsize=18)

plt.ylabel('Sales', fontsize=16)

plt.xlabel('Week', fontsize=16)

plt.axvline(6, color='red')

plt.axvline(13, color='red')

plt.axvline(14, color='red')

plt.axvline(36, color='red')

plt.axvline(47, color='red')

plt.axvline(52, color='red')

plt.show()
df_train_feats.Type.unique()
fig = plt.figure(figsize=(20,8))

gs = GridSpec(1,2)

sns.boxplot(y=df_train_feats.Weekly_Sales, x=df_train_feats.Type, ax=fig.add_subplot(gs[0,0]))

plt.ylabel('Sales', fontsize=16)

plt.xlabel('Type', fontsize=16)

sns.stripplot(y=df_train_feats.Weekly_Sales, x=df_train_feats.Type, ax=fig.add_subplot(gs[0,1]))

plt.ylabel('Sales', fontsize=16)

plt.xlabel('Type', fontsize=16)

fig.show()
def nominal_to_ordinal(x):

    if x == 'A':

        return 3

    elif x == 'B':

        return 2

    else:

        return 1

    

df_train_feats.Type = df_train_feats.Type.apply(nominal_to_ordinal)

df_test_feats.Type = df_test_feats.Type.apply(nominal_to_ordinal)
df_train_feats.Type.unique()
df_test_feats.Type.unique()
fig = plt.figure(figsize=(20,8))

gs = GridSpec(1,2)

sns.boxplot(y=df_train_feats.Weekly_Sales, x=df_train_feats.Type, ax=fig.add_subplot(gs[0,0]))

plt.ylabel('Sales', fontsize=16)

plt.xlabel('TypeOrd', fontsize=16)

sns.stripplot(y=df_train_feats.Weekly_Sales, x=df_train_feats.Type, ax=fig.add_subplot(gs[0,1]))

plt.ylabel('Sales', fontsize=16)

plt.xlabel('TypeOrd', fontsize=16)

fig.show()
def bool_to_num(x):

    if x:

        return 1

    else:

        return 0

    

df_train_feats.IsHoliday = df_train_feats.IsHoliday.apply(bool_to_num)

df_test_feats.IsHoliday = df_test_feats.IsHoliday.apply(bool_to_num)
df_train_feats.head()
df_test_feats.head()
df_train_feats.IsHoliday.unique()
X_train = df_train_feats.drop('Weekly_Sales', axis= 1).values

y_train = df_train_feats.Weekly_Sales.values
len(X_train), len(y_train) 
# TPOT takes 10hours to get a good parameters estimation so it is set False here.

if enable_tpot:

    X_tpot_train, X_tpot_test, y_tpot_train, y_tpot_test = train_test_split(X_train,  y_train, train_size=0.75, test_size=0.25, random_state=42)

    tpot = TPOTRegressor(generations=5, population_size=50, verbosity=2, random_state=42)

    tpot.fit(X_tpot_train, y_tpot_train)

    print(tpot.score(X_tpot_test, y_tpot_test))

    tpot.export('tpot_pipeline.py')
model = RandomForestRegressor(n_estimators=100, max_depth=27, max_features=6, min_samples_split=8, min_samples_leaf=1)

model.fit(X_train, y_train)

preds = model.predict(X_train)

rmse_all = np.sqrt(mean_squared_error(y_train, preds))

print("RMSE: %f" % (rmse_all))

del model
model = RandomForestRegressor(n_estimators=50, max_depth=27, max_features=6, min_samples_split=8, min_samples_leaf=1)

model.fit(X_train, y_train)

preds = model.predict(X_train)

rmse_all = np.sqrt(mean_squared_error(y_train, preds))

print("RMSE: %f" % (rmse_all))

del model
model = RandomForestRegressor(n_estimators=50, max_depth=27, max_features=6, min_samples_split=4, min_samples_leaf=1)

model.fit(X_train, y_train)

preds = model.predict(X_train)

rmse_all = np.sqrt(mean_squared_error(y_train, preds))

print("RMSE: %f" % (rmse_all))

del model
model = RandomForestRegressor(n_estimators=50, max_depth=27, max_features=6, min_samples_split=3, min_samples_leaf=1)

model.fit(X_train, y_train)

preds = model.predict(X_train)

rmse_all = np.sqrt(mean_squared_error(y_train, preds))

print("RMSE: %f" % (rmse_all))

del model
model = RandomForestRegressor(n_estimators=50, max_depth=27, max_features=6, min_samples_split=2, min_samples_leaf=1)

model.fit(X_train, y_train)

preds = model.predict(X_train)

rmse_all = np.sqrt(mean_squared_error(y_train, preds))

print("RMSE: %f" % (rmse_all))
len(df_test_feats), len(df_sample_submission)
df_test_feats.info()
predicted = model.predict(df_test_feats)
df_sample_submission['Weekly_Sales'] = predicted

df_sample_submission.to_csv('submission.csv',index=False)