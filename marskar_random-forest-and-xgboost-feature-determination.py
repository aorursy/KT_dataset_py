import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # data visualization

import seaborn as sns

import xgboost as xgb

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import ExtraTreesRegressor;
# Set plot parameters

from IPython.display import set_matplotlib_formats



plt.rcParams['savefig.dpi'] = 300



%matplotlib inline

### Seaborn style

sns.set_style("whitegrid")
# Input data files are available in the "../input/" directory, on Kaggle and the GitHub repo for this project.

prop = pd.read_csv("../input/properties_2016.csv", low_memory=False)

prop.shape;
# Now I will calculate the percent missing values(NaN)

nan = prop.isnull().sum()/len(prop)*100
### Plotting NaN counts

nan_sorted = nan.sort_values(ascending=False).to_frame().reset_index()

nan_sorted.columns = ['Column', 'percentNaN']

nan_sorted.head();
train = pd.read_csv("../input/train_2016_v2.csv", parse_dates=["transactiondate"])
train['transaction_month'] = pd.DatetimeIndex(train['transactiondate']).month

train.sort_values('transaction_month', axis=0, ascending=True, inplace=True)
# Here I will merge the train and properties datasets

train = pd.merge(train, prop, on='parcelid', how='left')
# Now I will impute the missing values with median values to compute the importance scores

median_values_train = train.median(axis=0)

train = train.fillna(median_values_train, inplace=True)
for c in train[['transactiondate', 'hashottuborspa', 'propertycountylandusecode', 'propertyzoningdesc', 'fireplaceflag', 'taxdelinquencyflag']]:

    label = LabelEncoder()

    label.fit(list(train[c].values))

    train[c] = label.transform(list(train[c].values))



x_train = train.drop(['parcelid', 'logerror', 'transactiondate'], axis=1)

y_train = train['logerror']
rf = RandomForestRegressor(n_estimators=30, max_features=None)

rf.fit(x_train, y_train);
rf_importance = rf.feature_importances_

rf_importance_df = pd.DataFrame()

rf_importance_df['features'] = x_train.columns

rf_importance_df['importance'] = rf_importance

rf_importance_df.head();
rf_importance_df.sort_values('importance', axis=0, inplace=True, ascending=False)



rf_importance_df_trim = rf_importance_df[rf_importance_df.importance>0.001]



rf_importance_df_trim.tail()



rf_feature_list = rf_importance_df_trim.features
xgb_params = {

    'eta': 0.05,

    'max_depth': 8,

    'subsample': 0.7,

    'colsample_bytree': 0.7,

    'objective': 'reg:linear',

    'silent': 1,

    'seed' : 0

}

dtrain = xgb.DMatrix(x_train, y_train, feature_names=x_train.columns.values)

model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=50)
xgbdict = model.get_score()



xgb_importance_df = pd.DataFrame()

xgb_importance_df['features'] = xgbdict.keys()

xgb_importance_df['importance'] = xgbdict.values()



xgb_importance_df.sort_values('importance', axis=0, inplace=True, ascending=False)



xgb_importance_df_trim = xgb_importance_df[xgb_importance_df.importance>=10]



xgb_feature_list = xgb_importance_df_trim.features



feature_list = xgb_feature_list.append(rf_feature_list)



feature_list = feature_list.unique()
feature_list = list(feature_list)
fig, ax = plt.subplots(figsize=(48, 100), dpi=300)

sns.set_context("poster", font_scale=6)

# [1]

sns.barplot(x="importance", y="features", data=rf_importance_df, color='Green', ax=ax)

ax.set(xlabel="Importance (Variance explained)", ylabel="")

ax.set_title('Random Forest Importance', fontsize= 96)

plt.show()
fig, ax = plt.subplots(figsize=(48, 100), dpi=300)

sns.set_context("poster", font_scale=6)

# [2]

xgb.plot_importance(model, height=0.85, grid = False, color="blue", ax=ax)

ax.xaxis.grid()

ax.set_title('XGBoost Importance', fontsize= 96)

ax.set(xlabel="Importance (F score)", ylabel="")

plt.show()
# Bibliography is added in post-processing