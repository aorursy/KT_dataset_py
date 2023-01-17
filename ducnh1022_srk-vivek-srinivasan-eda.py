# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from scipy import stats

import missingno as msno

from datetime import datetime

import matplotlib

import matplotlib.pyplot as plt

from scipy.stats import kendalltau

import warnings



color = sns.color_palette()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("../input/train_2016.csv", parse_dates=["transactiondate"])

prop_df = pd.read_csv("../input/properties_2016.csv")

train_df = pd.merge(train_df, prop_df, on='parcelid', how='left')
# initial

col_tar = "logerror"

col_id = "parcelid"

print("SHAPE: " + str(train_df.shape))

print("\n" + "="*90)

print("\nSAMPLE DATA:" )

with pd.option_context('display.max_rows', None, 'display.max_columns', None):

    print(train_df.head(2).transpose())

    

    

# target distribution



ulimit = np.percentile(train_df[col_tar].values, 99)

llimit = np.percentile(train_df[col_tar].values, 1)

train_df[col_tar].ix[train_df[col_tar]>ulimit] = ulimit

train_df[col_tar].ix[train_df[col_tar]<llimit] = llimit



fig,ax = plt.subplots()

fig.set_size_inches(20,5)

sns.distplot(train_df[col_tar].values, bins=50,kde=False,color="#34495e",ax=ax)

ax.set(xlabel=col_tar, ylabel='VIF Score',title="Distribution Of Dependent Variable")

plt.show()



# dtype print, you should encode this   



dtype_df = train_df.dtypes.reset_index()

dtype_df.columns = ["count", "column_type"]

print("\n" + "="*90)

print("\n DTYPE_DF: " )

with pd.option_context('display.max_rows', None, 'display.max_columns', None):

    print(dtype_df)



dtype_count_df = dtype_df.groupby("column_type").aggregate('count').reset_index()

print("\n" + "="*90)

print("\n DTYPE_COUNT_DF: " )

with pd.option_context('display.max_rows', None, 'display.max_columns', None):

    print(dtype_count_df)

    

fig,ax = plt.subplots()

fig.set_size_inches(20,len(dtype_count_df))

plt.ylabel('column_type', fontsize=20)

plt.xlabel('count', fontsize=20)

plt.title("Dtype Count", fontsize=30)

sns.barplot(data=dtype_count_df,y="column_type",x="count",ax=ax,palette="Blues_d",orient="h")

plt.show()



# missing values



missing_df = train_df.isnull().sum(axis=0).reset_index()

missing_df.columns = ['column_name', 'missing_count']

missing_df['missing_ratio'] = missing_df['missing_count'] / train_df.shape[0]

missing_df_top = missing_df.ix[missing_df['missing_ratio']>0.8].sort_values(by="missing_ratio", ascending=False)

print("\n" + "="*90)

print("\n MISSING_DF: " )

with pd.option_context('display.max_rows', None, 'display.max_columns', None):

    print(missing_df_top)



fig,ax = plt.subplots()

fig.set_size_inches(20,len(missing_df_top))

sns.barplot(data=missing_df_top,y="column_name",x="missing_count",ax=ax,color="#34495e",orient="h")

plt.ylabel('column_type', fontsize=20)

plt.xlabel('count', fontsize=20)

plt.title("Missing Value Count", fontsize=30)

plt.show()







# dtype print, you should encode this 

# missing values

# drop col Object

# detect col Date, generate year, month, day_month, day_week, plot with target value and # of occurences

# detect long, lat and plot

# correlation, plot, imp1

# make predict with xgboost and random_forest, plot importance, imp2, imp3

# imp1, imp2, imp3 => 10 most important feature

#
# dtype print, you should encode this 

# missing values

# drop col Object

dropping = dtype_df[dtype_df['column_type'] == 'object']['count'].values.tolist() + missing_df_top[missing_df_top['missing_ratio'] > 0.9]['column_name'].values.tolist()

for drop in dropping:

    print("Drop: "+ str(drop))



train_df_0 = train_df.drop(labels=dropping,axis=1)

train_df_0.head(2)

# detect col Date, generate year, month, day_month, day_week, plot with target value and # of occurences

# detect long, lat and plot

# correlation, plot, imp1

# make predict with xgboost and random_forest, plot importance, imp2, imp3

# imp1, imp2, imp3 => 10 most important feature
col_date = dtype_df[dtype_df['column_type'] == 'datetime64[ns]']['count'].values.tolist()[0]



train_date_df = train_df.copy()



train_date_df[col_date+'_year'] = train_date_df[col_date].dt.year

train_date_df[col_date+'_month'] = train_date_df[col_date].dt.month

train_date_df[col_date+'_day'] = train_date_df[col_date].dt.day

train_date_df[col_date+'_dayofyear'] = train_date_df[col_date].dt.dayofyear

train_date_df[col_date+'_weekday'] = train_date_df[col_date].dt.weekday



for col in [col_date+'_year',col_date+'_month',col_date+'_day',col_date+'_dayofyear',col_date+'_weekday']:

    print(col)

    cnt_srs = train_date_df[col].value_counts()

    plt.figure(figsize=(12,6))

    sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color="#34495e")

    plt.xticks(rotation='vertical')

    plt.xlabel(col + ' of transaction', fontsize=12)

    plt.ylabel('Number of Occurrences', fontsize=12)

    plt.show()



    train_group_temp = train_date_df.groupby(col)['logerror'].mean().to_frame().reset_index()

    plt.figure(figsize=(12,6))

    sns.pointplot(x=train_group_temp[col], y=train_group_temp["logerror"], data=train_group_temp, join=True,color="#34495e")

    plt.xlabel(col + ' of transaction', fontsize=12)

    plt.ylabel('Mean target', fontsize=12)

    plt.show()
lng = False

lat = False

for col in train_df.columns:

    if "longitude" in col:

        lng = True

    if "latitude" in col:

        lat = True

if(lng and lat):

    plt.figure(figsize=(12,12))

    sns.jointplot(x=prop_df.latitude.values, y=prop_df.longitude.values, size=10)

    plt.ylabel('Longitude', fontsize=12)

    plt.xlabel('Latitude', fontsize=12)

    plt.show()

# detect long, lat and plot

# correlation, plot, imp1

# make predict with xgboost and random_forest, plot importance, imp2, imp3

# imp1, imp2, imp3 => 10 most important feature
# Let us just impute the missing values with mean values to compute correlation coefficients #

mean_values = train_df.mean(axis=0)

train_df_new = train_df.fillna(mean_values, inplace=True)



# Now let us look at the correlation coefficient of each of these variables #

x_cols = [col for col in train_df_new.columns if col not in [col_tar] if train_df_new[col].dtype=='float64']



labels = []

values = []

for col in x_cols:

    labels.append(col)

    values.append(np.corrcoef(train_df_new[col].values, train_df_new[col_tar].values)[0,1])

corr_df = pd.DataFrame({'col_labels':labels, 'corr_values':values})

corr_df = corr_df.sort_values(by='corr_values')



ind = np.arange(len(labels))

width = 0.9

fig, ax = plt.subplots(figsize=(12,40))

rects = ax.barh(ind, np.array(corr_df.corr_values.values), color='#34495e')

ax.set_yticks(ind)

ax.set_yticklabels(corr_df.col_labels.values, rotation='horizontal')

ax.set_xlabel("Correlation coefficient")

ax.set_title("Correlation coefficient of the variables")

#autolabel(rects)

plt.show()



corr_df["abs"] = np.abs(corr_df["corr_values"])

col_use_0 = corr_df.sort_values(by="abs",ascending=False)[:10]["col_labels"].tolist()



temp_df = train_df[col_use_0 + [col_tar]]

corrmat = temp_df.corr(method='spearman')

f, ax = plt.subplots(figsize=(8, 8))



# Draw the heatmap using seaborn

sns.heatmap(corrmat, vmax=1., square=True)

plt.title("Important variables correlation map", fontsize=15)

plt.show()





# correlation, plot, imp1

# make predict with xgboost and random_forest, plot importance, imp2, imp3

# imp1, imp2, imp3 => 10 most important feature
from sklearn import model_selection, preprocessing

import xgboost as xgb



train_df_new = train_df_new.fillna(-999)

for f in train_df_new.columns:

    if train_df_new[f].dtype=='object':

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(train_df_new[f].values)) 

        train_df_new[f] = lbl.transform(list(train_df_new[f].values))

        

train_y = train_df_new[col_tar].values

train_X = train_df_new.drop([col_tar,col_id,col_date], axis=1)



xgb_params = {

    'eta': 0.05,

    'max_depth': 8,

    'subsample': 0.7,

    'colsample_bytree': 0.7,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'silent': 1

}

dtrain = xgb.DMatrix(train_X, train_y, feature_names=train_X.columns.values)

model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=100)
featureImportance = model.get_fscore()

features = pd.DataFrame()

features['features'] = featureImportance.keys()

features['importance'] = featureImportance.values()

features.sort_values(by=['importance'],ascending=False,inplace=True)

fig,ax= plt.subplots()

fig.set_size_inches(20,10)

plt.xticks(rotation=90)

sns.barplot(data=features.head(15),x="importance",y="features",ax=ax,orient="h",color="#34495e")

col_use_1 = features[:10]["features"].tolist()
from sklearn import ensemble

model0 = ensemble.ExtraTreesRegressor(n_estimators=25, max_depth=30, max_features=0.3, n_jobs=-1, random_state=0)

model0.fit(train_X, train_y)



importances = model0.feature_importances_

std = np.std([tree.feature_importances_ for tree in model0.estimators_], axis=0)

indices = np.argsort(importances)[::-1][:20]

feat_names = train_X.columns

plt.figure(figsize=(12,12))

plt.title("Feature importances")

plt.bar(range(len(indices)), importances[indices], color="#34495e", yerr=std[indices], align="center")

plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical')

plt.xlim([-1, len(indices)])

plt.show()

col_use_2 = feat_names[indices][:10].tolist()
cols = {}

for col in (col_use_0 + col_use_1 + col_use_2):

    if col not in cols.keys():

        cols[col] = 1

    else:

        cols[col] = cols[col] + 1



col_df = pd.DataFrame(cols,index=[0]).transpose()

col_df.columns = ["count"]

col_df.sort_values(by="count",inplace=True,ascending=False)

col_list = col_df.index[:10].tolist()

col_df
col_list
for col in col_list:

    uniq = len(train_df[col].unique())

    print(uniq)