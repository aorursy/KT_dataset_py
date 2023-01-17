# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

#from sklearn.cluster import DBSCAN, KMeans

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV

import xgboost as xgb

import lightgbm as lgb



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

sns.set_style('darkgrid')

import os



print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")
train_df.head(10)
train_df.dtypes
train_df['date'] = pd.to_datetime(train_df.date)

train_df.head()
train_df["Year"] = train_df.date.dt.year

train_df["Month"] = train_df.date.dt.month

train_df["day"] = train_df.date.dt.day

train_df["dayOfweek"] = train_df.date.dt.dayofweek
train_df.head(10)
train_df = train_df.drop('date', axis = 1)

# 시계열데이터가 아니므로 date의 정보를 연,월,일 별로 나눈후 없애줘도 된다고 생각함
test_df.drop(["date"], axis=1, inplace =True)
plt.figure(figsize = (12,8))

sns.distplot(train_df.price , kde = True, bins=60, color = "magenta") #예측변수 분포 확인 
plt.figure(figsize = (12,8))

sns.distplot(np.log(train_df.price), kde = True, bins=60, color = "green") #log변환후 분포 확인
plt.figure(figsize = (12,8))

sns.distplot(train_df.sqft_living, kde = True, bins = 120 , color = 'blue')
fig, axes = plt.subplots(nrows=2)

fig.set_size_inches(12,8)

sns.scatterplot(x=train_df.sqft_living, y= train_df.bedrooms, ax = axes[0]) 

sns.scatterplot(x=train_df.sqft_living, y= train_df.bathrooms, ax = axes[1])  
train_df.bathrooms.unique()
plt.figure(figsize=(12,8))

sns.scatterplot(x=train_df.sqft_living, y=np.log(train_df.price))
fig, axes = plt.subplots(nrows=2, ncols = 2)

fig.set_size_inches(16,8)



sns.scatterplot(x=train_df.sqft_living, y=np.log(train_df.price), ax = axes[0][0], color = 'red', label = "sqft_living")

sns.scatterplot(x=train_df.sqft_above, y=np.log(train_df.price), ax = axes[0][1], color = 'red', label = "sqft_above")

sns.scatterplot(x= train_df[train_df.sqft_basement == 0].sqft_living, y= np.log(train_df.price), ax=axes[1][0], label = "no_basement")

sns.scatterplot(x= train_df[train_df.sqft_basement != 0].sqft_living, y= np.log(train_df.price), ax=axes[1][1], label = "has_basement")
fig, axes = plt.subplots(ncols =2)

fig.set_size_inches(18,6)



sns.scatterplot(x=train_df.yr_built, y=np.log(train_df.price), ax = axes[0])

sns.scatterplot(x=train_df[train_df.yr_renovated != 0].yr_renovated, y=np.log(train_df[train_df.yr_renovated != 0].price), ax = axes[1])
train_df["yr_recent_built"]=np.where(train_df.yr_built >= train_df.yr_renovated, train_df.yr_built, train_df.yr_renovated)

plt.figure(figsize = (10,5))

sns.scatterplot(x= train_df.yr_recent_built, y= np.log(train_df.price))
test_df["yr_recent_built"]=np.where(test_df.yr_built >= test_df.yr_renovated, test_df.yr_built, test_df.yr_renovated)
fig, axes = plt.subplots(nrows=2)

fig.set_size_inches(16,10)

sns.boxplot(x=train_df.Month , y= np.log(train_df.price), ax = axes[0])

sns.boxplot(x=train_df.Year, y=np.log(train_df.price), ax = axes[1])
train_df.drop(["Year","Month", "day", "dayOfweek"], axis=1, inplace = True)
plt.figure(figsize = (18,8))

cnt_plot = sns.countplot(train_df.grade)

ncount = len(train_df)

for p in cnt_plot.patches:

    x = p.get_bbox().get_points()[:,0]

    y = p.get_bbox().get_points()[1,1]

    cnt_plot.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y))
logprice_by_grade = np.log(train_df[["grade","price"]].groupby("grade").mean()).reset_index()

plt.figure(figsize=(16,10))

sns.barplot(x="grade" , y="price", data = logprice_by_grade)
price_by_grade = train_df[["grade","price"]].groupby("grade").mean().reset_index()

price_by_grade["tmp"] = price_by_grade.price.shift()

price_by_grade
price_by_grade.loc[0,"tmp"] = 0 

price_by_grade["Flactuation_rate"] = (price_by_grade.price-price_by_grade.tmp)/price_by_grade.price * 100

price_by_grade = price_by_grade.drop("tmp", axis=1)

price_by_grade = price_by_grade.set_index("grade")

price_by_grade 
plt.figure(figsize = (16,8))



sns.boxplot(x = train_df.grade , y=np.log(train_df.price))
train_df[train_df.grade.isin([1,3])]
plt.figure(figsize = (12,8))

sns.boxplot(x= train_df.view, y= np.log(train_df.price))
train_df[["view", "waterfront"]].groupby("view").mean()
pd.DataFrame(train_df[["view", "waterfront"]].groupby(["view","waterfront"]).size())
cnt_plot = sns.countplot(train_df.waterfront)

for p in cnt_plot.patches:

    x = p.get_bbox().get_points()[:,0]

    y = p.get_bbox().get_points()[1,1]

    cnt_plot.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y))
train_df[train_df.sqft_above + train_df.sqft_basement != train_df.sqft_living]

#sqft_living  = sqft_above + sqft_basement
change_rate_15 = abs(train_df.sqft_living+train_df.sqft_lot - train_df.sqft_living15 - train_df.sqft_lot15) / (train_df.sqft_living+train_df.sqft_lot) * 100
f, ax = plt.subplots()

sns.distplot(change_rate_15 , kde = True, bins = 120)

ax.set_xlim(0,100)
living_change_rate_15 = abs(train_df.sqft_living - train_df.sqft_living15) / (train_df.sqft_living)

lot_change_rate_15 = abs(train_df.sqft_lot - train_df.sqft_lot15) / (train_df.sqft_lot)
test_living_change_rate_15 = abs(test_df.sqft_living - test_df.sqft_living15) / (test_df.sqft_living)

test_lot_change_rate_15 = abs(test_df.sqft_lot - test_df.sqft_lot15) / (test_df.sqft_lot)
fig, axes = plt.subplots(nrows=2)

fig.set_size_inches(12,6)



sns.distplot(living_change_rate_15 , kde = True, bins = 120 , label="living_change_rate", ax = axes[0], color = 'blue')

sns.distplot(lot_change_rate_15 , kde = True, bins = 120 , label="lot_change_rate", ax = axes[1], color = 'green')

axes[0].set_xlim(0,1)

axes[0].legend()

axes[1].set_xlim(0,1)

axes[1].legend()
train_df["living_change_rate_15"] = living_change_rate_15

train_df["lot_change_rate_15"] = lot_change_rate_15

train_df.head(10)
test_df["living_change_rate_15"] = test_living_change_rate_15

test_df["lot_change_rate_15"] = test_lot_change_rate_15
"""train_df["living_change_rate_15"] = train_df.living_change_rate_15.astype(int)

train_df["lot_change_rate_15"] =train_df.lot_change_rate_15.astype(int)

train_df.head(10)"""



"""test_df["living_change_rate_15"] = test_df.living_change_rate_15.astype(int)

test_df["lot_change_rate_15"] =test_df.lot_change_rate_15.astype(int)"""
"""tmp_df = pd.DataFrame(living_change_rate_15)

tmp_df.columns = ["rate"]

tmp_df["rate"] = np.round(tmp_df.rate)

tmp_df["rate"] = tmp_df.rate.astype(int)

tmp_df = tmp_df.rate.value_counts()

tmp_df = pd.DataFrame(tmp_df)

tmp_df = tmp_df.reset_index()

tmp_df.columns = ["rate", "counts"]

tmp_df.head(10)



tmp_df["cum_counts"] = tmp_df.counts.cumsum()

tmp_df["percetage"] = tmp_df.counts / train_df.shape[0]

tmp_df = tmp_df.sort_values(by = "rate")

tmp_df["cum_percentage"] = tmp_df.percetage.cumsum() * 100

tmp_df["cum_cat"] = pd.cut(tmp_df.cum_percentage, 5, labels = range(1,6))

tmp_df["cum_cat"] = tmp_df.cum_cat.astype(int)

tmp_df_living = tmp_df[["rate","cum_cat"]]

tmp_df_living.columns = ["living_change_rate_15", "Liv_Change_cat"]

train_df = pd.merge(train_df, tmp_df_living, on = "living_change_rate_15")"""
"""tmp_df = pd.DataFrame(lot_change_rate_15)

tmp_df.columns = ["rate"]

tmp_df["rate"] = np.round(tmp_df.rate)

tmp_df["rate"] = tmp_df.rate.astype(int)

tmp_df = tmp_df.rate.value_counts()

tmp_df = pd.DataFrame(tmp_df)

tmp_df = tmp_df.reset_index()

tmp_df.columns = ["rate", "counts"]



tmp_df["cum_counts"] = tmp_df.counts.cumsum()

tmp_df["percetage"] = tmp_df.counts / train_df.shape[0]

tmp_df = tmp_df.sort_values(by = "rate")

tmp_df["cum_percentage"] = tmp_df.percetage.cumsum() * 100

tmp_df["cum_cat"] = pd.cut(tmp_df.cum_percentage, 5, labels = range(1,6))

tmp_df["cum_cat"] = tmp_df.cum_cat.astype(int)

tmp_df_living = tmp_df[["rate","cum_cat"]]

tmp_df_living.columns = ["lot_change_rate_15", "Lot_Change_cat"]

train_df = pd.merge(train_df, tmp_df_living, on = "lot_change_rate_15")

train_df.head(10)"""
#train_df.drop(["Liv_Change_cat", "Lot_Change_cat"], axis=1, inplace=True)
train_df["has_basement"] = np.where(train_df.sqft_basement != 0, 1, 0)

test_df["has_basement"] = np.where(test_df.sqft_basement != 0, 1, 0)
plt.figure(figsize = (20,12))

sns.scatterplot(x=train_df.sqft_living, y=(train_df.sqft_living - train_df.sqft_living15), hue=train_df.has_basement, size =1 , alpha = 0.6)
plt.figure(figsize = (20,12))

sns.lmplot(x="sqft_living", y="price", hue="has_basement", data=train_df)
plt.figure(figsize=(15,15))

cor = train_df.corr(method="spearman")

sns.heatmap(cor, square=True ,cmap = "YlGnBu", annot=True, vmax=.8)
plt.figure(figsize=(12,8))

cnt_plot = sns.countplot(train_df.condition)

for p in cnt_plot.patches:

    x = p.get_bbox().get_points()[:,0]

    y = p.get_bbox().get_points()[1,1]

    cnt_plot.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y))
pd.DataFrame(train_df[["condition", "grade"]].groupby(["condition","grade"]).size()).reset_index().pivot("condition","grade",0)
f = sns.FacetGrid(data = train_df, row="condition", height=2.4, aspect=5)

f.map(sns.countplot,"grade")
plt.figure(figsize = (16,10))

sns.countplot(train_df.condition, hue=train_df.grade)
train_df["sqft_all"] = train_df.sqft_living + train_df.sqft_lot

train_df["sqft_all_15"] = train_df.sqft_living15 + train_df.sqft_lot15



test_df["sqft_all"] = test_df.sqft_living + test_df.sqft_lot

test_df["sqft_all_15"] = test_df.sqft_living15 + test_df.sqft_lot15
livingByLot = np.divide(train_df.sqft_living, train_df.sqft_lot)

livingByLot15 = np.divide(train_df.sqft_living15, train_df.sqft_lot15)



train_df["LivingByLot"] = livingByLot

train_df["LivingByLot15"] = livingByLot15



test_livingByLot = np.divide(test_df.sqft_living, test_df.sqft_lot)

test_livingByLot15 = np.divide(test_df.sqft_living15, test_df.sqft_lot15)



test_df["LivingByLot"] = test_livingByLot

test_df["LivingByLot15"] = test_livingByLot15
fig, ax = plt.subplots()

fig.set_size_inches(10,5)

sns.distplot(livingByLot, ax = ax, kde=True, label = "Original")

sns.distplot(livingByLot15, ax = ax, kde=True, label = "Changed at 2015")

ax.set_xlim(0,2)

ax.legend()
fig, ax = plt.subplots(figsize=(12,8))

ax.hist([livingByLot, livingByLot15], histtype = 'bar', bins=60, align='mid', label=['Original', 'Changed at 15'], alpha = 0.5)

ax.set_xlim(0,2)

ax.legend()
tmp_index = livingByLot[livingByLot >= 1].index

train_df.iloc[tmp_index,:][["floors"]].groupby("floors").size()
sns.distplot(np.log(train_df[(train_df.floors % 1 == 0)].price))
sns.distplot(np.log(train_df[(train_df.floors % 1 != 0)].price))
plt.figure(figsize = (6,10))

sns.scatterplot(x="long", y="lat", hue="price", data=train_df )
train_df["price_per_sqft"] = train_df.price / train_df.sqft_living
%%time

fig, axes = plt.subplots(ncols = 5)

fig.set_size_inches(30,10)

for i in range(5,10):

    price_cut = pd.cut(train_df.price_per_sqft, i, labels = range(1,i+1))

    train_df["sqft_price_range"] = price_cut

    train_df["sqft_price_range"] = train_df.sqft_price_range.astype(int)

    sns.scatterplot(x="long", y="lat", hue="sqft_price_range",data=train_df, ax = axes[i-5], size =1, alpha=0.6)
#train_df.drop("sqft_price_range", axis=1, inplace = True)

#N값 변화시키면서 몇개의 cluster로 나눌지 생각
fig, ax = plt.subplots()

fig.set_size_inches(6,10)

sns.scatterplot(x= "long", y="lat", hue="sqft_price_range", data = train_df, size =1.5, ax = ax)

sns.scatterplot(x="long", y="lat",data = train_df[["long","lat","zipcode","price_per_sqft"]].groupby("zipcode").mean(), ax = ax)

#train_df[["long","lat","zipcode","price_per_sqft"]].groupby("zipcode").mean()
#train_df[["long","lat","zipcode","sqft_price_range"]].groupby("zipcode").mean()

tmp_cut = pd.DataFrame(pd.cut(train_df[["long","lat","zipcode","sqft_price_range"]].groupby("zipcode").mean().sqft_price_range, 10, labels = range(1,11))).reset_index()

tmp_cut.columns = ["zipcode", "cluster"]
tmp_point_df = pd.merge(train_df[["long","lat","zipcode","sqft_price_range"]].groupby("zipcode").mean().reset_index(), tmp_cut, on = "zipcode")

tmp_point_df["cluster"] = tmp_point_df.cluster.astype(int)
zip_cluter = tmp_point_df[["zipcode", "cluster"]]



fig, ax = plt.subplots()

fig.set_size_inches(6,10)

sns.scatterplot(x= "long", y="lat", hue="sqft_price_range", data = train_df, size =1.5, ax = ax)

sns.scatterplot(x="long", y="lat",data = tmp_point_df, hue="cluster", ax = ax, palette=['black','red','brown','green','skyblue','blue','magenta','yellow','gold'], legend='full')
#zip_cluster의 개수를 바꿔가며 모델의 성능을 봐야할 듯
train_df = pd.merge(train_df, zip_cluter, on = "zipcode")

test_df = pd.merge(test_df, zip_cluter, on = "zipcode")
mean_price_By_zipcode = train_df.groupby("zipcode").mean()["price_per_sqft"].reset_index()

mean_price_By_zipcode.columns = ["zipcode", "mean_price"]

var_price_By_zipcode = np.square(train_df.groupby("zipcode").std()["price_per_sqft"]).reset_index()

var_price_By_zipcode.columns = ["zipcode", "var_price"]
plt.figure(figsize=(16,10))

sns.boxplot(x="cluster", y="mean_price", data = pd.merge(mean_price_By_zipcode, zip_cluter))
tmp = pd.merge(mean_price_By_zipcode, var_price_By_zipcode)

train_df = pd.merge(train_df, tmp)



test_df = pd.merge(test_df, tmp)
#grade, condition, view, floors, bedrooms, bathrooms
cat_cols = ["grade", "condition", "view", "floors", "bedrooms", "bathrooms"]



fig , axes = plt.subplots(ncols=2, nrows =6)

fig.set_size_inches(24,36)

i = 0

for col in cat_cols:

    mean_tmp_df = train_df.groupby(col).mean()["price_per_sqft"].reset_index()

    var_tmp_df = np.square(train_df.groupby(col).std()["price_per_sqft"]).reset_index()

    sns.barplot(x = col , y = "price_per_sqft", data = mean_tmp_df ,ax = axes[i][0])

    sns.barplot(x = col , y = "price_per_sqft", data = mean_tmp_df ,ax = axes[i][1])

    i += 1
plt.figure(figsize = (20,20))

cor = train_df.corr(method="spearman")

sns.heatmap(cor, square=True, annot=True, vmax=.8, cmap = "YlGnBu")
import scipy as sp



cor_abs = abs(train_df.corr(method='spearman')) 

cor_cols = cor_abs.nlargest(n=10, columns='price').index # price과 correlation이 높은 column 10개 뽑기(내림차순)

# spearman coefficient matrix

cor = np.array(sp.stats.spearmanr(train_df[cor_cols].values))[0] # 10 x 10

print(cor_cols.values)

plt.figure(figsize=(10,10))

sns.set(font_scale=1.25)

sns.heatmap(cor, fmt='.2f', annot=True, square=True , annot_kws={'size' : 8} ,xticklabels=cor_cols.values, yticklabels=cor_cols.values)
train_df.drop(["sqft_price_range","price_per_sqft"], axis=1, inplace=True)
skewed_col = ["mean_price","var_price","sqft_living", "sqft_lot", "sqft_living15","sqft_lot15", "sqft_all", "sqft_all_15", "LivingByLot","LivingByLot15", "living_change_rate_15", "lot_change_rate_15", "sqft_above","sqft_basement"]
for col in skewed_col:

    train_df[col] = np.log1p(train_df[col].values)



for col in skewed_col:

    test_df[col] = np.log1p(test_df[col].values)
print("Shape of Train_df : {} \nShape of Test_df : {}".format(train_df.shape, test_df.shape))
train_df = train_df.sort_values(by="id").set_index("id").reset_index()

test_df = test_df.sort_values(by="id").set_index("id").reset_index()

#train_X, valid_X, train_y, valid_y = train_test_split(train_df.drop("price", axis=1), train_df.price, test_size=0.2, random_state=123)
train_columns = train_df.drop(["id","price"], axis=1).columns.tolist()
train_df["price"] = np.log(train_df.price)
param = {'num_leaves': 31,

         'min_data_in_leaf': 30, 

         'objective':'regression',

         'max_depth': -1,

         'learning_rate': 0.02,

         "min_child_samples": 20,

         "boosting": "gbdt",

         "feature_fraction": 0.9,

         "bagging_freq": 1,

         "bagging_fraction": 0.9 ,

         "bagging_seed": 11,

         "metric": 'rmse',

         "lambda_l1": 0.1,

         "verbosity": -1,

         "nthread": 4,

         "random_state": 4950}



y_reg = train_df['price']



folds = KFold(n_splits=10, shuffle=True, random_state=42)



oof = np.zeros(len(train_df))

predictions = np.zeros(len(test_df))

feature_importance_df = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df)):

    trn_data = lgb.Dataset(train_df.iloc[trn_idx][train_columns], label=y_reg.iloc[trn_idx])#, categorical_feature=categorical_feats)

    val_data = lgb.Dataset(train_df.iloc[val_idx][train_columns], label=y_reg.iloc[val_idx])#, categorical_feature=categorical_feats)

    

    num_round = 10000

    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data,val_data], 

                   verbose_eval = 500, early_stopping_rounds= 100)

    oof[val_idx] = clf.predict(train_df.iloc[val_idx][train_columns], num_iteration = clf.best_iteration)

    

    fold_importance_df = pd.DataFrame()

    fold_importance_df["Feature"] = train_columns

    fold_importance_df["importance"] = clf.feature_importance()

    fold_importance_df["fold"] = fold_ + 1

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    #predictions

    predictions += clf.predict(test_df[train_columns], num_iteration=clf.best_iteration) / folds.n_splits

    

cv = np.sqrt(mean_squared_error(oof, y_reg))

print(cv)

    
lgb_predictions = np.exp(predictions)

lgb_predictions
xgb_params = {

    'max_depth' : 6,

    'learning_rate' : 0.02,

    'n_estimator' : 100,

    'objective' : 'reg:squarederror',

    'eval_metric' : 'rmse',

    'colsample_bytree' : 0.6,

    'subsample' : 0.8,

    'seed' : 123

}
train_X = train_df[train_columns]

train_y = train_df['price']

test_X = test_df[train_columns]
dtrain = xgb.DMatrix(train_X, train_y)

dtest = xgb.DMatrix(test_X)



cv_output = xgb.cv(xgb_params,

                  dtrain,

                  num_boost_round=5000,

                  early_stopping_rounds=300,

                  nfold=5,

                  verbose_eval=100,

                  show_stdv = False)

cv_output.columns
best_rounds = cv_output.index.size

score = round(cv_output.iloc[-1]['test-rmse-mean'], 2)



# plotting

fig, ax1 = plt.subplots(1, 1, figsize=(14,5))

cv_output[['train-rmse-mean', 'test-rmse-mean']].plot(ax=ax1)

ax1.set_title('RMSE_log', fontsize=20)
xgb_model = xgb.train(xgb_params, dtrain, num_boost_round=best_rounds)
xgb_pred = xgb_model.predict(dtest)

xgb_pred = np.exp(xgb_pred)
sub = pd.DataFrame({'id' : test_df.id.values,

             'price' : lgb_predictions})
xgb_sub = pd.DataFrame({'id' : test_df.id.values,

             'price' : xgb_pred})
sub.to_csv('submission.csv', index=False)

xgb_sub.to_csv('xgb_submisson.csv', index=False)