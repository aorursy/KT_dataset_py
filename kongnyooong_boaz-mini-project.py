import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os

from collections import Counter



sns.set_style('whitegrid')

sns.set(font_scale = 1.2)



import missingno as msno

import warnings 

warnings.filterwarnings("ignore")



%matplotlib inline
os.listdir("../input")
df_train = pd.read_csv("../input/boazminipro/train.csv")

df_test = pd.read_csv("../input/boazminipro/test.csv")
df_train.head()
df_train.shape
numerical_feats = df_train.dtypes[df_train.dtypes != "object"].index

print("Numerical Features: ", len(numerical_feats))



categorical_feats = df_train.dtypes[df_train.dtypes == "object"].index

print("Categorical Features: ",len(categorical_feats))



# 범주형 변수가 없다 (숫자로 이루어진 범주형 변수가 있다. ex.등급)

# 하나있는 범주형 변수는  ["date"]이다. datetime으로 설정한다.
for col in df_train.columns:

    Null = "Feature: {:<10}\t Count of Null: {}".format(col, df_train[col].isnull().sum())

    print(Null)
msno.matrix(df = df_train.iloc[:,:], color = (0.1, 0.4, 0.5), figsize = (15, 6))



#missingno의 matrix로 시각화한 모습, 역시 null값은 없다.
numerical_feats = df_train.dtypes[df_train.dtypes != "object"].index

print("Numerical Features: ", len(numerical_feats))



for col in numerical_feats.difference(["date"]):

    SkewKurt = "{:<10}\t Skewness: {:.4f}\t Kurtosis: {:.4f}".format(col, df_train[col].skew(), df_train[col].kurt())

    print(SkewKurt)

    

#각 feature들의 왜도와 첨도를 살펴본다. 

#target feature인 price 역시 조정이 필요.

#처음에 수치형, 범주형 구분할 때 말했던 범주가 수치(등급)로 지정되어있는 변수들을 주의한다.
df_train["price"] = df_train["price"].map(lambda i:np.log(i) if i>0 else 0)
corr_data = df_train[numerical_feats]



colormap = plt.cm.PuBu

sns.set(font_scale=1.3)



f , ax = plt.subplots(figsize = (14,12))

plt.title('Correlation of Numeric Features with Price',size=18)

sns.heatmap(corr_data.corr(),square = True, linewidths = 0.1,

            cmap = colormap, linecolor = "white", vmax=0.8)



#전체적인 correlation heatmap
k= 12

cols = corr_data.corr().nlargest(k,'price')['price'].index

print(cols)

cm = np.corrcoef(df_train[cols].values.T)

f , ax = plt.subplots(figsize = (12,10))

sns.heatmap(cm, vmax=.8, linewidths=0.1,square=True,annot=True,cmap=colormap,

            linecolor="white",xticklabels = cols.values ,annot_kws = {'size':14},yticklabels = cols.values)



#전체적인 correlation heatmap에서 상관계수가 높은 순으로 12개를 뽑아서 다시 만든 heatmap
df_train.plot(kind='scatter', x='long', y='lat', alpha=.3, figsize=(10,7),

         c=df_train['price'], cmap=plt.get_cmap('jet'), colorbar=True)
fig, ((ax1, ax2), (ax3, ax4),(ax5,ax6)) = plt.subplots(nrows=3, ncols=2, figsize=(18,15))



sqft_living_scatter_plot = pd.concat([df_train['price'],df_train['sqft_living']],axis = 1)

sns.regplot(x='sqft_living',y = 'price',data = sqft_living_scatter_plot,scatter= True, fit_reg=True, ax=ax1)



sqft_lot_scatter_plot = pd.concat([df_train['price'],df_train['sqft_lot']],axis = 1)

sns.regplot(x='sqft_lot',y = 'price',data = sqft_lot_scatter_plot,scatter= True, fit_reg=True, ax=ax2)



sqft_above_scatter_plot = pd.concat([df_train['price'],df_train['sqft_above']],axis = 1)

sns.regplot(x='sqft_above',y = 'price',data = sqft_above_scatter_plot,scatter= True, fit_reg=True, ax=ax3)



sqft_basement_scatter_plot = pd.concat([df_train['price'],df_train['sqft_basement']],axis = 1)

sns.regplot(x='sqft_basement',y = 'price',data = sqft_basement_scatter_plot,scatter= True, fit_reg=True, ax=ax4)



sqft_living15_scatter_plot = pd.concat([df_train['price'],df_train['sqft_living15']],axis = 1)

sns.regplot(x='sqft_living15',y = 'price',data = sqft_living15_scatter_plot,scatter= True, fit_reg=True, ax=ax5)



sqft_lot15_scatter_plot = pd.concat([df_train['price'],df_train['sqft_lot15']],axis = 1)

sns.regplot(x='sqft_lot15',y = 'price',data = sqft_lot15_scatter_plot,scatter= True, fit_reg=True, ax=ax6)



# 수치형 변수들에 따른 price를 scatter plot으로 그려본다. 

# sqft_living, sqft_above, sqft_living15는 어느정도 이상치가 있어보인다.

# 나머지들은 분산이 매우 커보인다. 

# basement는 0의 값이 굉장히 많다. 또한 이상치도 많아보인다.
df_train[["grade", "price"]].groupby(["grade"], as_index = True).describe()



# 등급에 따른 주택가격
fig, ax = plt.subplots(figsize = (12, 8))



sns.boxplot(x = df_train["grade"], y = df_train["price"], data = df_train, ax = ax, palette = "Blues_d")



# grade에 따른 price의 boxplot을 그려본 결과 

# 1) 2등급은 존재하지 않음

# 2) 3등급의 평균값이 4등급보다 높게 나타남

# 3) 7, 8, 11등급에 상당한 이상치가 존재하는 것으로 보임

# 4) 7~10 등급의 이상치가 꽤 많은 것으로 보임



# 데이터 설명에 따르면 grade의 경우 "1-3은 건물 건축 및 디자인에 미치지 못하고 7은 평균 수준의 건축 및 디자인을, 11-13은 높은 수준의 건축 및 디자인을 지니고 있습니다." 
df_train[["bedrooms", "price"]].groupby(["bedrooms"], as_index = True).describe()



# 방의 수에 따른 주택 가격
fig, ax = plt.subplots(figsize = (16, 10))



sns.boxplot(x = df_train["bedrooms"], y = df_train["price"], data = df_train, ax = ax, palette = "Blues_d")



# boxplot을 살펴보면 방의 수에 따른 가격이 직관적으로 대략 선형임을 알 수 있음.

# 윗 셀의 describe를 봐도 가격의 평균값이 방에 따라 증가하는 것을 볼 수 있음.

# boxplot을 벗어난 이상치들은 지역특성(땅값이 비싸거나 대도시이거나?)에 영향을 받는다고 생각할 수 있다.
df_train[["bathrooms", "price"]].groupby(["bathrooms"], as_index = True).describe()



# 데이터 설명에 따르면

# - 0.5 : 세면대, 화장실

# - 0.75 : 세면대, 화장실, 샤워실

# - 1 : 세면대, 화장실, 샤워실, 욕조

# 의 값을 갖는다고 한다.
fig, ax = plt.subplots(figsize = (16, 10))



sns.boxplot(x = df_train["bathrooms"], y = df_train["price"], data = df_train, ax = ax, palette = "Blues_d")
df_train[["floors", "price"]].groupby(["floors"], as_index = True).describe()



# 층수의 경우 1.5, 2.5, 3.5와 같이 소숫점을 가진다. 

# 미국에서 흔히 볼 수 있는 형태로 다락방을 끼고 있는 형태

# floors, price는 선형관계로 보임
df_train[["waterfront", "price"]].groupby(["waterfront"], as_index = True).describe()



# 바이너리 (리버뷰? 등이 있고 없고)

# waterfront, price는 선형관계로 보임
df_train[["view", "price"]].groupby(["view"], as_index = True).describe()



# view, price는 선형관계로 보임
df_train[["condition", "price"]].groupby(["condition"], as_index = True).describe()



# condition, price는 선형관계로 보임
for data in [df_train, df_test]:

    data['date'] = pd.to_datetime(data['date'])

    data['date_year'] = data['date'].dt.year

    data['date_month'] = data['date'].dt.month
train_pivot = df_train.pivot_table(index=['date_month'], columns=['date_year'], values='price') # price mean

plt.figure(figsize=(10,8))

sns.heatmap(train_pivot, annot=True, cmap = colormap)
plt.figure(figsize=(10,5))

sns.lineplot(x=df_train['date_month'], y=df_train['price'])
df_train.loc[df_train.grade==3]



# EDA 과정에서 살펴본 grade = 3인 이상치 제거
print(df_train[df_train['grade']==3].sqft_lot.mean())

print(df_train[df_train['grade']==4].sqft_lot.mean())
df_train.drop([2302,4123], axis=0, inplace=True)
df_train.loc[(df_train.grade==11) & (df_train.price>15.5)]



# grade 11 확인, sqft_living이 크므로 제거하지 않음
df_train[(df_train.grade)==11].sqft_living.max()
df_train.loc[(df_train.bathrooms==4.5)&(df_train.price>15)]



# bathroom 이상치, 제거 X
df_train.loc[(df_train.bathrooms==5.25)&(df_train.price<13)]



# bathroom 이상치, 제거 X
df_train[df_train.sqft_living > 13000]



# sqft_living 이상치 제거
df_train.drop(8912, axis = 0, inplace = True)
df_train.loc[(df_train.sqft_lot>1500000)&(df_train.price>13)]



# sqft_lot 이상치 제거
df_train.drop(1231, axis = 0, inplace = True)
df_train["sqft_above"] = df_train["sqft_above"].map(lambda i:np.log(i) if i>0 else 0)

df_train["sqft_basement"] = df_train["sqft_basement"].map(lambda i:np.log(i) if i>0 else 0)

df_train["sqft_living"] = df_train["sqft_living"].map(lambda i:np.log(i) if i>0 else 0)

df_train["sqft_living15"] = df_train["sqft_living15"].map(lambda i:np.log(i) if i>0 else 0)

df_train["sqft_lot"] = df_train["sqft_lot"].map(lambda i:np.log(i) if i>0 else 0)

df_train["sqft_lot15"] = df_train["sqft_lot15"].map(lambda i:np.log(i) if i>0 else 0)



df_test["sqft_above"] = df_test["sqft_above"].map(lambda i:np.log(i) if i>0 else 0)

df_test["sqft_basement"] = df_test["sqft_basement"].map(lambda i:np.log(i) if i>0 else 0)

df_test["sqft_living"] = df_test["sqft_living"].map(lambda i:np.log(i) if i>0 else 0)

df_test["sqft_living15"] = df_test["sqft_living15"].map(lambda i:np.log(i) if i>0 else 0)

df_test["sqft_lot"] = df_test["sqft_lot"].map(lambda i:np.log(i) if i>0 else 0)

df_test["sqft_lot15"] = df_test["sqft_lot15"].map(lambda i:np.log(i) if i>0 else 0)
logdata = df_train[["price", "sqft_above", "sqft_basement", "sqft_living", "sqft_living15", "sqft_lot", "sqft_lot15"]]



for i in range(7):

    print("{:<10}\t Skewness: {:.3f}\t Kurtosis: {:.3f}".format(logdata.columns[i], df_train[logdata.columns[i]].skew(), df_train[logdata.columns[i]].kurt()))

    

# log를 취해주어 분포를 조정해준다. 

# 수치적으로 정규분포에 가까워진 것을 확인할 수 있다.
df_train["total_sqft"] = df_train["sqft_above"] + df_train["sqft_basement"]

df_test["total_sqft"] = df_test["sqft_above"] + df_test["sqft_basement"]
df_train[["total_sqft", "sqft_living"]].head()



# 건물의 총 면적을 만들어서 거주공간 면적과 비교해보면 값이 같다. 

# 이는 sqft_living 변수가 건물의 연면적을 의미함을 알 수 있다.
df_train.drop(["total_sqft"], inplace = True, axis = 1)

df_test.drop(["total_sqft"], inplace = True, axis = 1)
df_train["Vol_ratio"] = (df_train["sqft_living"] / df_train["sqft_lot"]) * 100

df_test["Vol_ratio"] = (df_test["sqft_living"] / df_test["sqft_lot"]) * 100



# 용적률 = 건물연면적 / 토지면적 * 100

# 건폐율을 사용했을 때 성능이 좋지않아 제거
for data in [df_train, df_test]:



    data['above_per_living'] = data['sqft_above']/data['sqft_living']
zipcode_data = df_train.groupby('zipcode').aggregate(np.mean)



zipcode_ranks = {}

rank = 1

for idx, row in zipcode_data.sort_values(by='price').iterrows():

    zipcode_ranks[idx] = rank

    rank += 1

    

# zipcode별로 price의 평균을 내어 rank를 매겨준다 (집값이 낮으면 1 올라갈수록 +)
for data in [df_train, df_test]:

    zipcode_feature = []

    for idx, row in data.iterrows():

        zipcode_feature.append(zipcode_ranks[row.zipcode])

    data['zipcode_ranks'] = zipcode_feature
zipcode_data = df_train.groupby('zipcode').aggregate(np.var)



zipcode_ranks_var = {}

rank = 1

for idx, row in zipcode_data.sort_values(by='price', ascending=False).iterrows():

    zipcode_ranks_var[idx] = rank

    rank +=1
for data in [df_train, df_test]:

    zipcode_feature = []

    for idx, row in data.iterrows():

        zipcode_feature.append(zipcode_ranks_var[row.zipcode])

    data['zipcode_ranks_var'] = zipcode_feature
month = df_train.groupby('date_month').aggregate(np.mean)



month_ranks = {}

rank = 1

for idx, row in month.sort_values(by='price').iterrows():

    month_ranks[idx] = rank

    rank += 1
for data in [df_train, df_test]:

    month_feature = []

    for idx, row in data.iterrows():

        month_feature.append(month_ranks[row.date_month])

    data['month_rank'] = month_feature
from haversine import haversine

bridge_wh = (47.641076, -122.259196)

for data in [df_train, df_test]:

    house_wh = data.loc[:, ['lat','long']]

    house_wh = list(house_wh.itertuples(index=False, name=None))

    

    dist = []

    for house in house_wh:

        dist.append(np.log(1/haversine(house, bridge_wh)))

    data['dist_bridge'] = dist
from sklearn.decomposition import PCA

from sklearn.cluster import KMeans
for data in [df_train, df_test]:

    coord = data[['lat','long']]

    pca_coord = PCA(n_components=2).fit(coord).transform(coord)

    data['pca1'] = pca_coord[:, 0]

    data['pca2'] = pca_coord[:, 1]

    

# 위경도를 기준으로 사용하여 pca를 진행하여 새로운 변수를 만들어준다. 
for data in [df_train, df_test]:

    

    data['term'] = -(data.date_year - data.yr_built)
df_train.columns
df_train.shape
df_train.drop(["date", "date_month"], inplace = True, axis = 1)

df_test.drop(["date", "date_month"], inplace = True, axis = 1)
df_train.head()
import eli5

from eli5.sklearn import PermutationImportance

from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV

from sklearn.metrics import mean_squared_error

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

import xgboost as xgb

import lightgbm as lgb
y_train = df_train.price

x_train = df_train.drop(['id', 'price'], axis=1)

x_test = df_test.drop(['id'], axis=1)
X_tr, X_vld, y_tr, y_vld = train_test_split(x_train, y_train, test_size = 0.3, random_state = 2019)



model = xgb.XGBRegressor()

model.fit(X_tr, y_tr)

y_val = model.predict(X_vld)
perm = PermutationImportance(model, random_state = 42).fit(X_vld, y_vld)

eli5.show_weights(perm, top = 33, feature_names = X_vld.columns.tolist())
dt = DecisionTreeRegressor(random_state=10)

dt_cv_score = cross_val_score(dt, x_train, y_train, cv=5)

print("score with cv = {} \n mean cv score = {:.5f} \n std = {:.5f}".format(dt_cv_score, dt_cv_score.mean(), dt_cv_score.std()))
xgb_params = {

    'eta': 0.01,

    'max_depth': 6,

    'subsample': 0.8,

    'colsample_bytree': 0.8,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'silent': 1

}
y_train = df_train.price



dtrain = xgb.DMatrix(x_train, y_train)

dtest = xgb.DMatrix(x_test)



cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=5000, early_stopping_rounds=50, verbose_eval=500, show_stdv=False)



rounds=len(cv_output)



xgb1 = xgb.train(xgb_params, dtrain, num_boost_round=rounds)

preds = xgb1.predict(dtest)



xgb1_sub = df_test[['id']]

xgb1_sub['price'] = preds
y_train = np.expm1(df_train.price)



dtrain = xgb.DMatrix(x_train, y_train)

dtest = xgb.DMatrix(x_test)



cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=20000, early_stopping_rounds=50, verbose_eval=500, show_stdv=False)

rounds=len(cv_output)



xgb2 = xgb.train(xgb_params, dtrain, num_boost_round=rounds)

preds = xgb2.predict(dtest)



xgb2_sub = df_test[['id']]

xgb2_sub['price'] = preds
xgb1_pred = xgb1.predict(dtrain)

xgb2_pred = xgb2.predict(dtrain)

mse = {}

ii = np.arange(0, 1, 0.01)

for i, ii in enumerate(ii):

    xgb_train_pred = ii*np.expm1(xgb1_pred) + (1-ii)*xgb2_pred

    mse[i] = np.sqrt(mean_squared_error(y_train, xgb_train_pred))



xgb_min = min(mse.values())



for i in range(100):

    if mse[i] == xgb_min:

        print(i)
xgb_train_pred = 0*np.expm1(xgb1_pred) + 1*xgb2_pred
xgb_sub = pd.merge(xgb1_sub, xgb2_sub, how='left', on='id')

xgb_sub.columns = ['id','price1','price2']

xgb_sub['price'] = (0*np.expm1(xgb_sub['price1']) + 1*xgb_sub['price2'])

xgb_sub = xgb_sub[['id','price']]

xgb_sub.to_csv('xgb_sub.csv', index=False)
param = {'num_leaves': 31,

         'min_data_in_leaf': 30, 

         'objective':'regression',

         'max_depth': -1,

         'learning_rate': 0.015,

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

         "random_state": 4950

}
y_train = df_train.price



folds = KFold(n_splits=5, shuffle=True, random_state=1)

predictions = np.zeros(len(x_test))



for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_train)):

    trn_data = lgb.Dataset(x_train.iloc[trn_idx], label=y_train.iloc[trn_idx])

    val_data = lgb.Dataset(x_train.iloc[val_idx], label=y_train.iloc[val_idx])

    

    num_round = 10000

    lgb1 = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=500, early_stopping_rounds = 100)

    predictions += lgb1.predict(x_test, num_iteration=lgb1.best_iteration) / folds.n_splits
lgb1_sub = pd.DataFrame()

lgb1_sub['id'] = df_test.id

lgb1_sub['price'] = predictions
y_train = np.expm1(df_train.price)



folds = KFold(n_splits=5, shuffle=True, random_state=1)

predictions = np.zeros(len(x_test))



for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_train)):

    trn_data = lgb.Dataset(x_train.iloc[trn_idx], label=y_train.iloc[trn_idx])

    val_data = lgb.Dataset(x_train.iloc[val_idx], label=y_train.iloc[val_idx])

    

    num_round = 10000

    lgb2 = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=500, early_stopping_rounds = 100)

    predictions += lgb2.predict(x_test, num_iteration=lgb2.best_iteration) / folds.n_splits
lgb2_sub = pd.DataFrame()

lgb2_sub['id'] = df_test.id

lgb2_sub['price'] = predictions
lgb1_pred = lgb1.predict(x_train)

lgb2_pred = lgb2.predict(x_train)

mse = {}

ii = np.arange(0, 1, 0.01)

for i, ii in enumerate(ii):

    train_pred = ii*np.expm1(lgb1_pred) + (1-ii)*lgb2_pred

    mse[i] = np.sqrt(mean_squared_error(y_train, train_pred))



lgb_min = min(mse.values())



for i in range(100):

    if mse[i] == lgb_min:

        print(i)
lgb_train_pred = 0.59*np.expm1(lgb1_pred)+0.41*lgb2_pred
lgb_sub = pd.merge(lgb1_sub, lgb2_sub, how='left', on='id')

lgb_sub.columns = ['id','price1','price2']

lgb_sub['price'] = (0.59*np.expm1(lgb_sub['price1']) + 0.41*lgb_sub['price2'])

lgb_sub = lgb_sub[['id','price']]
forest_regr = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,

           max_features=28, max_leaf_nodes=None,

           min_impurity_decrease=0.0, min_impurity_split=None,

           min_samples_leaf=1, min_samples_split=2,

           min_weight_fraction_leaf=0.0, n_estimators=90, n_jobs=1,

           oob_score=False, random_state=None, verbose=0, warm_start=False)



rf = forest_regr.fit(x_train, y_train)

predictions = rf.predict(x_test)
rf_sub = pd.DataFrame()

rf_sub['id'] = df_test.id

rf_sub['price'] = predictions
gdb = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5)



gdb.fit(x_train, y_train)

predictions = gdb.predict(x_test)
gdb_sub = pd.DataFrame()

gdb_sub['id'] = df_test.id

gdb_sub['price'] = predictions
gdb_train_pred = gdb.predict(x_train)

rf_train_pred = rf.predict(x_train)

mse = {}



ii = np.arange(0, 0.1, 0.01)



for i, ii in enumerate(ii):

    submse={}

    train_pred = 0.8*xgb_train_pred + (1-ii)*lgb_train_pred + 0.1*gdb_train_pred + ii*rf_train_pred

    mse[i] = np.sqrt(mean_squared_error(y_train, train_pred))
train_min = min(mse.values())



for i in range(10):

    if mse[i] == train_min:

        print(i)
ensemble_sub = pd.DataFrame()



ensemble_sub['id'] = df_test.id

ensemble_sub['price'] = xgb_sub['price']*0.8 + lgb_sub['price']*0.05 + gdb_sub['price']*0.1 + rf_sub['price']*0.05  

ensemble_sub.to_csv('sub.csv', index=False)