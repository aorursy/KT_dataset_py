# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# from subprocess import check_output

# print(check_output(['ls', '../input']).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv', parse_dates=["date"])

test = pd.read_csv('../input/test.csv', parse_dates=["date"])
train.info()
print(train.shape)

train.head()
train.columns
train.describe()
test.describe()
train['price'].describe()
sns.distplot(train["price"])
train["log_price"] = np.log(train["price"] + 1)

print(train.shape)

train[["price", "log_price"]].head()
figure, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

figure.set_size_inches(18, 4)

sns.distplot(train["price"], ax=ax1)

sns.distplot(train["log_price"], ax=ax2)
train_corr = train.corr()   

print(train_corr.shape)

train_corr.head()
train_corr_abs = abs(train_corr)   # 절대값으로 변환

print(train_corr_abs.shape)

train_corr_abs
f,ax = plt.subplots(figsize=(20,20))

sns.heatmap(train_corr_abs, annot = True, linewidth=.4, fmt='.1f', ax=ax)

plt.show()
train_corr_price_abs = train_corr_abs[['log_price', 'price']].sort_values(by='log_price', ascending=False)

print(train_corr_price_abs.shape)

train_corr_price_abs
f,ax = plt.subplots(figsize=(20,20))

sns.heatmap(train_corr_price_abs, annot = True, linewidth=.4, fmt='.1f', ax=ax)

plt.show()
train_corr_price_abs.index
train["date-year"] = train["date"].dt.year

test["date-year"] = test["date"].dt.year
sns.barplot(data=train, x="date-year", y="price")
train[["date", "date-year", "yr_built"]].head()
train["period_built"] = train["date-year"] - train["yr_built"] +1

train[["date", "date-year", "yr_built", "period_built"]].head()
train['period_built'].describe()
fig = plt.figure()

ax = fig.add_subplot(111)

ax.scatter(train['period_built'], train['price'], alpha = 0.3)

plt.show()
test[["date", "date-year", "yr_built"]].head()
test["period_built"] = test["date-year"] - test["yr_built"] +1

test[["date", "date-year", "yr_built", "period_built"]].head()
train['sqft_living15'].describe()
train['sqft_living15_trans'] = (train['sqft_living15'] * 0.0281).astype(int) # 평수로 전환

train[['sqft_living15', 'sqft_living15_trans']].head()
fig = plt.figure()

ax = fig.add_subplot(111)

ax.scatter(train['sqft_living15_trans'], train['price'], alpha = 0.3)

plt.show()
test['sqft_living15_trans'] = (test['sqft_living15'] * 0.0281).astype(int)

test[['sqft_living15', 'sqft_living15_trans']].head()
train['sqft_above'].describe()
train['sqft_above_trans'] = (train['sqft_above'] * 0.0281).astype(int)

train[['sqft_above', 'sqft_above_trans']].head()
fig = plt.figure()

ax = fig.add_subplot(111)

ax.scatter(train['sqft_above_trans'], train['price'], alpha = 0.3)

plt.show()
test['sqft_above_trans'] = (test['sqft_above'] * 0.0281).astype(int)

test[['sqft_above', 'sqft_above_trans']].head()
train['grade'].describe()
sns.barplot(data=train, x="grade", y="price")
train['view'].describe()
sns.barplot(data=train, x="view", y="price")
train['condition'].describe()
sns.barplot(data=train, x="condition", y="price")
train_corr = train.corr()   

print(train_corr.shape)

train_corr.head()
train_corr_abs = abs(train_corr)   # 절대값으로 변환

print(train_corr_abs.shape)

train_corr_abs
f,ax = plt.subplots(figsize=(20,20))

sns.heatmap(train_corr_abs, annot = True, linewidth=.4, fmt='.1f', ax=ax)

plt.show()
train_corr_price_abs = train_corr_abs[['log_price', 'price']].sort_values(by='log_price', ascending=False)

print(train_corr_price_abs.shape)

print(train_corr_price_abs)

train_corr_price_abs2 = train_corr_abs[['log_price', 'price']].sort_values(by='price', ascending=False)

print(train_corr_price_abs2.shape)

print(train_corr_price_abs2)
f,ax = plt.subplots(figsize=(20,20))

sns.heatmap(train_corr_price_abs, annot = True, linewidth=.4, fmt='.1f', ax=ax)

plt.show()
train_corr_price_abs.index
print(train.shape)

train
print(test.shape)

test
train.columns
feature_names = ['grade', 'sqft_living',

       'sqft_above_trans', 'bathrooms',

       'lat', 'long', 'bedrooms', 'view', 'floors', 'sqft_basement', 'waterfront',

       'yr_renovated', 'period_built', 'sqft_lot', 

       'condition', 'date-year'] 



feature_names
label_name = "log_price"

label_name
X_train = train[feature_names]

print(X_train.shape)

X_train.head()
X_test = test[feature_names]

print(X_test.shape)

X_test.head()
log_y_train = train[label_name]

print(log_y_train.shape)

log_y_train.head()
from sklearn.metrics import make_scorer



def rmse(predict, actual):

    score = np.sqrt(((predict-actual)**2).mean())

    return score



rmse_score = make_scorer(rmse)

rmse_score
import xgboost as xgb



from sklearn.model_selection import cross_val_score
# # 시간이 오래걸리는 관계로 값 확인 후 주석처리 (상위 10개 값 아래 기입)

# # Coarse Search 



# num_epoch = 100



# coarse_hyperparameters_list = []



# for epoch in range(num_epoch):

#     n_estimators = np.random.randint(low=100, high=1000)



#     max_depth = np.random.randint(low=2, high=100)



#     learning_rate = 10 ** -np.random.uniform(low=0, high=10)



#     subsample = np.random.uniform(low=0.1, high=1.0)



#     colsample_bytree = np.random.uniform(low=0.4, high=1.0)



#     colsample_bylevel = np.random.uniform(low=0.4, high=1.0)



    

#     model = xgb.XGBRegressor(n_estimators=n_estimators,

#                              max_depth=max_depth,

#                              learning_rate=learning_rate,

#                              subsample=subsample,

#                              colsample_bylevel=colsample_bylevel,

#                              colsample_bytree=colsample_bytree,

#                              seed=35)





#     score = cross_val_score(model, X_train, log_y_train, cv=20, scoring=rmse_score).mean()

    

#     hyperparameters = {

#         'epoch': epoch,

#         'n_estimators': n_estimators,

#         'max_depth': max_depth,

#         'learning_rate': learning_rate,

#         'subsample': subsample,

#         'colsample_bylevel': colsample_bylevel,

#         'colsample_bytree': colsample_bytree,

#         'score': score

#     }



#     coarse_hyperparameters_list.append(hyperparameters)



#     print(f"{epoch:2} n_estimators = {n_estimators}, max_depth = {max_depth:2}, learning_rate = {learning_rate:.10f}, subsample = {subsample:.6f}, colsample_bylevel = {colsample_bylevel:.6f}, colsample_bytree = {colsample_bytree:.6f}, Score = {score:.5f}")





# coarse_hyperparameters_list = pd.DataFrame.from_dict(coarse_hyperparameters_list)



# coarse_hyperparameters_list = coarse_hyperparameters_list.sort_values(by="score")





# print(coarse_hyperparameters_list.shape)



# coarse_hyperparameters_list.head(10)
# coarse_hyperparameters_list 상위 10개 값

# 	colsample_bylevel	colsample_bytree	epoch	learning_rate	max_depth	n_estimators	score	subsample

# 26	0.905786	0.775778	26	0.012984	81	620	0.160483	0.844448

# 76	0.566049	0.687101	76	0.019889	56	575	0.160867	0.608273

# 10	0.748305	0.913150	10	0.027850	40	778	0.161094	0.648715

# 66	0.492319	0.818670	66	0.039377	44	703	0.161677	0.790363

# 89	0.895474	0.951872	89	0.007535	86	978	0.162777	0.940204

# 51	0.741066	0.971158	51	0.048315	63	505	0.163107	0.522884

# 43	0.889138	0.998115	43	0.036997	61	664	0.165543	0.181896

# 87	0.617580	0.486746	87	0.029929	85	879	0.166008	0.322514

# 69	0.605612	0.899400	69	0.056621	96	621	0.167002	0.273504

# 44	0.892544	0.456279	44	0.032803	40	255	0.167648	0.433693
# # 시간이 오래걸리는 관계로 값 확인 후 주석처리 (상위 10개 값 아래 기입)



# #Finer Search



# num_epoch = 100



# finer_hyperparameters_list = []



# for epoch in range(num_epoch):

#     n_estimators = np.random.randint(low= 500, high= 1000)

#     max_depth = np.random.randint(low= 35, high= 90)

#     learning_rate = 10 ** -np.random.uniform(low= 1, high= 3)

#     subsample = np.random.uniform(low= 0.5, high= 1.0)

#     colsample_bytree = np.random.uniform(low= 0.6, high=1.0)

#     colsample_bylevel = np.random.uniform(low=0.4, high=1.0)

#     model = xgb.XGBRegressor(n_estimators=n_estimators,

#                              max_depth=max_depth,

#                              learning_rate=learning_rate,

#                              subsample=subsample,

#                              colsample_bylevel=colsample_bylevel,

#                              colsample_bytree=colsample_bytree,

#                              seed=35)

#     score = cross_val_score(model, X_train, log_y_train, cv=20, scoring=rmse_score).mean()



#     hyperparameters = {

#         'epoch': epoch,

#         'score': score,

#         'n_estimators': n_estimators,

#         'max_depth': max_depth,

#         'learning_rate': learning_rate,

#         'subsample': subsample,

#         'colsample_bylevel': colsample_bylevel,

#         'colsample_bytree': colsample_bytree,

#     }



#     finer_hyperparameters_list.append(hyperparameters)



#     print(f"{epoch:2} n_estimators = {n_estimators}, max_depth = {max_depth:2}, learning_rate = {learning_rate:.10f}, subsample = {subsample:.6f}, colsample_bylevel = {colsample_bylevel:.6f}, colsample_bytree = {colsample_bytree:.6f}, Score = {score:.5f}")



# finer_hyperparameters_list = pd.DataFrame.from_dict(finer_hyperparameters_list)



# finer_hyperparameters_list = finer_hyperparameters_list.sort_values(by="score")



# print(finer_hyperparameters_list.shape)



# finer_hyperparameters_list.head(10)
# # finer_hyperparameters_list 상위 10개 값

# 	colsample_bylevel	colsample_bytree	epoch	learning_rate	max_depth	n_estimators	score	subsample

# 77	0.945440	0.716207	77	0.012352	57	784	0.159183	0.518462

# 75	0.568592	0.856282	75	0.014767	88	920	0.159186	0.560818

# 30	0.669199	0.867096	30	0.018533	55	661	0.159456	0.662782

# 21	0.640838	0.813904	21	0.015461	86	926	0.159467	0.537221

# 58	0.744535	0.769410	58	0.010110	43	968	0.159726	0.954902

# 73	0.679905	0.878580	73	0.015902	54	939	0.159755	0.503059

# 32	0.755865	0.925292	32	0.012935	77	757	0.160132	0.673962

# 70	0.474171	0.911513	70	0.022869	49	656	0.160185	0.713985

# 88	0.551497	0.768746	88	0.026950	74	680	0.160219	0.767849

# 22	0.893991	0.988563	22	0.012484	44	897	0.160219	0.638358
# # Coarse-Finer를 주석처리 하며 함께 주석처리 (최적값 아래 기입)



# #best_hyperparameters



# best_hyperparameters = finer_hyperparameters_list.iloc[0]



# best_n_estimators = int(best_hyperparameters["n_estimators"])



# best_max_depth = int(best_hyperparameters["max_depth"])



# best_learning_rate = best_hyperparameters["learning_rate"]



# best_subsample = best_hyperparameters["subsample"]



# best_colsample_bytree = best_hyperparameters["colsample_bytree"]



# best_colsample_bylevel = best_hyperparameters["colsample_bylevel"]



# print(f"n_estimators(best) = {best_n_estimators}, max_depth(best) = {best_max_depth}, learning_rate(best) = {best_learning_rate:.6f}, subsample(best) = {best_subsample:.6f}, colsample_bytree(best) = {best_colsample_bytree:.6f}, colsample_bylevel(best) = {best_colsample_bylevel:.6f}")
# Coarse-Finer Search로 찾은 최적값 (이 값을 사용할 경우 소수점 등의 이유로 인하여 위의 Coarse-Finer로 나온 실제 값을 실행했을 때의 score보다 살짝 안좋게 나옴)

best_n_estimators = 784

best_max_depth = 57

best_learning_rate = 0.012352 

best_subsample = 0.518462

best_colsample_bytree = 0.716207

best_colsample_bylevel = 0.945440
model = xgb.XGBRegressor(n_estimators=best_n_estimators,

                         max_depth=best_max_depth,

                         learning_rate=best_learning_rate,

                         subsample=best_subsample,

                         colsample_bytree=best_colsample_bytree,

                         colsample_bylevel=best_colsample_bylevel,

                         seed=35)



model
# from sklearn.ensemble import RandomForestRegressor



# model = RandomForestRegressor(n_estimators=best_n_estimators,

#                               max_depth=best_max_depth,

#                               random_state=35,

#                               n_jobs=-1)
# from sklearn.ensemble import RandomForestRegressor



# model = RandomForestRegressor(n_jobs=-1,

#                               random_state=35)

# model
y_train= np.exp(log_y_train) - 1    # score 확인을 위한 log해제
# score 확인하기

from sklearn.model_selection import cross_val_score



score = cross_val_score(model, X_train, y_train,

                        cv=20, scoring=rmse_score).mean()



print("Score = {0:.5f}".format(score))
model.fit(X_train,log_y_train)
log_predictions = model.predict(X_test)

print(log_predictions.shape)

log_predictions
predictions = np.exp(log_predictions) - 1

print(predictions.shape)

predictions
submission = pd.read_csv("../input/sample_submission.csv")

print(submission.shape)

submission.head()
submission["price"] = predictions

print(submission.shape)

submission.head()
submission = pd.DataFrame({"id": submission.id, "price": submission.price})

submission.to_csv("submission.csv", index=False)