# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from scipy import stats

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")#忽略掉普通的warning

print(os.listdir("../input"))

os.environ["CUDA_VISIBLE_DEVICES"]="1,2"

# Any results you write to the current directory are saved as output.
def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))
train=pd.read_csv("../input/train.csv",index_col=0)

test=pd.read_csv(("../input/test.csv"),index_col=0)

test['SalePrice']=-99
train.head()
sns.distplot(np.log1p(train['SalePrice']),color="r")
var = 'OverallQual'

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y="SalePrice", data=train)

fig.axis(ymin=0, ymax=800000);
na_des=train.isna().sum()

# train['SalePrice']=np.log1p(train['SalePrice'])

na_des[na_des>0].sort_values(ascending=False)
new_data=pd.concat([train,test],axis=0,sort=False)

new_data.head()
# list1=['MSSubClass']

# for i in list1:

#     new_data=new_data.drop(i,axis=1)
# cols1 = ["PoolQC","MiscFeature",'SaleType', "Alley", "Fence", "FireplaceQu", "GarageQual", "GarageCond", "GarageFinish", "GarageYrBlt", "GarageType", "BsmtExposure", "BsmtCond", "BsmtQual", "BsmtFinType2", "BsmtFinType1", "MasVnrType"]

cols1=dict(new_data.dtypes[new_data.dtypes=='object'])

for col in cols1:

    new_data[col].fillna(new_data[col][new_data[col].notna()].mode()[0],inplace=True)

# cols=["MasVnrArea", "BsmtUnfSF", "TotalBsmtSF", "GarageCars", "BsmtFinSF2", "BsmtFinSF1", "GarageArea"]

cols=dict(new_data.dtypes[train.dtypes=='int64'])

for col in cols:

    new_data[col].fillna(new_data[col][new_data[col].notna()].median(),inplace=True)

new_data["LotFrontage"] = new_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
a=new_data.isna().sum()

a=a[a>0]

a=dict(a).keys()
for col in a :

    new_data[col].fillna(new_data[col][new_data[col].notna()].mode()[0],inplace=True)
print("Skewness: %f" % train['SalePrice'].skew())

print("Kurtosis: %f" % train['SalePrice'].kurt())
total= 'TotalBsmtSF'

sns.scatterplot(x=total, y='SalePrice',data=train,style='Street',markers={'Pave':'^','Grvl':'o'});
sns.scatterplot(x='GrLivArea', y='SalePrice',data=train,color='b',style='Street',markers={'Pave':'^','Grvl':'o'});
f, ax = plt.subplots(figsize=(16, 8))

fig = sns.boxplot(x='YearBuilt', y="SalePrice", data=train)

plt.xticks(rotation=90);



k = 10 

corrmat = train.corr()

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train[cols].values.T)

sns.set(font_scale=1.25)

ax = sns.heatmap(cm, annot=True,annot_kws={'size': 10}, fmt=".2f",xticklabels=cols.values,yticklabels=cols.values)

plt.show()
# sns.set()

# cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt','Street']

# sns.pairplot(train[cols], size = 2.5,hue="Street", palette="husl")

# plt.show();
sns.violinplot(x="SaleType", y="SalePrice", data=train,hue="Street",palette="Set2")
f, ax = plt.subplots(figsize=(16, 8))

fig = sns.boxplot(x='YearBuilt', y="GrLivArea", data=train)

plt.xticks(rotation=90);
train.groupby(['YearBuilt']).SalePrice.aggregate(['mean','std','max']).plot()
train.groupby(['YearBuilt']).GrLivArea.aggregate(['mean','std','max']).plot()
train.sort_values(by = 'GrLivArea', ascending = False)['GrLivArea'][:1]

train = train.drop(train[train.index == 1299].index)
fig = plt.figure()

res = stats.probplot(train['SalePrice'], plot=plt)

sns.distplot(np.log1p(train['GrLivArea']));

fig = plt.figure()

res = stats.probplot(np.log1p(train['GrLivArea']), plot=plt)

new_data['HasBsmt']=new_data['TotalBsmtSF'].apply(lambda x:1 if x!=0 else 0)
# new_data.loc[new_data['HasBsmt']==1,'TotalBsmtSF'] = np.log1p(new_data['TotalBsmtSF'])
fig = plt.figure()

res = stats.probplot(new_data[new_data['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)
# new_data['MSSubClass'] =new_data['MSSubClass'].apply(str)

# new_data['YrSold'] = new_data['YrSold'].astype(str)

# new_data['MoSold'] = new_data['MoSold'].astype(str)



new_data['GrLivArea'][new_data['GrLivArea']==0]=1

new_data['1stFlrSF'][new_data['1stFlrSF']==0]=1

new_data['2ndFlrSF'][new_data['2ndFlrSF']==0]=1



# new_data['GrLivArea']=np.log1p(new_data['GrLivArea'])

# new_data['1st_GrLivArea']=new_data['1stFlrSF']/new_data['GrLivArea']

# new_data['2st_GrLivArea']=new_data['2ndFlrSF']/new_data['GrLivArea']

# new_data['1st_2st']=new_data['1stFlrSF']/new_data['2ndFlrSF']

new_data['TotalSF'] = new_data['TotalBsmtSF'] + new_data['1stFlrSF'] + new_data['2ndFlrSF']

# list_del=['1stFlrSF','BsmtUnfSF','BsmtFinSF2','BsmtFinType1','2ndFlrSF','Fireplaces','GarageArea','GarageCond','HalfBath','MSSubClass','BsmtCond','Utilities', 'Street', 'PoolQC']

# for i in list_del:

#     new_data=new_data.drop(i,axis=1)
from scipy.stats import skew

numeric_feats = new_data.dtypes[new_data.dtypes != "object"].index

# Check the skew of all numerical features

skewed_feats = new_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

print("\nSkew in numerical features: \n")

skewness = pd.DataFrame({'Skew' :skewed_feats})



skewness.head(10)
skewness = skewness[abs(skewness) > 0.75]

skewed_features = skewness.index

lam = 0.15

for feat in skewed_features:   

    new_data[feat] = np.log1p(new_data[feat])
new_data=pd.get_dummies(new_data)
train=new_data.loc[np.array(train.index)]

test=new_data.loc[np.array(test.index)]
x=train.drop('SalePrice',axis=1)

y=train['SalePrice']

x_test=test.drop('SalePrice',axis=1)
x=np.array(x)

y=np.array(y)

x_test=np.array(x_test)

x=x.reshape(x.shape[0],x.shape[1],1)

x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],1)
from keras import backend as K

from keras import layers

from keras.callbacks import EarlyStopping



from keras.layers  import MaxPool1D,GRU,RNN

from keras import Sequential

from keras.layers import LSTM,Conv1D,Dense,Flatten,Conv2D,LeakyReLU, Activation,Bidirectional,BatchNormalization,Dropout,GlobalAveragePooling1D,Embedding

from keras.activations import relu

from keras import initializers

from keras.optimizers import Adam,RMSprop

from keras import backend as K

from keras.callbacks import ReduceLROnPlateau

from keras.models import Model



reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')



def RMSE(y_true, y_pred):

        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

ear=EarlyStopping(monitor='val_loss',min_delta=0,patience=5,verbose=0, mode='auto',restore_best_weights=True)
train1=x
model=Sequential()



model.add(Dense(units=train1.shape[1]))

model.add(Dense(units=int(train1.shape[1]/2)))

model.add(Dense(units=int(train1.shape[1]/2)))

model.add(Dense(units=train1.shape[1]))

# model.add(Conv1D(train1.shape[1],2))

# model.add(Conv1D(train1.shape[1],2))

# model.add(MaxPool1D(2,1))

# model.add(Dropout(0.15))



model.add(LSTM(32,return_sequences=True,input_shape=(None,1)))

model.add(Activation(K.relu))

model.add(Dense(units=train1.shape[1]))

model.add(Dense(units=int(train1.shape[1]/2)))

model.add(Dense(units=int(train1.shape[1]/2)))

model.add(Dense(units=train1.shape[1]))

model.add(Conv1D(train1.shape[1],2))

model.add(Conv1D(train1.shape[1],2))

model.add(MaxPool1D(2,1))

model.add(Dropout(0.15))

model.add(LSTM(32,return_sequences=True))



model.add(LSTM(32,return_sequences=True,input_shape=(None,1)))

model.add(Activation(K.relu))

model.add(Dense(units=train1.shape[1]))

model.add(Dense(units=int(train1.shape[1]/2)))

model.add(Dense(units=int(train1.shape[1]/2)))

model.add(Dense(units=train1.shape[1]))

model.add(Conv1D(train1.shape[1],2))

model.add(Conv1D(train1.shape[1],2))

model.add(MaxPool1D(2,1))

model.add(Dropout(0.15))



model.add(GRU(32,return_sequences=True))

model.add(Flatten())



model.add(Dropout(0.25))

model.add(Dense(units=100))

model.add(Dropout(0.25))

model.add(Dense(units=10))

model.add(Dense(units=5))

model.add(Dense(units=1,activation='selu'))



model.compile(optimizer=Adam(), loss=RMSE)
hist=model.fit(train1,y,epochs=100, batch_size=80,validation_split=0.1,verbose=1,callbacks=[reduce_lr,ear])

model.summary()
y
# from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC

# from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

# from catboost import CatBoostRegressor

# from sklearn.kernel_ridge import KernelRidge

# from sklearn.pipeline import make_pipeline

# from sklearn.preprocessing import RobustScaler

# from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

# from sklearn.model_selection import KFold, cross_val_score, train_test_split

# from sklearn.metrics import mean_squared_error

# import xgboost as xgb

# import lightgbm as lgb

# import time

# #Validation function

# n_folds = 5



# def rmsle_cv(model):

#     kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)

#     rmse = np.sqrt(-cross_val_score(model,x,y, scoring="neg_mean_squared_error", cv=kf))

#     return(rmse)



# def eval_model(model, name):

#     start_time = time.time()

#     score = rmsle_cv(model)

#     print("{} score: {:.4f} ({:.4f}),     execution time: {:.1f}".format(name, score.mean(), score.std(), time.time()-start_time))
# from sklearn.kernel_ridge import KernelRidge
# # 

# mod_lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))

# eval_model(mod_lasso, "lasso")

# mod_enet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))

# eval_model(mod_enet, "enet")

# KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

# eval_model(KRR, "KRR")

# # mod_cat = CatBoostRegressor(iterations=10000, learning_rate=0.01,

# #                             depth=5, eval_metric='RMSE',

# #                             colsample_bylevel=0.7, random_seed = 17, silent=True,

# #                             bagging_temperature = 0.2, early_stopping_rounds=200)

# # eval_model(mod_cat, "cat")

# mod_gboost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.06,

#                                    max_depth=3, max_features='sqrt',

#                                    min_samples_leaf=7, min_samples_split=10, 

#                                    loss='huber', random_state=5)

# eval_model(mod_gboost, "gboost")

# mod_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

#                              learning_rate=0.05, max_depth=3, 

#                              min_child_weight=1.7817, n_estimators=2200,

#                              reg_alpha=0.4640, reg_lambda=0.8571,

#                              subsample=0.5213, silent=1,

#                              random_state=7, nthread=-1)

# eval_model(mod_xgb, "xgb")

# mod_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=6,

#                               learning_rate=0.05, n_estimators=650,

#                               max_bin=58, bagging_fraction=0.8,

#                               bagging_freq=5, feature_fraction=0.2319,

#                               feature_fraction_seed=9, bagging_seed=9,

#                               min_data_in_leaf=7, min_sum_hessian_in_leaf=11)

# eval_model(mod_lgb, "lgb")
# def valid(model):

#     x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3, random_state=0)

#     model.fit(x_train.values,y_train.values)

#     train_pred = model.predict(test.values)

#     print(rmsle(y, train_pred))

#     return (pred)
# class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):

#     def __init__(self, base_models, meta_model, n_folds=5):

#         self.base_models = base_models

#         self.meta_model = meta_model

#         self.n_folds = n_folds

   

#     # We again fit the data on clones of the original models

#     def fit(self, X, y):

#         self.base_models_ = [list() for x in self.base_models]

#         self.meta_model_ = clone(self.meta_model)

#         kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        

#         # Train cloned base models then create out-of-fold predictions

#         # that are needed to train the cloned meta-model

#         out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))

#         for i, model in enumerate(self.base_models):

#             for train_index, holdout_index in kfold.split(X, y):

#                 instance = clone(model)

#                 self.base_models_[i].append(instance)

#                 instance.fit(X[train_index], y[train_index])

#                 y_pred = instance.predict(X[holdout_index])

#                 out_of_fold_predictions[holdout_index, i] = y_pred

                

#         # Now train the cloned  meta-model using the out-of-fold predictions as new feature

#         self.meta_model_.fit(out_of_fold_predictions, y)

#         return self

   

#     #Do the predictions of all base models on the test data and use the averaged predictions as 

#     #meta-features for the final prediction which is done by the meta-model

#     def predict(self, X):

#         meta_features = np.column_stack([

#             np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)

#             for base_models in self.base_models_ ])

#         return self.meta_model_.predict(meta_features)

    

# mod_stacked = StackingAveragedModels(base_models = (mod_enet,KRR,mod_gboost), meta_model =mod_lasso )

# eval_model(mod_stacked, "stacked")
# def rmsle(y, y_pred):

#     return np.sqrt(mean_squared_error(y, y_pred))



# def predict(model):

#     model.fit(x,y)

#     train_pred = model.predict(x)

#     pred = np.expm1(model.predict(x_test))

#     print(rmsle(y, train_pred))

#     return (pred)
# pre_lasso = predict(mod_lasso)

# pre_enet = predict(mod_enet)

# pre_krr = predict(KRR)

# pre_xgb = predict(mod_xgb)

# pre_gboost = predict(mod_gboost)

# pre_lgb = predict(mod_lgb)

# pre_stack1 = predict(mod_stacked)
# test['id']=test.index

# test['SalePrice']=0.7*pre_stack1+0.15*pre_lgb+0.15*pre_xgb

# test[['id','SalePrice']].to_csv('submission_Dragon3.csv', index=False)

# test[['id','SalePrice']].head()
test['id']=test.index

test['SalePrice']=np.expm1(model.predict(x_test))

test[['id','SalePrice']].to_csv('submission_Dragon4.csv', index=False)