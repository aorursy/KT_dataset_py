# Loading packages

import pandas as pd #Analysis 

import matplotlib.pyplot as plt #Visulization

import seaborn as sns #Visulization

import numpy as np #Analysis 

from scipy.stats import norm #Analysis 

from sklearn.preprocessing import StandardScaler #Analysis 

from scipy import stats #Analysis 

import warnings 

warnings.filterwarnings('ignore')

%matplotlib inline

import gc

import plotly.graph_objs as go

import plotly.offline as py

from plotly import tools
df_train = pd.read_csv('../input/train.csv')

df_test  = pd.read_csv('../input/test.csv')
print("train.csv. Shape: ",df_train.shape)

print("test.csv. Shape: ",df_test.shape)
df_train.head(10)
df_test.head(10)
f, ax = plt.subplots(figsize=(8, 6))

sns.distplot(df_train['price'])
df_train['price'] = np.log1p(df_train['price'])

f, ax = plt.subplots(figsize=(8, 6))

sns.distplot(df_train['price'])
df_train["bedrooms"].drop_duplicates()
df_train["bathrooms"].drop_duplicates()
data = pd.concat([df_train['price'], df_train['sqft_living']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.regplot(x='sqft_living', y="price", data=data)
df_train[df_train["sqft_living"]>13000]
df_train["floors"].drop_duplicates()
fig, ax = plt.subplots(figsize=(8, 6))

sns.boxplot(x="floors", y="price", data=df_train)

plt.title("Box Plot")

plt.show()
df_train["waterfront"].drop_duplicates()
fig, ax = plt.subplots(figsize=(8, 6))

sns.boxplot(x="waterfront", y="price", data=df_train)

plt.title("Box Plot")

plt.show()
df_train["view"].drop_duplicates()
fig, ax = plt.subplots(figsize=(8, 6))

sns.boxplot(x="view", y="price", data=df_train)

plt.title("Box Plot")

plt.show()
df_train["condition"].drop_duplicates()
fig, ax = plt.subplots(figsize=(8, 6))

sns.boxplot(x="condition", y="price", data=df_train)

plt.title("Box Plot")

plt.show()
df_train["grade"].drop_duplicates()
fig, ax = plt.subplots(figsize=(8, 6))

sns.boxplot(x="grade", y="price", data=df_train)

plt.title("Box Plot")

plt.show()
df_train["yr_built"].describe()
df_train[df_train["yr_renovated"]!=0]["yr_renovated"].describe()
df_train.plot(kind = "scatter", x = "long", y = "lat", alpha = 0.1, s = df_train["sqft_living"]*0.02, 

             label = "sqft_living", figsize = (10, 8), c = "price", cmap = plt.get_cmap("jet"), colorbar = True, sharex = False)
df_train[["sqft_living","sqft_lot","sqft_living15","sqft_lot15","yr_renovated","lat","long"]].head(30)
from scipy.stats import spearmanr



df_train_noid = df_train.drop("id",1)

df_train_noid = df_train_noid.drop("date",1)



plt.figure(figsize=(21,21))

sns.set(font_scale=1.25)

sns.heatmap(df_train_noid.corr(method='spearman'),fmt='.2f', annot=True, square=True , annot_kws={'size' : 15})
cor = df_train_noid.corr(method='spearman')

cor["price"].nlargest(n=20).index
df_train[df_train["sqft_living"]>13000]
data = pd.concat([df_train['price'], df_train['grade']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x='grade', y="price", data=data)
df_train[df_train["grade"]==3]
df_train.loc[(df_train['price']>14.7) & (df_train['grade'] == 8)]
df_train.loc[(df_train['price']>15.5) & (df_train['grade'] == 11)]
df_train.plot(kind = "scatter", x = "long", y = "lat", alpha = 0.1, s = df_train["sqft_living"]*0.02, 

             label = "sqft_living", figsize = (10, 8), c = "price", cmap = plt.get_cmap("jet"), colorbar = True, sharex = False)
df_train = df_train.loc[df_train['id']!=2302]
skew_columns = ['bedrooms', 'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement']



for c in skew_columns:

    df_train[c] = np.log1p(df_train[c].values)

    df_test[c] = np.log1p(df_test[c].values)
for df in [df_train,df_test]:

    df['date'] = df['date'].apply(lambda x: x[0:8])
df_train.head(4)
df_train.loc[df_train['price']==15.856731016694035] # 가장 비싼 집
for df in [df_train,df_test]:

    df['total_rooms'] = df['bedrooms'] + df['bathrooms']

    df['sqft_ratio'] = df['sqft_living'] / df['sqft_lot']

    df['sqft_ratio15'] = df['sqft_living15'] / df['sqft_lot15']

    df['date'] = df['date'].astype('int')
# 가장 비싼 집 과의 상대적인 거리 (유클리디안 거리 사용)

for df in [df_train,df_test]:

    df['re_lat'] =  df['lat'] - 47.6298

    df['re_long'] = df['long'] - (-122.323)

    df['re_location'] = (df['re_lat']**2 + df['re_long']**2)**0.5

    del df['re_lat'],df['re_long']
print(df_train.dtypes)
df_train.head(3)
#non_categorical_features = ['id', 'price','date','sqft_living', 'sqft_lot', 'floors','sqft_above', 'sqft_basement','yr_built','yr_renovated','sqft_living15','sqft_lot15','been_renovated']
#for categorical_feature in list(df_train.columns):

#    if categorical_feature not in non_categorical_features:

#        df_train[categorical_feature] = df_train[categorical_feature].astype('category')
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import Imputer

import xgboost as xgb

from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import KFold, cross_val_score

from sklearn.model_selection import cross_val_predict

from sklearn.preprocessing import RobustScaler

from sklearn.pipeline import make_pipeline

from sklearn.linear_model import RidgeCV

from sklearn.preprocessing import LabelEncoder



target_train = pd.DataFrame(df_train['price'].values)

id_test = pd.DataFrame(df_test['id'].values)



train = df_train.drop(['id','price'],1)

test = df_test.drop('id', 1)



#my_imputer = Imputer()

#train_X = my_imputer.fit_transform(X_train)

#test_X = my_imputer.transform(X_test)
train_columns = [c for c in df_train.columns if c not in ['id','price']]

feature_importance_df = pd.DataFrame()

kf = KFold(n_splits=5, random_state = 42, shuffle=True)
xgb_preds=[]

for i, (train_index, test_index) in enumerate(kf.split(train)):

    print("TRAIN:", train_index, "TEST:", test_index)

    train_X, valid_X = train.iloc[train_index], train.iloc[test_index]

    train_y, valid_y = target_train.iloc[train_index], target_train.iloc[test_index]





    xgb_pars = {'objective' : 'reg:linear', 

            'booster' : "gbtree",

            'eval_metric' : 'rmse', # rmse, 

            'nthread' : 4,

            'eta' : 0.015, # learning Rate

            'gamma' : 0,

            'max_depth' : 6, 

            'subsample' : 0.8, 

            'colsample_bytree' : 0.6, 

            'min_child_weight' : 10,

            'random_state' : 42, 

            'nrounds' : 2000,

            'n_estimators' : 3200,

            'tree_method' : 'hist'}





    d_train = xgb.DMatrix(train_X, train_y)

    d_valid = xgb.DMatrix(valid_X, valid_y)

    d_test = xgb.DMatrix(test)

    

  

    watchlist = [(d_train, 'train'), (d_valid, 'valid')]

    model = xgb.train(xgb_pars, d_train, 4000,  watchlist, verbose_eval=500, early_stopping_rounds=100)

    

    y = model.get_score(importance_type = 'weight')

    importance = pd.DataFrame.from_dict(y, orient = 'index')

    importance = importance.reset_index()

    importance = importance.rename(columns={'index': 'Feature', 0: 'Importance'})

    feature_importance_df = pd.concat([feature_importance_df, importance], axis=0)



    xgb_pred = model.predict(d_test, ntree_limit = model.best_ntree_limit)

    xgb_preds.append(list(xgb_pred))
feature_importance = feature_importance_df.groupby("Feature").mean().reset_index()



plt.figure(figsize=(14,26))

sns.barplot(x="Importance", y="Feature", data=feature_importance.sort_values(by="Importance",ascending=False))

plt.title('XGBoost Features (averaged over folds)')

plt.tight_layout()

plt.savefig('xgb_importances.png')
import lightgbm as lgb

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import KFold, cross_val_score

from sklearn.model_selection import cross_val_predict

from sklearn.preprocessing import RobustScaler

from sklearn.pipeline import make_pipeline



from sklearn.linear_model import RidgeCV



target_train = pd.DataFrame(df_train['price'].values)

id_test = pd.DataFrame(df_test['id'].values)



train = df_train.drop(['id','price'],1)

test = df_test.drop('id', 1)



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

         "random_state": 4950}

train_columns = [c for c in df_train.columns if c not in ['id','price']]

feature_importance_df = pd.DataFrame()

kf = KFold(n_splits=5, random_state = 42, shuffle=True)
folds = KFold(n_splits=5, shuffle=True, random_state=42)

oof = np.zeros(len(train))

predictions = np.zeros(len(test))

y_reg = df_train['price']

feature_importance_df = pd.DataFrame()



#run model

for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_train)):

    trn_data = lgb.Dataset(df_train.iloc[trn_idx][train_columns], label=y_reg.iloc[trn_idx])#, categorical_feature=categorical_feats)

    val_data = lgb.Dataset(df_train.iloc[val_idx][train_columns], label=y_reg.iloc[val_idx])#, categorical_feature=categorical_feats)



    num_round = 10000

    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=500, early_stopping_rounds = 100)

    oof[val_idx] = clf.predict(df_train.iloc[val_idx][train_columns], num_iteration=clf.best_iteration)

    #feature importance

    fold_importance_df = pd.DataFrame()

    fold_importance_df["Feature"] = train_columns

    fold_importance_df["importance"] = clf.feature_importance()

    fold_importance_df["fold"] = fold_ + 1

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    #predictions

    predictions += clf.predict(df_test[train_columns], num_iteration=clf.best_iteration) / folds.n_splits

    

cv = np.sqrt(mean_squared_error(oof, y_reg))

print(cv)
##plot the feature importance

cols = (feature_importance_df[["Feature", "importance"]]

        .groupby("Feature")

        .mean()

        .sort_values(by="importance", ascending=False)[:1000].index)

best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]



plt.figure(figsize=(14,26))

sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance",ascending=False))

plt.title('LightGBM Features (averaged over folds)')

plt.tight_layout()

plt.savefig('lgbm_importances.png')
submission=pd.read_csv('../input/sample_submission.csv')

submission['price']=np.expm1(predictions)

submission.to_csv('submission.csv',index=False) 
# importances = my_model.booster().get_fscores()

# importances = importances.tolist()

# importances
# len(df_train.columns)
# importance_frame = pd.DataFrame({'Importance': importances , 'Feature': df_test.columns})

# importance_frame.sort_values(by = 'Importance', inplace = True)

# importance_frame.plot(kind = 'barh', x = 'Feature', figsize = (8,8), color = 'orange')
# sns.set(font_scale = 1.5)

# fig, ax = plt.subplots(figsize=(8, 6))

# xgb.plot_importance(my_model, ax=ax)
# submission=pd.read_csv('../input/sample_submission.csv')
# submission['price']=np.exp(xgb_preds)

# submission.to_csv('submission.csv',index=False)
# submission
# import torch

# import torch.nn as nn

# import torchvision

# import torchvision.transforms as transforms



# lat_values = np.array(df_train['lat'].values)

# long_values = np.array(df_train['long'].values)

# location = torch.from_numpy(np.vstack((lat_values,long_values)).T)



# price = np.exp(torch.from_numpy(np.array(df_train['price'].values)))



# print(location)

# print(price)

# input_size = 2

# hidden_size = 3

# num_classes = 1

# learning_rate = 0.001

# class NeuralNet(nn.Module):

#     def __init__(self, input_size, hidden_size, num_classes):

#         super(NeuralNet, self).__init__()

#         self.fc1 = nn.Linear(input_size, hidden_size) 

#         self.relu = nn.Sigmoid()

#         self.fc2 = nn.Linear(hidden_size, num_classes)  

# #        self.relu = nn.ReLU()

# #        self.fc3 = nn.Linear(hidden_size, num_classes)

    

#     def forward(self, x):

#         out = self.fc1(x)

#         out = self.relu(out)

#         out = self.fc2(out)

#  #       out = self.relu(out)

#  #       out = self.fc3(out)

#         return out

    

# model = NeuralNet(input_size, hidden_size, num_classes)

# print(model)

# # 또한, nn 패키지에는 널리 사용하는 손실 함수들에 대한 정의도 포함하고 있습니다;

# # 여기에서는 평균 제곱 오차(MSE; Mean Squared Error)를 손실 함수로 사용하겠습니다.

# loss_fn = torch.nn.MSELoss(size_average=False)

# optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
# x_1 = location.float()

# y_1 = price.float()



# for t in range(256):

#     # 순전파 단계: 모델에 x를 전달하여 예상하는 y 값을 계산합니다. 모듈 객체는

#     # __call__ 연산자를 덮어써서(Override) 함수처럼 호출할 수 있게 합니다.

#     # 그렇게 함으로써 입력 데이터의 Tensor를 모듈에 전달하고 출력 데이터의

#     # Tensor를 생성합니다.

#     y_pred = model(x_1)



#     # 손실을 계산하고 출력합니다. 예측한 y값과 정답 y를 갖는 Tensor들을 전달하고,

#     # 손실 함수는 손실(loss)을 갖는 Tensor를 반환합니다.

#     loss = loss_fn(y_pred, y_1)

#     print(t, loss.item(), y_1[t])



#     # 역전파 단계를 실행하기 전에 변화도를 0으로 만듭니다.

#     optimizer.zero_grad()



#     # 역전파 단계: 모델의 학습 가능한 모든 매개변수에 대해서 손실의 변화도를

#     # 계산합니다. 내부적으로 각 모듈의 매개변수는 requires_grad=True 일 때

#     # Tensor 내에 저장되므로, 이 호출은 모든 모델의 모든 학습 가능한 매개변수의

#     # 변화도를 계산하게 됩니다.

#     loss.backward()



#     # 경사하강법(Gradient Descent)를 사용하여 가중치를 갱신합니다. 각 매개변수는

#     # Tensor이므로 이전에 했던 것과 같이 변화도에 접근할 수 있습니다.

#     optimizer.step()

    

# #    with torch.no_grad():

# #        for param in model.parameters():

# #            param -= learning_rate * param.grad
# torch.save(y_pred, 'model.ckpt')
# model1 = torch.load('model.ckpt')
# model1.detach().numpy()
# np.exp(model1.detach().numpy())