import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

infile = "../input/"



from scipy import stats

from scipy.stats import norm, skew
train = pd.read_csv(infile+"train.csv")

test = pd.read_csv(infile+"test.csv")

submit = pd.read_csv(infile+"sample_submission.csv")
train.head(5)
test.head(5)
print("train 셋의 (행,열) : {}".format(train.shape))

print("test 셋의 (행,열) : {}".format(test.shape))



train = train.drop(['id'],axis=1)

test = test.drop(['id'],axis=1)



print("수정된 train 셋의 (행,열) : {}".format(train.shape))

print("수정된 test 셋의 (행,열) : {}".format(test.shape))
train.corr(method='pearson')
plt.figure(figsize = (15,10))

sns.heatmap(train.corr(), annot=True, cmap='Blues',fmt='.2f',linewidths=.5)
#corr 값이 높은 10개의 데이터를 heatmap 하는 과정



#saleprice correlation matrix

corrmat = train.corr()

k = 10 #number of variables for heatmap

cols = corrmat.nlargest(k, 'price')['price'].index

cm = np.corrcoef(train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',cmap='Blues', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
train.info()
sns.distplot(train['price'],fit=norm)
(mu, sigma) = norm.fit(train['price'])

print("\n mu = {:.2f} and sigma = {:.2f}".format(mu,sigma))
fig = plt.figure()

res = stats.probplot(train['price'], plot= plt)

plt.show()
train['price'] = np.log1p(train['price'])



sns.distplot(train['price'],fit=norm)



(mu, sigma) = norm.fit(train['price'])

print("\n mu = {:.2f} and sigma = {:.2f}".format(mu,sigma))



fig = plt.figure()

res = stats.probplot(train['price'],plot=plt)

plt.show()
ntrain = train.shape[0]

ntest = train.shape[0]

y_train = train['price']
train = train.drop(['price'],axis=1)
df = pd.concat([train, test],axis=0)
df.info()
df.date.head()
year = df.date.apply(lambda x:x[0:4]).astype(int)

month = df.date.apply(lambda x:x[4:6]).astype(int)

day = df.date.apply(lambda x:x[6:8]).astype(int)
df['year'] = year

df['month'] = month

df['day'] = day
df = df.drop(['date'],axis=1)
df.describe()
print("zipcode의 인덱스 개수 : {}".format(len(df.zipcode.value_counts().index))) #zipcode 인덱스의 개수

print("zipcode의 인덱스 중 최솟값 : {}".format(df.zipcode.value_counts().min())) #zipcode 인덱스 중 최소값

print("zipcode의 인덱스 중 최댓값 : {}".format(df.zipcode.value_counts().max())) #zipcode 인덱스 중 최대값
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

le.fit(df.zipcode)



df['zipcode'] = le.transform(df.zipcode)

df.zipcode.value_counts()
plt.figure(figsize = (10,10))

sns.boxplot(x=df.zipcode, y=df.sqft_living)
df.columns
sns.countplot(df.bedrooms)
sns.countplot(df.bathrooms)
df['room_sum'] = df.bedrooms + df.bathrooms
sns.countplot(df.room_sum)
print("재건축한 건물 개수 :",df[df.sqft_living == df.sqft_living15].shape[0])

print("재건축하지 않은 건물 개수 :",df[df.sqft_living != df.sqft_living15].shape[0])
df.room_sum = df.room_sum+1 #0값이 있기 때문에 나눠줄때 오류가 발생하는 것을 막아주기 위해 1을 더한다.
sns.countplot(df.room_sum)
df = df.reset_index(drop=True) #concat으로 인해 인덱스 오류가 발생함
row = df.shape[0]

sqft_per_rooms = []

for i in range(row):

    if df.sqft_living[i] == df.sqft_living15[i]:

        sqft_per_rooms.append(df.sqft_living[i]/df.room_sum[i])

    else:

        sqft_per_rooms.append(df.sqft_living15[i]/df.room_sum[i])

df['sqft_per_rooms'] = sqft_per_rooms
sns.distplot(df.sqft_per_rooms,fit=norm)
print("재건축 하지 않은 집의 갯수 :",df[df.yr_renovated==0].shape[0])

print("재건축 된 집의 갯수 :",df[df.yr_renovated>0].shape[0])
during_yr = []

for i in range(row):

    if df.yr_renovated[i]==0:

        during_yr.append(df.year[i]-df.yr_built[i])

    else:

        during_yr.append(df.year[i]-df.yr_renovated[i])



df['during_yr'] = during_yr
sns.distplot(df.during_yr,fit=norm)
df.floors.value_counts()
sqft_per_floor = []

for i in range(row):

    if df.sqft_living[i]==df.sqft_living15[i]:

        sqft_per_floor.append(df.sqft_living[i]/df.floors[i])

    else:

        sqft_per_floor.append(df.sqft_living15[i]/df.floors[i])



df['sqft_per_floor'] = sqft_per_floor
sns.distplot(df.sqft_per_floor,fit=norm)
df['sqft_total'] = df['sqft_above'] + df['sqft_basement']
df.columns
use_col = ['sqft_living', 'sqft_lot','sqft_above','sqft_basement','sqft_living15','sqft_lot15','sqft_total','sqft_per_rooms','sqft_per_floor']
skewed_feats = df[use_col].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

print("\nSkew in numerical features: \n")

skewness = pd.DataFrame({"Skew":skewed_feats})

skewness



#skewness에 대한 간단한 설명

#데이터의 분포가 한쪽으로 치우친 정도를 의미한다.

#왼쪽으로 치우쳐져 있을 때는 skewness가 음수, 오른쪽으로 치우쳐져 있을 때는 skewness 양수.
skewness = skewness[abs(skewness) > 0.75]

#절댓값이 0.75보다 높은 skewness를 가지는 행을 Box Cox transform 해준다

print("행 개수 :",skewness.shape[0])
#Box Cox transform을 해줌으로써 한쪽으로 길어진 꼬리의 모양을 잡아줄 수 있다.

#편향을 잡아주는데 도움이된다.

#np.log1p를 사용해도 비슷한 효과를 얻을 수 있다.



from scipy.special import boxcox1p

skewed_features = skewness.index

lam = 0.15

for feat in skewed_features:

    df[feat] = boxcox1p(df[feat],lam)
fig, ax = plt.subplots(3,4,figsize=(20,20))

n=0

for r in range(3):

    for c in range(4):

        sns.distplot(df[use_col[n]],fit=norm,ax=ax[r][c])

        ax[r][c].set_title(use_col[n],fontsize=20)

        n+=1

        if n==len(use_col):

            break
train = df.iloc[:ntrain,:]

test = df.iloc[ntrain:,:]
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

import xgboost as xgb

import lightgbm as lgb
#Validation function

n_folds = 5



def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)

    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)
lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))

ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9,random_state=3))

KRR = KernelRidge(alpha=0.6,kernel='polynomial', degree=2, coef0=2.5)

GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5)
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state =7, nthread = -1)



model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.05, n_estimators=720,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
score = rmsle_cv(lasso)

print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(ENet)

print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(KRR)

print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(GBoost)

print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(model_xgb)

print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(model_lgb)

print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, models):

        self.models = models

        

    # we define clones of the original models to fit the data in

    def fit(self, X, y):

        self.models_ = [clone(x) for x in self.models]

        

        # Train cloned base models

        for model in self.models_:

            model.fit(X, y)



        return self

    

    #Now we do the predictions for cloned models and average them

    def predict(self, X):

        predictions = np.column_stack([

            model.predict(X) for model in self.models_

        ])

        return np.mean(predictions, axis=1)   
averaged_models = AveragingModels(models = (ENet, GBoost, KRR, lasso))



score = rmsle_cv(averaged_models)

print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, base_models, meta_model, n_folds=5):

        self.base_models = base_models

        self.meta_model = meta_model

        self.n_folds = n_folds

   

    # We again fit the data on clones of the original models

    def fit(self, X, y):

        self.base_models_ = [list() for x in self.base_models]

        self.meta_model_ = clone(self.meta_model)

        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        

        # Train cloned base models then create out-of-fold predictions

        # that are needed to train the cloned meta-model

        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))

        for i, model in enumerate(self.base_models):

            for train_index, holdout_index in kfold.split(X, y):

                instance = clone(model)

                self.base_models_[i].append(instance)

                instance.fit(X[train_index], y[train_index])

                y_pred = instance.predict(X[holdout_index])

                out_of_fold_predictions[holdout_index, i] = y_pred

                

        # Now train the cloned  meta-model using the out-of-fold predictions as new feature

        self.meta_model_.fit(out_of_fold_predictions, y)

        return self

   

    #Do the predictions of all base models on the test data and use the averaged predictions as 

    #meta-features for the final prediction which is done by the meta-model

    def predict(self, X):

        meta_features = np.column_stack([

            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)

            for base_models in self.base_models_ ])

        return self.meta_model_.predict(meta_features)
stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, KRR),

                                                 meta_model = lasso)



#score = rmsle_cv(stacked_averaged_models)

#print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))
def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))
stacked_averaged_models.fit(train.values, y_train)

stacked_train_pred = stacked_averaged_models.predict(train.values)

stacked_pred = np.expm1(stacked_averaged_models.predict(test.values))

print(rmsle(y_train, stacked_train_pred))
model_xgb.fit(train, y_train)

xgb_train_pred = model_xgb.predict(train)

xgb_pred = np.expm1(model_xgb.predict(test))

print(rmsle(y_train, xgb_train_pred))
model_lgb.fit(train, y_train)

lgb_train_pred = model_lgb.predict(train)

lgb_pred = np.expm1(model_lgb.predict(test.values))

print(rmsle(y_train, lgb_train_pred))
'''RMSE on the entire Train data when averaging'''



print('RMSLE score on train data:')

print(rmsle(y_train,stacked_train_pred*0.70 +

               xgb_train_pred*0.15 + lgb_train_pred*0.15 ))
ensemble = stacked_pred*0.70 + xgb_pred*0.15 + lgb_pred*0.15
submit['price'] = ensemble

submit.to_csv("stacked_suwon_study.csv",index=False)