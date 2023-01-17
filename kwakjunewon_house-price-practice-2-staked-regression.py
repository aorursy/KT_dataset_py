# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# stacking이란 여러 모델을 만들고 이를 계층화시켜 그 장단점을 활용하는 머신러닝기법인듯. 여기서 참조한 코드는 이 stacking을 이용해 regression한듯.

#import some necessary librairies

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
%matplotlib inline
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)


from scipy import stats
from scipy.stats import norm, skew #for some statistics


pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8")) #check the files available in the directory


# 1. intro 작업
# 여기까지 pandas로 train과 test 읽어서 각각 train, test로 만듬. 그리고 colunm을 분석해보니 81개, 그리고 test는 prrice제외한 80개 나옴.
# 그 다음 id는 분석하는데 필요없으니 따로 save해두고 train과 test에서 id를 drop함.
#Now let's import and put the train and test datasets in  pandas dataframe

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

##display the first five rows of the train dataset.
train.head(5)

#check the numbers of samples and features
print("The train data size before dropping Id feature is : {} ".format(train.shape))
print("The test data size before dropping Id feature is : {} ".format(test.shape))

#Save the 'Id' column
train_ID = train['Id']
test_ID = test['Id']

#Now drop the  'Id' colum since it's unnecessary for  the prediction process.
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)

#check again the data size after dropping the 'Id' variable
print("\nThe train data size after dropping Id feature is : {} ".format(train.shape)) 
print("The test data size after dropping Id feature is : {} ".format(test.shape))


print(train['YrSold'])

# 2. Data Processing
# (1) outliare들을 제거하는 작업이 필요. 
# GrLivArea는 Above grade (ground) living area square feet 지상의 거주가능면적을 뜻하는 듯?
# 이 변수와 feature간의 관계를 분석해보니 아름다운 비례관계를 무너뜨리는 우촉하단의 outliar 2개 발견

fig, ax = plt.subplots()
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()

# GrLivArea가 4000 이상이고 가격이 300000 밑에 이쓴ㄴ ourliar제거
#Deleting outliers
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

#Check the graphic again
fig, ax = plt.subplots()
ax.scatter(train['GrLivArea'], train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()

# 이 outliers들은 매우 안 좋은 분포를 보이기에 지웠지만 모든 outlier들을 지우는 것은 모들의 예측성을 저하시킬 수 있다. test에도 outlier는 있을테니

# (2) target variables
#  1) sale price (예상해야하는 변수니 가장 먼저 분석해야)
sns.distplot(train['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function  ~ 표준편차(mu)와 평균(sigma)를 norm이라는 라이브러리로 얻은 듯.
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution ~ 얻은 표준편차와 평균으로 시각화
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()

# 이렇게 시각화한 자료들을 살펴보면 가격이 균일하지 못하고 정규분표와 같은 그래프가 되지 못함.
# 모델들은 정규분포화된 데이터를 좋아하니 이 비뚤어짐(skewed)를 수정할 필요있음.
#  2) Log-transformation of the target variable(목표로 하는 price 변수를 로그변화)
#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
# 이 log1p를 찾아보니 1+x(대입한 값)의 자연로그를 반환하는 거라고함. 
# 음 통계적인 부분이라 정확히는 모르겠지만 심하게 skewed된 멱함수(power law function) 분포를 띠는 데이터를 정규분포(normal distribution) 로
# 변환할 때 로그 변환 (log transformation)을 사용하곤 하는데 이게 그 도구라고함.


train["SalePrice"] = np.log1p(train["SalePrice"])

#Check the new distribution 
sns.distplot(train['SalePrice'] , fit=norm);
# 수정해서 얻은 salesprice의 분포 그래프는 매우 균일한 걸 알 수 있음. 

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
# 이렇게 자연로그 형태로 수정한 분포의 표준편차, 평균을 빼고. 

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()
# 비례를 이루는지 그래프로 다시 한 번 확인
# (3) feature enginnering
ntrain = train.shape[0]  # column말고 다른 축인 data수를 제외한 값.
ntest = test.shape[0]
y_train = train.SalePrice.values # train에서 가격을 빼고.
all_data = pd.concat((train, test)).reset_index(drop=True)  # train과 test를 합쳐서 다음줄에 saleprice를 뺌.
all_data.drop(['SalePrice'], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))

#  1) Missing Data 파악
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100 # null인 data 수를 길이로 나논거네
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30] # null이 하나도 없는 값은 빼고
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})  # 최종적으로 null인 비율을 feature별로 나타낸 것. 

f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
sns.barplot(x=all_data_na.index, y=all_data_na)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)
# 그래프로도 보여줌. 
# 2) Data 상관관계 파악
#Correlation map to see how features are correlated with SalePrice
corrmat = train.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)  #feature가 너무 많아서 heatmap으로 나타내니 보기 힘듬. 
# 3) missing value 넣기
# PoolQC은 주변에 거지가 없다는 것을 의미하는 변수인데 이 feature는 거의 99퍼 null임. 사실 집 근처에 거지가 있는 경우는 거의 없으므로 
# fillna함수를 이용해 누락된 값에 모두 None을 넣음
all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
# 밑에 쥐가 있는지, 골목이 있는지, 담장이 있는지, 난로가 없는지도 모두 동일한 맥락에서 None넣음.
all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")
all_data["Alley"] = all_data["Alley"].fillna("None")
all_data["Fence"] = all_data["Fence"].fillna("None")
all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")

# 인접한 길의 직선거리는 인접 주택과 비슷하므로 이웃이 지는 값의 평균으로
#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))

# 차고 관련 feature들은 매우작으므로 문자열로 나타내어지는 것들은 None, 숫자인 것들은 0으로. 
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)

# 지하관련 feature들도 차고 관련된 것드로가 마찬가지
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')

# 벽돌관련 feature도 마찬가지로
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)

# 일반적인 구역분류(?) 라는 feature인데 이 feature은 대부분 RL이라서 첫 번째 MSZoing인 RL을 fillna함.
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])

# utilities는 두 개 말고는 allpub으로 통일되어있어서 유의미한 feature가 되지 못함. 싹 다 지우기
all_data = all_data.drop(['Utilities'], axis=1)

# TYP가 대부분. 그래서 NULL 이걸로 채우기.
all_data["Functional"] = all_data["Functional"].fillna("Typ")

# 대부분이 SBrkr이고 NA가 1개뿐이라 채우기
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])

# 이것도 하나 비어있음.
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])

# 이것도 하나 비어있음.
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])

# WD가 대부분. 그래서 NULL 이걸로 채우기.
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])

# BuildingClass가 높이를 뜻하는지 등급을 뜻하는지 명확하지 않지만 null이면 보통 none으로 봐야한다고 설명되어있음. 
all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")

#Check remaining missing values if any 
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head()
# 남은 게 없음을 알 수 있음. 
# (4) more feature enginnering
#  1) Transforming some numerical variables that are really categorical(범주화할 수 있는 변수들을 범주화하기)
#MSSubClass=The building class
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

#Changing OverallCond into a categorical variable
all_data['OverallCond'] = all_data['OverallCond'].astype(str)

#Year and month sold are transformed into categorical features.
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)       # str 타입으로 변하게하는 함수인 것 같은데 str 타입으로 바꾼다고 범주화가 가능한가? 흠

# 2) Label Encoding some categorical variables that may contain information in their ordering set
# 범주화해서 label을 붙이는 전처리과정인 것 같음.
from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data[c].values)) 
    all_data[c] = lbl.transform(list(all_data[c].values))

# shape        
print('Shape all_data: {}'.format(all_data.shape))
# 3) Adding one more important feature
# 공간 or 면적(area) 관련 feature은 매우 중요하므로 층별 area를 모두 합한 feature 만듬.
# Adding total sqfootage feature 
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
# 4) Skewed features
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
# feature들의 비뚤어진 정도를 나타내는 것 같은데 함수 찾아봐도 세세한 의미는 이해 안됨. 
# Check the skew of all numerical features
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)
# 5) Box Cox Transformation of (highly) skewed features
# 매우 왜곡된 feature들을 box cox 변형
# Box-Cox 변환은 찾아보니 정규분포가 아닌 자료를 정규분포로 변환하기 위해 사용되는 변환기법이라고 함. 
# 4)에서 살펴본 skrewed정도가 심한 feature들을 변형하기 위한 방법인 듯. 
skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    #all_data[feat] += 1
    all_data[feat] = boxcox1p(all_data[feat], lam)
    
#all_data[skewed_features] = np.log1p(all_data[skewed_features])


# 6) Getting dummy categorical features
# get_dummies를 통해 데이터프레임을 범주화할 수 있는 더미변수를 만듬.
all_data = pd.get_dummies(all_data)
print(all_data.shape)
# 새로운 train과 test 만들기. all_data가 pd.concat로 train과 test를 합쳐놓은 형태니 다시 분리하기.
train = all_data[:ntrain]
test = all_data[ntrain:]
# 3. Modeling(모델링!)
# 필요한 라이브러리 import하고
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb

# (1) Validation function. cross validation하기 위해서 data값들을 섞고 5개 fold로 validation하게 됨. 
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

# (2) 기본적인 모델들(여기서는 staked regression할테니 )
#  1) LASSO Regression. 근데 lasso는 outlier에 취약해서 RobustScaler로 전처리해줌.
# lasso자체는 오버피팅을 막기 위해 가중치의 절대값을 최소화하는 시그마를 추가적으로 더해줌. 
# https://datascienceschool.net/view-notebook/83d5e4fff7d64cb2aecfd7e42e1ece5e/
# http://rpago.tistory.com/59 ~ 이 사이트들에 릿지, 랏소, 엘라스틱 넷 회귀에 대한 내용있는데 이건 회귀계산에 대한 통계적 지식없으면...
# 찾아보니 중앙값(median)과 IQR(interquartile range)을 사용해 아웃라이어의 영향을 최소화해주는 함수라고 함. 
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))

# 2) Elastic Net Regression. 이것도 똑같이 전처리
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))

# 3) Kernel Ridge Regression
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

# 4) Gradient Boosting Regression : outlier에 강한 huber loss 사용. 
# gradient boosting 회귀 트리는 random forest같이 decision tree를 변형한 거라고 함. decision tree는 과잉적합되기 쉬운 취약점이 있으므로 그 약점을
# 보완하는 모델 중 하나인데 이 방법은 random forest와 같이 여러 개의 결정 트리를 묶어 모델을 만들지만 랜덤이 아니라 이전 트리의 오차를 보완하는 방식으로
# 순차적으로 모델을 만든다고 함. 따라서 무작위성이 없고 사전 가지치기가 사용되며 하나에서 다섯개정도의 깊지 않은 트리 사용. 
# huber loss는 outlier에 강한 손실함수라고 하는데 robust regression에 쓰인다고 함.
# 통계용어라 정확히는 인지 못해도 아웃라이어에 대비하기 위해 사용한 함수도구 정도로 이해하면 될 듯.
#ㅡ기계학습에서 부스팅(Boosting)이란 단순하고 약한 학습기(Weak Learner)를 결합해서 보다 정확하고 강력한 학습기(Strong Learner)를 만드는 방식을 의미한다.
# 정확도가 낮더라도 일단 모델을 만들고, 드러난 약점(예측 오류)은 두 번째 모델이 보완한다. 
# 이 둘을 합치면 처음보다는 정확한 모델이 만들어지고, 그럼에도 여전히 남아 있는 문제는 다음 모델에서 보완하여 계속 더하는 과정을 반복하는 원리
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)

# 5) XGBoost~ gradient boosting을 이용한 라이브러리라고 함. 
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)

# 6) LightGBM ~ xgboost와 비슷한 개념의 라이브러리인듯.
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

# 각 모델별로 score 내보기
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

# (3) Stacking models방법! 
# 1) Simplest Stacking approach : Averaging base models
# stacked model기법을 위해 각 모델별 평균구할 것. 모델별 코드의 재사용화, 캡슐화, 상속을 위해 클래스 만듬.

# Averaged base models class
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

# Averaged base models score ~ 밑에 집어넣은 네 방법의 단순평균을 구할 것!
averaged_models = AveragingModels(models = (ENet, GBoost, KRR, lasso))

score = rmsle_cv(averaged_models)
print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
# 2) Less simple Stacking : Adding a Meta-model ~ 후 솔직히 이해 안됨. 
# cross validation처럼 부분별로 나눠서 각 부분을 모델링 한다음 예측하는 것을 학습시키고 마지막에 평규내는 것 같은데..흠
# 핵심은 meta model을 하나 정하고 다른 모델들이 prediction하는걸 meta model에다 학습시키는 것인듯.
# Stacking averaged Models Class
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

    
# Stacking Averaged models Score(이제 staking한 모델로 점수내보기)
stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, KRR),
                                                 meta_model = lasso)

score = rmsle_cv(stacked_averaged_models)
print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))

# 4. Ensembling StackedRegressor, XGBoost and LightGBM해서 마지막으로 prediction하기. 
# 먼저 evaluation함수 만들기. 
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred)) # mean_squared_error은 (실제값-예측값)**2라고함. 코드상에서 쓰는 측정방법
# (1) stacked regressor로 구한 score
stacked_averaged_models.fit(train.values, y_train)
stacked_train_pred = stacked_averaged_models.predict(train.values)
stacked_pred = np.expm1(stacked_averaged_models.predict(test.values))
print(rmsle(y_train, stacked_train_pred))

lasso.fit(train, y_train)
lasso_result_train_pred = lasso.predict(train)
lasso_result_pred = np.expm1(lasso.predict(test))
print(rmsle(y_train, lasso_result_train_pred))
# (2) XGboost로
model_xgb.fit(train, y_train)
xgb_train_pred = model_xgb.predict(train)
xgb_pred = np.expm1(model_xgb.predict(test))
print(rmsle(y_train, xgb_train_pred))
# (3) LightGBM로
model_lgb.fit(train, y_train)
lgb_train_pred = model_lgb.predict(train)
lgb_pred = np.expm1(model_lgb.predict(test.values))
print(rmsle(y_train, lgb_train_pred))
# (4) 최종적으로 저 위의 3개를 ensemble한 값
'''RMSE on the entire Train data when averaging'''

print('RMSLE score on train data:')
print(rmsle(y_train,stacked_train_pred*0.70 +
               xgb_train_pred*0.15 + lgb_train_pred*0.15))

print(rmsle(y_train,stacked_train_pred*0.40 +
               xgb_train_pred*0.30+lgb_train_pred*0.30))
# ensemble해서
ensemble = stacked_pred*0.40 + xgb_pred*0.3 + lgb_pred*0.3
# submission파일 만듬. 
sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = ensemble
sub.to_csv('submission.csv',index=False)