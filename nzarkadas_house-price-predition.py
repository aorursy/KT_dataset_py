import numpy as np 

import pandas as pd 

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





import os

DATA_DIR='../input'

print(os.listdir(DATA_DIR))
train_df = pd.read_csv(DATA_DIR+'/train.csv')

test_df = pd.read_csv(DATA_DIR+'/test.csv')
train_df.head(5)
test_df.head(5)
train_df.shape
test_df.shape
train_ID = train_df['Id']

test_ID = test_df['Id']
#Now drop the  'Id' colum since it's unnecessary for  the prediction process.

train_df.drop("Id", axis = 1, inplace = True)

test_df.drop("Id", axis = 1, inplace = True)



#check again the data size after dropping the 'Id' variable

print("\nThe train data size after dropping Id feature is : {} ".format(train_df.shape)) 

print("The test data size after dropping Id feature is : {} ".format(test_df.shape))
fig, ax = plt.subplots()

ax.scatter(x = train_df['GrLivArea'], y = train_df['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

plt.show()
#Deleting outliers

train_df = train_df.drop(train_df[(train_df['GrLivArea']>4000) & (train_df['SalePrice']<300000)].index)



#Check the graphic again

fig, ax = plt.subplots()

ax.scatter(train_df['GrLivArea'], train_df['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

plt.show()
sns.distplot(train_df['SalePrice'] , fit=norm);



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(train_df['SalePrice'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')



#Get also the QQ-plot

fig = plt.figure()

res = stats.probplot(train_df['SalePrice'], plot=plt)

plt.show()
#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column

train_df["SalePrice"] = np.log1p(train_df["SalePrice"])



#Check the new distribution 

sns.distplot(train_df['SalePrice'] , fit=norm);



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(train_df['SalePrice'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')



#Get also the QQ-plot

fig = plt.figure()

res = stats.probplot(train_df['SalePrice'], plot=plt)

plt.show()

ntrain = train_df.shape[0]

ntest = test_df.shape[0]

y_train = train_df.SalePrice.values

all_data = pd.concat((train_df, test_df)).reset_index(drop=True)

all_data.drop(['SalePrice'], axis=1, inplace=True)

print("all_data size is : {}".format(all_data.shape))
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100

all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})

missing_data.head(20)
f, ax = plt.subplots(figsize=(15, 12))

plt.xticks(rotation='90')

sns.barplot(x=all_data_na.index, y=all_data_na)

plt.xlabel('Features', fontsize=15)

plt.ylabel('Percent of missing values', fontsize=15)

plt.title('Percent missing data by feature', fontsize=15)
#Correlation map to see how features are correlated with SalePrice

corrmat = train_df.corr()

plt.subplots(figsize=(12,9))

sns.heatmap(corrmat, vmax=0.9, square=True)
all_data['PoolQC'] = all_data['PoolQC'].fillna('None')
all_data['MiscFeature'] = all_data['MiscFeature'].fillna('None')
all_data['Alley'] = all_data['Alley'].fillna('None')
all_data['Fence'] = all_data['Fence'].fillna('None')
all_data['FireplaceQu'] = all_data['FireplaceQu'].fillna('None')
#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood

all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(

    lambda x: x.fillna(x.median()))
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):

    all_data[col] = all_data[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    all_data[col] = all_data[col].fillna('None')
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")

all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
all_data = all_data.drop(['Utilities'], axis=1)
all_data['Functional'] = all_data['Functional'].fillna('Typ')
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])

all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100

all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)

missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})

missing_data.head()
#MSSubClass=The building class

all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)





#Changing OverallCond into a categorical variable

all_data['OverallCond'] = all_data['OverallCond'].astype(str)





#Year and month sold are transformed into categorical features.

all_data['YrSold'] = all_data['YrSold'].astype(str)

all_data['MoSold'] = all_data['MoSold'].astype(str)
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
# Adding total sqfootage feature 

all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index



# Check the skew of all numerical features

skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

print("\nSkew in numerical features: \n")

skewness = pd.DataFrame({'Skew' :skewed_feats})

skewness.head(10)
skewness = skewness[abs(skewness) > 0.75]

print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))



from scipy.special import boxcox1p

skewed_features = skewness.index

lam = 0.15

for feat in skewed_features:

    #all_data[feat] += 1

    all_data[feat] = boxcox1p(all_data[feat], lam)
all_data = pd.get_dummies(all_data)

print(all_data.shape)
train = all_data[:ntrain]

test = all_data[ntrain:]
from sklearn.linear_model import Ridge, RidgeCV, ElasticNetCV, LassoCV, LassoLarsCV

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler
#Validation function

n_folds = 5



def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)

    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)
model_ridge = Ridge()
alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]

cv_ridge = [rmsle_cv(make_pipeline(RobustScaler(), Ridge(alpha = alpha, random_state=1))).mean() 

            for alpha in alphas]
cv_ridge = pd.Series(cv_ridge, index = alphas)

cv_ridge.plot(title = "Validation")

plt.xlabel("alpha")

plt.ylabel("rmsle")
cv_ridge.min()
model_ridge = make_pipeline(RobustScaler(), Ridge(alpha = 10, random_state=1))

ridge_res = rmsle_cv(model_ridge)

print('Ridge evaluation result : {:<8.3f}'.format(ridge_res.min()))
pd_scores = pd.DataFrame(data={'Model':['Ridge'], 'Mean':[ridge_res.mean()], 'Std':[ridge_res.std()],'Min':[ridge_res.min()]})
pd_scores
model_lassocv = make_pipeline(RobustScaler(), LassoCV(alphas = [1, 0.1, 0.001, 0.0005], random_state=1))

lassocv_res = rmsle_cv(model_lassocv)

print('LassoCV Score : {:<8.4f}, with min value : {:<8.4f} and std : {:<8.4f}'.format(lassocv_res.mean(), lassocv_res.min(),lassocv_res.std()))
coef = pd.Series(model_lassocv.steps[1][1].fit(train,y_train).coef_, index = train.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
imp_coef = pd.concat([coef.sort_values().head(12),

                     coef.sort_values().tail(12)])
plt.rcParams['figure.figsize'] = (8.0, 10.0)

imp_coef.plot(kind = "barh")

plt.title("Coefficients in the Lasso Model")
#let's look at the residuals as well:

plt.rcParams['figure.figsize'] = (6.0, 6.0)



preds = pd.DataFrame({"preds":model_lassocv.steps[1][1].predict(train), "true":y_train})

preds["residuals"] = preds["true"] - preds["preds"]

preds.plot(x = "preds", y = "residuals",kind = "scatter")
lasso_score=[{'Model':'LassoCV', 'Mean':lassocv_res.mean(),'Std':lassocv_res.std(), 'Min':lassocv_res.min()}]

pd_scores = pd_scores.append(lasso_score,ignore_index=True, sort=False)
pd_scores
model_enet_cv = make_pipeline(RobustScaler(), ElasticNetCV(cv=None, random_state=0))
enet_score = rmsle_cv(model_enet_cv)

print('ElasticNet Score : {:<8.4f}, with min value : {:<8.4f} and std : {:<8.4f}'.format(enet_score.mean(), enet_score.min(),enet_score.std()))
score = [{'Model':'ElasticNetCV', 'Mean':enet_score.mean(),'Std':enet_score.std(), 'Min':enet_score.min()}]

pd_scores = pd_scores.append(score,ignore_index=True, sort=False)
from sklearn.kernel_ridge import KernelRidge

KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

KRR_Score = rmsle_cv(KRR)

print('KernelRidge Score : {:<8.4f}, with min value : {:<8.4f} and std : {:<8.4f}'.format(KRR_Score.mean(), KRR_Score.min(),KRR_Score.std()))
score = [{'Model':'KernelRidge', 'Mean':KRR_Score.mean(),'Std':KRR_Score.std(), 'Min':KRR_Score.min()}]

pd_scores = pd_scores.append(score,ignore_index=True, sort=False)
import xgboost as xgb
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state =7, nthread = -1)

score1 = rmsle_cv(model_xgb)

print('xgboost Score : {:<8.4f}, with min value : {:<8.4f} and std : {:<8.4f}'.format(score1.mean(), score1.min(),score1.std()))
# dtrain = xgb.DMatrix(train, label = y_train)

# dtest = xgb.DMatrix(test)



# params = {"max_depth":2, "eta":0.1}

# model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)



# model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()



# model_xgb1 = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1) #the params were tuned using xgb.cv

# score1 = rmsle_cv(model_xgb1)

# print('xgboost Score : {:<8.4f}, with min value : {:<8.4f} and std : {:<8.4f}'.format(score1.mean(), score1.min(),score1.std()))

#model_xgb.fit(train, y_train)



# Adding xgboost score to scores board

score = [{'Model':'XGBoost', 'Mean':score1.mean(),'Std':score1.std(), 'Min':score1.min()}]

pd_scores = pd_scores.append(score,ignore_index=True, sort=False)
pd_scores.sort_values(by=['Mean','Std', 'Min'])
lassocv = model_lassocv.steps[1][1]

lassocv.fit(train,y_train)

preds = np.expm1(lassocv.predict(test))

preds
solution = pd.DataFrame({"id":test_ID, "SalePrice":preds})

solution.head(5)
solution.to_csv("lasso_sol.csv", index = False)