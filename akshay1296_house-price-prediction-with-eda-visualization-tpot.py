import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from scipy import stats

from statsmodels.graphics.gofplots import qqplot

from sklearn.preprocessing import LabelEncoder

from scipy.stats import norm, skew, boxcox_normmax
hp_train = pd.read_csv("../input/train.csv")

hp_test = pd.read_csv("../input/test.csv")
hp_train.head()
print("Train set size:", hp_train.shape)

print("Test set size:", hp_test.shape)
train_ID = hp_train['Id']

test_ID = hp_test['Id']



# Now drop the  'Id' colum since it's unnecessary for  the prediction process.

hp_train.drop(['Id'], axis=1, inplace=True)

hp_test.drop(['Id'], axis=1, inplace=True)
col_name = ['GrLivArea','TotalBsmtSF','1stFlrSF','BsmtFinSF1','LotArea']

outlier = [4500, 3000, 2500, 2000, 55000]

for i, c in zip(range(5), col_name):

    fig = plt.figure(figsize=(15,5))

    plt.subplot(1,2,1)

    plt.scatter(np.abs(hp_train[hp_train[c] < outlier[i]][c]), np.array(hp_train[hp_train[c] < outlier[i]]['SalePrice']), c='b')

    plt.scatter(np.abs(hp_train[hp_train[c] >= outlier[i]][c]), np.array(hp_train[hp_train[c] >= outlier[i]]['SalePrice']), c='r')

    plt.title('Before removing outliers for '+c)

    plt.xlabel(c)

    plt.ylabel('SalePrice')

    

    

    plt.subplot(1,2,2)

    plt.scatter(np.abs(hp_train[hp_train[c] < outlier[i]][c]), np.array(hp_train[hp_train[c] < outlier[i]]['SalePrice']), c='b')

    plt.title('After removing outliers for '+c)

    plt.xlabel(c)

    plt.ylabel('SalePrice')

    plt.show()

# removing outliers

print(hp_train.shape)

hp_train = hp_train[hp_train['GrLivArea'] < 4500]

hp_train = hp_train[hp_train['LotArea'] < 550000]

hp_train = hp_train[hp_train['TotalBsmtSF'] < 3000]

hp_train = hp_train[hp_train['1stFlrSF'] < 2500]

hp_train = hp_train[hp_train['BsmtFinSF1'] < 2000]
#Describing SalePrice

hp_train.SalePrice.describe()
#Understanding the distribution of SalePrice

fig = plt.figure(figsize=(15,5))

plt.subplot(1,2,1)

sns.distplot(hp_train['SalePrice'])

plt.title('Understanding the distribution of SalePrice')



plt.subplot(1,2,2)

stats.probplot((hp_train['SalePrice']), plot=plt)

plt.show()
print("Skewness: %f" % hp_train['SalePrice'].skew())

print("Kurtosis: %f" % hp_train['SalePrice'].kurt())
fig = plt.figure(figsize=(15,5))

plt.subplot(1,2,1)

sns.distplot(np.sqrt(hp_train['SalePrice']))

plt.title('Distribution of SalePrice after square root transformation')



plt.subplot(1,2,2)

stats.probplot(np.sqrt(hp_train['SalePrice']), plot=plt)

plt.title('Square root transformation')

plt.show()



fig = plt.figure(figsize=(15,5))

plt.subplot(1,2,1)

sns.distplot(np.cbrt(hp_train['SalePrice']))

plt.title('Distribution of SalePrice after cube root transformation')



plt.subplot(1,2,2)

stats.probplot(np.cbrt(hp_train['SalePrice']), plot=plt)

plt.title('Cube root transformation')

plt.show()



fig = plt.figure(figsize=(15,5))

plt.subplot(1,2,1)

sns.distplot(np.log1p(hp_train['SalePrice']))

plt.title('Distribution of SalePrice after log transformation')



plt.subplot(1,2,2)

stats.probplot(np.log1p(hp_train['SalePrice']), plot=plt)

plt.title('Log transformation')

plt.show()

print("After log transformation")

hp_train['SalePrice'] = np.log1p(hp_train['SalePrice']) 

print("Skewness: %f" % (hp_train['SalePrice'].skew()))

print("Kurtosis: %f" % (hp_train['SalePrice'].kurt()))
hp_train.SalePrice.describe()
#Correlation map to see how features are correlated with SalePrice

corrmat = hp_train.corr()



# select top 10 highly correlated variables with SalePrice

num = 10

col = corrmat.nlargest(num, 'SalePrice')['SalePrice'].index

coeff = np.corrcoef(hp_train[col].values.T)



mask = np.zeros_like(coeff, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

fig = plt.figure(figsize=(15,10))

sns.heatmap(coeff, vmin = -1, annot = True, mask = mask, square=True, xticklabels = col.values, yticklabels = col.values);
#Categorical Variables

fig, axes = plt.subplots(ncols=4, nrows=4, 

                         figsize=(4 * 4, 4 * 4), sharey=True)



axes = np.ravel(axes)



cols = ['OverallQual','OverallCond','ExterQual','ExterCond','BsmtQual',

        'BsmtCond','GarageQual','GarageCond', 'MSSubClass','MSZoning',

        'Neighborhood','BldgType','HouseStyle','Heating','Electrical','SaleType']



for i, c in zip(np.arange(len(axes)), cols):

    ax = sns.boxplot(x=c, y='SalePrice', data=hp_train, ax=axes[i], palette="Set2")

    ax.set_title(c)

    ax.set_xlabel("")
ntrain = hp_train.shape[0]

ntest = hp_test.shape[0]

y_train = hp_train.SalePrice.values

df_all = pd.concat((hp_train, hp_test), sort=False).reset_index(drop=True)

df_all.drop(['SalePrice'], axis=1, inplace=True)

print("df_all size after concatenation of train and test data is : {}".format(df_all.shape))
df_all.info()
missing_data = pd.DataFrame({'total_missing': df_all.isnull().sum(), 'perc_missing': (df_all.isnull().sum()/len(df_all))*100})

len(missing_data[missing_data.total_missing>0])
fig = plt.figure(figsize=(15,10))

missing_data[missing_data.total_missing>0].sort_values(by='perc_missing')['perc_missing'].plot(kind='barh')

plt.xlabel('Percentage of missing values')

plt.ylabel('Features')

plt.title('Percentage of missing values for different features')

plt.show()
#Creating a list for features with No amenities

no_amen_cat = ['PoolQC','MiscFeature','Alley','Fence','GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType', 'FireplaceQu']

no_amen_num = ['GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea']
df_all1 = df_all.copy() #Creating a copy of original dataframe
for col in no_amen_cat:

    df_all1[col] = df_all1[col].fillna('None')
for col in no_amen_num:

    df_all1[col] = df_all1[col].fillna(0)
mode_replace = ['Electrical', 'MSZoning', 'Utilities', 'SaleType', 'KitchenQual', 'Exterior2nd', 'Exterior1st']

for col in mode_replace:

    df_all1[col] = df_all1[col].fillna(df_all1[col].mode()[0]) 
df_all1["Functional"] = df_all1["Functional"].fillna("Typ")
df_all1.isna().sum()[df_all1.isna().sum()>0].sort_values(ascending=False)
df_all1["LotFrontage"] = df_all1.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))
df_all1.isna().sum()[df_all1.isna().sum()>0].sort_values(ascending=False)
df_all2 = df_all1.copy() #Creating a copy of original dataframe
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir')

# process columns, apply LabelEncoder to categorical features

for c in cols:

    lbl = LabelEncoder() 

    lbl.fit(list(df_all2[c].values)) 

    df_all2[c] = lbl.transform(list(df_all2[c].values))



# shape        

print('Shape : {}'.format(df_all2.shape))
numeric_feats = df_all2.dtypes[df_all2.dtypes != "object"].index



# Check the skew of all numerical features

skewed_feats = df_all2[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

print("\nSkew in numerical features: \n")

skewness = pd.DataFrame({'Skew' :skewed_feats})

skewness.head(10)
skewness = skewness[abs(skewness) > 0.5]

print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))



from scipy.special import boxcox1p

skewed_features = skewness.index

lam = 0.45

for feat in skewed_features:

    df_all2[feat] = boxcox1p(df_all2[feat], lam)
#Dropping dominating features with more than 95% same values

df_all2 = df_all2.drop(['Street', 'Utilities', 'Condition2', 'RoofMatl', 'Heating', 'PoolQC'], axis = 1)
df_all2 = pd.get_dummies(df_all2).reset_index(drop=True)

print(df_all2.shape)
#Adding new features from existing features

df_all2['TotalSF']=df_all2['TotalBsmtSF'] + df_all2['1stFlrSF'] + df_all2['2ndFlrSF']



df_all2['Total_sqr_footage'] = (df_all2['BsmtFinSF1'] + df_all2['BsmtFinSF2'] +

                                 df_all2['1stFlrSF'] + df_all2['2ndFlrSF'])



df_all2['Total_Bathrooms'] = (df_all2['FullBath'] + (0.5 * df_all2['HalfBath']) +

                               df_all2['BsmtFullBath'] + (0.5 * df_all2['BsmtHalfBath']))



df_all2['Total_porch_sf'] = (df_all2['OpenPorchSF'] + df_all2['3SsnPorch'] +

                              df_all2['EnclosedPorch'] + df_all2['ScreenPorch'] +

                              df_all2['WoodDeckSF'])

#Creating a binary features from existing features

df_all2['haspool'] = df_all2['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

df_all2['has2ndfloor'] = df_all2['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

df_all2['hasgarage'] = df_all2['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

df_all2['hasbsmt'] = df_all2['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

df_all2['hasfireplace'] = df_all2['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
df_all2.shape
train = df_all2[:ntrain]

test = df_all2[ntrain:]

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor, AdaBoostRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline, Pipeline

from sklearn.preprocessing import RobustScaler, StandardScaler

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

import xgboost as xgb

import lightgbm as lgb

from  sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer, mean_squared_error

from sklearn.linear_model import LassoLarsCV, RidgeCV, ElasticNetCV

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import MinMaxScaler

from sklearn.svm import LinearSVR

from sklearn.pipeline import make_pipeline, make_union

from tpot.builtins import StackingEstimator

from tpot.builtins import ZeroCount

from sklearn.decomposition import PCA

from imblearn.pipeline import make_pipeline
#Defining score as we will need to score for multiple models

def score(model, train_x, train_y):

    score = np.sqrt(-cross_val_score(model, train_x, train_y, cv=5, scoring="neg_mean_squared_error"))

    print(score.mean(), score.std())
from tpot import TPOTRegressor

tpot = TPOTRegressor(generations=1,verbosity=2,scoring='neg_mean_squared_error')
tpot.fit(train.values, y_train)
#top performer

tpot.score(train.values, y_train)
models_tested = pd.DataFrame(tpot.evaluated_individuals_).transpose()
#We will using top 5 models for our final submission

models_tested.sort_values(['internal_cv_score'], ascending=False).head(10)
exported_pipeline1 = make_pipeline(

    StandardScaler(),

    ElasticNetCV(l1_ratio=0.7000000000000001, tol=0.001, cv=5)

)



exported_pipeline1.fit(train.values, y_train)
score(exported_pipeline1, train.values, y_train)

exported_pipeline_pred1 = exported_pipeline1.predict(train.values)

score(exported_pipeline1, train.values, exported_pipeline_pred1)
exported_pipeline2 = make_pipeline(

    MinMaxScaler(),

    ElasticNetCV(l1_ratio=0.7000000000000001, tol=0.01, cv=5)

)



exported_pipeline2.fit(train.values, y_train)
score(exported_pipeline2, train.values, y_train)

exported_pipeline_pred2 = exported_pipeline2.predict(train)

score(exported_pipeline2, train.values, exported_pipeline_pred2)
exported_pipeline3 = make_pipeline(

    ZeroCount(),

    PCA(iterated_power=4, svd_solver='randomized'),

    RidgeCV()

)



exported_pipeline3.fit(train.values, y_train)
score(exported_pipeline3, train.values, y_train)

exported_pipeline_pred3 = exported_pipeline3.predict(train)

score(exported_pipeline3, train.values, exported_pipeline_pred3)
exported_pipeline4 = make_pipeline(

    LassoLarsCV(normalize=True, max_iter=60, cv=5)

)



exported_pipeline4.fit(train.values, y_train)
score(exported_pipeline4, train.values, y_train)

exported_pipeline_pred4 = exported_pipeline4.predict(train)

score(exported_pipeline4, train.values, exported_pipeline_pred4)
exported_pipeline5 = make_pipeline(

    RidgeCV()

)



exported_pipeline5.fit(train.values, y_train)
score(exported_pipeline5, train.values, y_train)

exported_pipeline_pred5 = exported_pipeline5.predict(train)

score(exported_pipeline5, train.values, exported_pipeline_pred5)
#Checking score after applying equal wightage to all models

np.sqrt(mean_squared_error(y_train,(exported_pipeline_pred1*0.2 + exported_pipeline_pred2*0.2 +

               exported_pipeline_pred3*0.2 + exported_pipeline_pred4*0.2 + exported_pipeline_pred5*0.2)))
#Here we will be deciding weightage of the models based on random selection which gives the least loss

best_value = []

min_value=1

for i in range(10000):

    random = np.random.dirichlet(np.ones(5),size=1)

    best_value.append(np.sqrt(mean_squared_error(y_train,(exported_pipeline_pred1*random[0][0] + exported_pipeline_pred2*random[0][1] +

               exported_pipeline_pred3*random[0][2] + exported_pipeline_pred4*random[0][3] + exported_pipeline_pred5*random[0][4]))))

     

    if(np.min(best_value) < min_value):

        min_value = np.min(best_value)

        min_array = random
np.min(best_value)
min_array
def blend_models(sub):

    return (exported_pipeline1.predict(sub)*min_array[0][0] + 

            exported_pipeline2.predict(sub)*min_array[0][1] + 

            exported_pipeline3.predict(sub)*min_array[0][2] + 

            exported_pipeline4.predict(sub)*min_array[0][3] + 

            exported_pipeline4.predict(sub)*min_array[0][4])
print('Predict submission')

submission = pd.DataFrame()

submission['Id'] = test_ID

submission['SalePrice'] = np.floor(np.expm1(blend_models(test)))
submission.head()
submission.to_csv('submission.csv',index=False)