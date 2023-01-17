# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns #plotting package

import matplotlib.pyplot as plt  # Matlab-style plotting

from sklearn.preprocessing import LabelEncoder



from scipy import stats

from scipy.stats import norm, skew 







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input")) 



import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



# Any results you write to the current directory are saved as output.
# download .csv file from Kaggle Kernel



from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv()

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



# create a random sample dataframe

df = pd.DataFrame(np.random.randn(50, 4), columns=list('ABCD'))



# create a link to download the dataframe
# Read data 

train_data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")
# Take a quick look at the data structure

train_data.head()
# Take a quick look at the data structure

test_data.head()
# Take a quick look at the data structure

print("The size of train data", train_data.shape)

print("The size of test data", test_data.shape)



# From the above, you can notice that train and test data almost 

# share the same number of columns, except "Sale Price" in the training set. 



# total number of rows in training data set

# ntrain = train_data.shape[0]
# The `info()` method is useful to get a quick description of the data, in particular 

# the total number of rows and each attributes types and number of non-null values.

train_data.info()
#correlation matrix

import matplotlib.pyplot as plt

import seaborn as sns



corrmat = train_data.corr()

f, ax = plt.subplots(figsize=(20, 9))

sns.heatmap(corrmat, vmax=.8, annot=True);
# Correlation matrix (heatmap style).

# correlation matri



corrmat = train_data.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True); 
# most correlated features

corrmat = train_data.corr()

top_corr_features = corrmat.index[abs(corrmat["SalePrice"])>0.5]

plt.figure(figsize=(10,10))

g = sns.heatmap(train_data[top_corr_features].corr(),annot=True,cmap="RdYlGn")
# Scatter Plot for numerical variables



li_cat_feats = [ 'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'GarageArea', 'LotFrontage', 'LotArea', 'MasVnrArea']   

target = 'SalePrice'

nr_rows = 2

nr_cols = 4



fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*4,nr_rows*3))



for r in range(0,nr_rows):

    for c in range(0,nr_cols):  

        i = r*nr_cols+c

        if i < len(li_cat_feats):

            sns.scatterplot(x=li_cat_feats[i], y=target, data=train_data, ax = axs[r][c])

    

plt.tight_layout()    

plt.show()  
#li_cat_feats = list(categorical_feats)



# Box plot for categorical variables



li_cat_feats = ['OverallQual','GarageCars', 'YearBuilt', 'YearRemodAdd', 'Fireplaces', 'GarageYrBlt', 'TotRmsAbvGrd']

target = 'SalePrice'

nr_rows = 2

nr_cols = 4



fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*4,nr_rows*3))



for r in range(0,nr_rows):

    for c in range(0,nr_cols):  

        i = r*nr_cols+c

        if i < len(li_cat_feats):

            sns.boxplot(x=li_cat_feats[i], y=target, data=train_data, ax = axs[r][c])

plt.tight_layout()    

plt.show()
# Copy/Paste the the ID of both training and testing dataset for future use

# We will be using ID, when we make submission to the compitions

train_ID = train_data['Id']

test_ID = test_data['Id']
# Drop the ID as they are not necessary for SalePrice prediction

train_data.drop("Id", axis = 1, inplace = True)

test_data.drop("Id", axis = 1, inplace = True)
def plot_dist_norm(dist, title):

    sns.distplot(dist, fit=norm);

    (mu, sigma) = norm.fit(dist);

    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')

    plt.ylabel('Frequency')

    plt.title(title)

    fig = plt.figure()

    res = stats.probplot(dist, plot=plt)

    plt.show()
# Plit SalePrice Distribution

plot_dist_norm(train_data['SalePrice'], 'SalePrice Distribution')

# transform the SalePrice

transform_log = np.log1p(train_data["SalePrice"])

plot_dist_norm(transform_log, 'log(SalePrice) Distribution') 



# Copy back the SalePrice to train_data 

train_data["SalePrice"] = transform_log
var = 'GrLivArea'

data = pd.concat([train_data['SalePrice'], train_data[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice')
# Show me ourliers

train_data[train_data.GrLivArea > 4500] 



# The outlier exists at 523 and 1298 index 

# To handle the outlier, we need to do two things

# (1) Remove the 523 and 1298 rows from train_data

# (2) Remove the 523 and 1298 rows from SalePrice
# (1)

# As you can see that there are some outliers in GrLivArea. That may harm ML model. 

# Let's remove these outliers 

# Remove GrLivArea value greater than 4500 (See the above image)

train_data = train_data[train_data.GrLivArea < 4500]
# (2)

# Remove the 523 and 1298 rows from SalePrice as the corresponding 

# train_data  are outlier and they are removed 



transform_log = transform_log.drop([523 , 1298], axis=0)
all_data = pd.concat((train_data, test_data)).reset_index(drop=True)
# Drop "SalePrice" as it is our target variable

all_data.drop(['SalePrice'], axis=1, inplace=True)
# get information about missing values 

total = all_data.isnull().sum().sort_values(ascending=False)

percent = (all_data.isnull().sum()/all_data.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data[total > 0]
# Imputation

# Our stratergy is to fill NA values with Mode of the column for some columns (Not All columnns!!!) as that value is appearing frequently

# This is "Data Scientist's Call!!" (There is no rule of thumb)



for col in ['SaleType', 'KitchenQual' , 'Exterior2nd' , 'Exterior1st' , 'Electrical' , 'Functional' , 'MSZoning' ] :

    all_data[col] = all_data[col].fillna(all_data[col].mode()[0]) 

    

# Fill NA with Zero

for col in ['BsmtFinSF1', 'BsmtFinSF2' , 'GarageCars' , 'GarageArea' , 'TotalBsmtSF' , 'BsmtUnfSF' , 'BsmtHalfBath' , 'BsmtFullBath' , 'MasVnrArea' ]:

    all_data[col]=all_data[col].fillna(0)



# fill NA with None

for col in ['MasVnrType','FireplaceQu' , 'Fence' , 'PoolQC' , 'MiscFeature' ,'Alley' , 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'GarageType', 'GarageFinish' , 'GarageCond' , 'GarageQual'] :

    all_data[col]=all_data[col].fillna('None')
# Drop the columns

all_data.drop(['Utilities', 'Street' , 'PoolQC'], axis=1, inplace=True)
# We can create a new feature that may be related more directly to SalePrice (in the hope of better linearity).



all_data['Total_sqr_footage'] = (all_data['BsmtFinSF1'] + all_data['BsmtFinSF2'] + all_data['1stFlrSF'] + all_data['2ndFlrSF'])

all_data['Total_Bathrooms'] = (all_data['FullBath'] + (0.5 * all_data['HalfBath']) + all_data['BsmtFullBath'] + (0.5 * all_data['BsmtHalfBath']))

all_data['Total_porch_sf'] = (all_data['OpenPorchSF'] + all_data['3SsnPorch'] + all_data['EnclosedPorch'] + all_data['ScreenPorch'] + all_data['WoodDeckSF'])

all_data['haspool'] = all_data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

all_data['has2ndfloor'] = all_data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

all_data['hasgarage'] = all_data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

all_data['hasbsmt'] = all_data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

all_data['hasfireplace'] = all_data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
# There are some wrong values in GarageYrBlt

all_data['GarageYrBlt'][all_data['GarageYrBlt']>2150]
all_data['GarageYrBlt']=all_data['GarageYrBlt'].fillna(all_data['YearBuilt'][ all_data['GarageYrBlt'].isnull()])

all_data['GarageYrBlt'][all_data['GarageYrBlt']>2018] = all_data['YearBuilt'][all_data['GarageYrBlt']>2018]
# I think it is reasonable to guess the values according to "Neighborhood", as suggested by others.



all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
all_data['MSSubClass'] = all_data['MSSubClass'].astype(str)

all_data['OverallCond'] = all_data['OverallCond'].astype(str)

all_data['BsmtFullBath'] = all_data['BsmtFullBath'].astype(str)

all_data['BsmtHalfBath'] = all_data['BsmtHalfBath'].astype(str)

all_data['YrSold'] = all_data['YrSold'].astype(str)

all_data['MoSold'] = all_data['MoSold'].astype(str)

all_data['GarageYrBlt'] = all_data['GarageYrBlt'].astype(str)

all_data['YearBuilt'] = all_data['YearBuilt'].astype(str)

all_data['YearRemodAdd'] = all_data['YearRemodAdd'].astype(str)

all_data['GarageCars'] = all_data['GarageCars'].astype(str)
# We can create a new feature that may be related more directly to SalePrice (in the hope of better linearity).

all_data['TotalSF']=all_data['TotalBsmtSF']+all_data['1stFlrSF']+all_data['2ndFlrSF']
cols = ('ExterCond','HeatingQC', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'CentralAir', 'MSSubClass', 'OverallCond',

        'YrSold', 'MoSold','GarageYrBlt','YearBuilt','YearRemodAdd', 'BsmtHalfBath','BsmtFullBath', 'GarageCars')

    

# process columns, apply LabelEncoder to categorical features

for c in cols:

    lbl = LabelEncoder() 

    lbl.fit(list(all_data[c].values)) 

    all_data[c] = lbl.transform(list(all_data[c].values))
all_data.FireplaceQu = all_data.FireplaceQu.astype('category', ordered=True, categories=['None','Po','Fa','TA','Gd','Ex']).cat.codes

all_data.BsmtQual = all_data.BsmtQual.astype('category', ordered=True, categories=['None','Fa','TA','Gd','Ex']).cat.codes

all_data.BsmtCond = all_data.BsmtCond.astype('category', ordered=True, categories=['None','Po','Fa','TA','Gd']).cat.codes

all_data.GarageQual = all_data.GarageQual.astype('category', ordered=True, categories=['None','Po','Fa','TA','Gd','Ex']).cat.codes

all_data.GarageCond = all_data.GarageCond.astype('category', ordered=True, categories=['None','Po','Fa','TA','Gd','Ex']).cat.codes

all_data.ExterQual = all_data.ExterQual.astype('category', ordered=True, categories=['Fa','TA','Gd','Ex']).cat.codes

#all_data.PoolQC = all_data.PoolQC.astype('category', ordered=True, categories=['None','Fa','Gd','Ex']).cat.codes

all_data.KitchenQual = all_data.KitchenQual.astype('category', ordered=True, categories=['Fa','TA','Gd','Ex']).cat.codes
skewed_feats = all_data[all_data.dtypes[all_data.dtypes != "object"].index].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

skewness = pd.DataFrame({'Skew' :skewed_feats})

f, ax = plt.subplots(figsize=(15, 12))

plt.xticks(rotation='90')

sns.barplot(x=skewed_feats.index, y=skewed_feats)

plt.xlabel('Features', fontsize=15)

plt.ylabel('Skewness', fontsize=15)

plt.title('Skewness by feature', fontsize=15)
from scipy.special import boxcox1p #for Box Cox transformation



for feat in skewness[abs(skewness)>0.5].index:

    all_data[feat] = boxcox1p(all_data[feat], 0.15)
# Dummification 

all_data = pd.get_dummies(all_data)



# To avoid dummy variable trap

all_data.drop(["Neighborhood_Veenker", "RoofMatl_WdShngl", "RoofStyle_Shed" , "SaleCondition_Partial" , "SaleType_WD"], axis=1, inplace=True)





# Now our training and testing dataset is imputed now, let's now separate as it.

# ntrain variable indicates the total number of rows in the training dataset



# We will use this dataset to train our model and test the accuracy of the model 

train_data = all_data[:train_data.shape[0]]



# Test data set - this data set will be used for submissions to the competitions

# We will keep this dataset aside now (DO NOT TOUCH!!!)

# We will use this dataset once, we finalize our model.

test_data = all_data[train_data.shape[0]:]



# target variable - SalePrice

y_train = transform_log
y_train.shape ,  train_data.shape
# Import libraries

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
#Cross-validation function

n_folds = 10



def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train_data.values)

    rmse= np.sqrt(-cross_val_score(model, train_data.values, y_train, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

score = rmsle_cv(KRR)

print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))

score = rmsle_cv(lasso)

print("Lasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))

score = rmsle_cv(ENet)

print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5)

score = rmsle_cv(GBoost)

print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state =7, nthread = -1)

score = rmsle_cv(model_xgb)

print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.05, n_estimators=720,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

score = rmsle_cv(model_lgb)

print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))
LassoMd = lasso.fit(train_data.values,y_train)

ENetMd = ENet.fit(train_data.values,y_train)

KRRMd = KRR.fit(train_data.values,y_train)

GBoostMd = GBoost.fit(train_data.values,y_train)

XGBMd = model_xgb.fit(train_data.values, y_train)

LGBMd = model_lgb.fit(train_data.values, y_train)
finalMd = (np.expm1(LassoMd.predict(test_data.values)) + 

           np.expm1(ENetMd.predict(test_data.values)) + 

           np.expm1(KRRMd.predict(test_data.values)) + 

           np.expm1(GBoostMd.predict(test_data.values))  + 

           np.expm1(XGBMd.predict(test_data.values))  +

           np.expm1(LGBMd.predict(test_data.values)) ) / 6

finalMd
# Code to submit your data to the competition





my_submission = pd.DataFrame({'Id': test_ID, 'SalePrice': finalMd}) 



# you could use any filename. We choose submission here

my_submission.to_csv('submission.csv', index=False)
my_submission