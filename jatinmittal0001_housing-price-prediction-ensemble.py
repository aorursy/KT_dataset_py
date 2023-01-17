# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import warnings

warnings.filterwarnings("ignore")

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, RobustScaler

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import seaborn as sns

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')

test_data_copy = pd.read_csv('../input/test.csv')

n_rows_train = train_data.shape[0]

Y_train = train_data.iloc[:,-1]

#train_data = train_data.iloc[:,:-1]



a = train_data.append(test_data, sort=False)



#total_data.drop(total_data[(total_data['OverallQual']<5) & (total_data['SalePrice']>200000)].index, inplace=True)

#total_data.drop(total_data[(total_data['GrLivArea']>4000) & (total_data['SalePrice']<300000)].index, inplace=True)

#total_data.reset_index(drop=True, inplace=True)



total_data = a.copy(deep=True)

total_data = total_data.drop(['Id','SalePrice'],axis=1)

total_data.head()

print(test_data.shape)
train_data.head()
print("Size of dataset: ", train_data.shape[0])
#correlation matrix

corrmat = train_data.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);
# NOTE: we can draw scatter plots, it is  just for visualization of relation between different variables

sns.set()

cols = ['SalePrice','OverallQual', 'GrLivArea','TotalBsmtSF', 'FullBath','YearBuilt']

sns.pairplot(train_data[cols], height=2.5)

plt.show()
#Missing data treatment

total_missing_values = total_data.isnull().sum().sort_values(ascending=False)

percentage_missing_data = (100*(total_data.isnull().sum()/total_data.isnull().count())).sort_values(ascending=False)

missing_data = pd.concat([total_missing_values, percentage_missing_data], axis=1, keys=['total_missing_values','percentage_missing_data'])

missing_data.head(20)
#missing value treatment

#we can see some of the features which have high missing values are categorical, 

#so we will replce their missing value by "None" which represents NA category as given in variable description

total_data["PoolQC"] = total_data["PoolQC"].fillna("None") 

total_data["MiscFeature"] = total_data["MiscFeature"].fillna("None") 

total_data["Alley"] = total_data["Alley"].fillna("None") 

total_data["Fence"] = total_data["Fence"].fillna("None")

total_data["FireplaceQu"] = total_data["FireplaceQu"].fillna("None") 



# LotFrontage is a continuous variable, so we replace missing values from houses of same neighborhood

# and take their median

total_data["LotFrontage"] = total_data.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))



total_data["GarageCond"] = total_data["GarageCond"].fillna("None") 

total_data["GarageQual"] = total_data["GarageQual"].fillna("None") 

total_data["GarageFinish"] = total_data["GarageFinish"].fillna("None") 

total_data["GarageType"] = total_data["GarageType"].fillna("None")



# we have replaced garage variables by none i.e. they don't have garage, so we can replace numeric

# variables of garage =0 

total_data["GarageYrBlt"] = total_data["GarageYrBlt"].fillna(0)

total_data["GarageCars"] = total_data["GarageCars"].fillna(0)

total_data["GarageArea"] = total_data["GarageArea"].fillna(0)



for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    total_data[col] = total_data[col].fillna(0)

    

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    total_data[col] = total_data[col].fillna('None')

    

total_data["MasVnrType"] = total_data["MasVnrType"].fillna("None")

total_data["MasVnrArea"] = total_data["MasVnrArea"].fillna(0)



# MSZoning is categorical variable but doesn't have any NA category, so we replace missing values 

# by most occured value in that variable

total_data['MSZoning'] = total_data['MSZoning'].fillna(total_data['MSZoning'].mode()[0])

total_data['MSZoning'] = total_data['MSZoning'].fillna(total_data['MSZoning'].mode()[0])

# NOTE: there are other variables which have 1or 2 missing values, they are very los, we can even drop 

# those obervations

# this variable has same value for all observations except 3, so we drop it

total_data = total_data.drop(['Utilities'], axis=1)



total_data["Functional"] = total_data["Functional"].fillna("Typ")

for col in ('KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType'):

    total_data[col] = total_data[col].fillna(total_data[col].mode()[0])

total_data['MSSubClass'] = total_data['MSSubClass'].fillna("None")



total_data['Electrical'] = total_data['Electrical'].fillna(total_data['Electrical'].mode()[0])

total_data['YearsSinceRemodel'] = total_data['YrSold'].astype(int) - total_data['YearRemodAdd'].astype(int)
total_data['Total_sqr_footage'] = (total_data['BsmtFinSF1'] + total_data['BsmtFinSF2'] +

                                 total_data['1stFlrSF'] + total_data['2ndFlrSF'])



total_data['Total_Bathrooms'] = (total_data['FullBath'] + (0.5*total_data['HalfBath']) + 

                               total_data['BsmtFullBath'] + (0.5*total_data['BsmtHalfBath']))



total_data['Total_porch_sf'] = (total_data['OpenPorchSF'] + total_data['3SsnPorch'] +

                              total_data['EnclosedPorch'] + total_data['ScreenPorch'] +

                             total_data['WoodDeckSF'])





#simplified features

total_data['haspool'] = total_data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

total_data['has2ndfloor'] = total_data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

total_data['hasgarage'] = total_data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

total_data['hasbsmt'] = total_data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

total_data['hasfireplace'] = total_data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
# converting some numericla variiables that really are categories

total_data['MSSubClass'] = total_data['MSSubClass'].apply(str)
sns.distplot(total_data['YrSold'], kde=False)
total_data['YrSold'] = 2010 - total_data['YrSold']
sns.boxplot(x = train_data['MoSold'],y= train_data['SalePrice'])
sns.distplot(total_data['MoSold'])
# Monthes with the largest number of deals may be significant

total_data['season'] = total_data.MoSold.replace( {1: 0, 

                                   2: 0, 

                                   3: 0, 

                                   4: 1,

                                   5: 1, 

                                   6: 1,

                                   7: 1,

                                   8: 0,

                                   9: 0,

                                  10: 0,

                                  11: 0,

                                  12: 0})



# Lable Encoding OverallQual

total_data = total_data.replace({'ExterQual':{'None':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}})

total_data = total_data.replace({'ExterCond':{'None':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}})

total_data = total_data.replace({'BsmtQual': {'NA':0,'None':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}})

total_data = total_data.replace({'BsmtCond': {'NA':0,'None':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}})

total_data = total_data.replace({'BsmtFinType1':{'NA':0,'None':0, 'Unf':1, 'LwQ':2, 'Rec':3, 'BLQ':4, 'ALQ':5, 'GLQ':6}})

total_data = total_data.replace({'BsmtFinType2':{'NA':0,'None':0, 'Unf':1, 'LwQ':2, 'Rec':3, 'BLQ':4, 'ALQ':5, 'GLQ':6}})

total_data = total_data.replace({'HeatingQC': {'NA':0,'None':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}})

total_data= total_data.replace({'CentralAir':{'N':0, 'None':0,'Y':1}})

total_data = total_data.replace({'KitchenQual': {'NA':0,'None':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}})

total_data = total_data.replace({'FireplaceQu': {'NA':0,'None':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}})

total_data = total_data.replace({'GarageQual': {'NA':0,'None':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}})

total_data = total_data.replace({'GarageCond': {'NA':0,'None':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}})

total_data = total_data.replace({'PoolQC': {'NA':0,'None':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}})

total_data = total_data.replace({'GarageFinish': {'NA':0,'None':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}})

total_data = total_data.replace({'Fence':{'NA':0,"None":0, 'MnWv':1, 'GdWo':2, 'MnPrv':3, 'GdPrv':4}})

total_data = total_data.replace({'LandSlope':{'None':0, 'Sev':1, 'Mod':2, 'Gtl':3}})

total_data = total_data.replace({'Functional':{'None':0, 'Sal':1, 'Sev':2, 'Maj2':3, 'Maj1':4, 'Mod':5, 'Min2':6, 'Min1':7, 'Typ':8}})

total_data = total_data.fillna(0)
print('Shape all_data: {}'.format(total_data.shape))
col = (total_data.dtypes[total_data.dtypes == "object"].index).tolist()
'''

for c in col:

    total_data[c] = total_data[c].astype('category')

    total_data[c] = total_data[c].cat.codes

'''
train_data.head()
train_data.loc[1,'Id']
n_row = np.shape(train_data)[0]

min = train_data.loc[1,'Id']

for x in range(n_row):

        if min > train_data.loc[x,'Id']:

            min = x
min
from scipy.stats import skew

#reducing skewness of all features and target variable

Y_train1 = np.log1p(Y_train)



n_rows_train = train_data.shape[0];

train_data = total_data.iloc[:n_rows_train,:]

test_data = total_data.iloc[n_rows_train:,:]

#finding numerical features



features = total_data.dtypes[total_data.dtypes != "object"].index



#finding skewness of all variables

skewed_feats = total_data[features].apply(lambda x: skew(x.dropna()))

#adjusting features having skewness >0.5

skewed_feats = skewed_feats[skewed_feats > 0.5]

skewed_feats = skewed_feats.index

total_data[skewed_feats] = np.log1p(total_data[skewed_feats])
# although we have applied norm distribution to all numeric variables, but here we will plot graph of

# target variable only

# NOTE: y axisis probability density estimates, # to get freq, use kde= False

chart1, ax1 = plt.subplots()

sns.distplot(Y_train, norm_hist=False,ax=ax1)

#after applying logarithm, we get plot relatively simiar to norm distribution

chart2, ax2 = plt.subplots()

sns.distplot(Y_train1, norm_hist=False,ax=ax2)


# now converting categorical features to one hot encoding vectors

total_data_oh = pd.get_dummies(total_data)

total_data_oh.head()
#split between X and test data

X = total_data_oh.iloc[:n_rows_train,:]

test_data = total_data_oh.iloc[n_rows_train:,:]

print(X.shape)

col = X.columns

#scaling the data

scaler = RobustScaler()

scaler = scaler.fit(X)



X = scaler.transform(X)

test_data = scaler.transform(test_data)

X = pd.DataFrame(X, columns = col)

test_data = pd.DataFrame(test_data, columns = col)

X.head()
'''

#PCA

from sklearn.decomposition import PCA

pca = PCA(n_components=200)

pca = pca.fit(X)

principalComponents_train = pca.transform(X)

principalComponents_test = pca.transform(test_data)

X = pd.DataFrame(principalComponents_train)

test_data = pd.DataFrame(principalComponents_test)

'''
#  split X between training and testing set

x_train, x_test, y_train, y_test = train_test_split(X,Y_train1, test_size=0.3, shuffle=True) 





# I have tried PCA code above, but the model is performing bad with it. So, I am not applying PCA.

from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import mean_squared_error

'''

Cross-validation is a resampling procedure used to evaluate machine learning models on a limited 

data sample. It is just used to check how this particular model will perform on different test sets. 

It is not used to say whether this particuclar model is best or not.

At the end final predictions are made by model.fit and model.predict only.

'''

#Validation function

n_folds = 5



def rmse_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X.values)

    rmse= np.sqrt(-cross_val_score(model, X, Y_train1, scoring="neg_mean_squared_error", cv = kf)) 

    # this computes rmse of each fold

    return(rmse)
from sklearn.metrics import mean_squared_error

from math import sqrt



def rmse(y_pred, y_test):

    rmse = sqrt(mean_squared_error(y_test,y_pred))

    return rmse
from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.0005, max_iter=10000)

lasso.fit(X, Y_train1)

#pred_lasso = lasso.predict(x_test)

#print(rmse(pred_lasso,y_test))

#print(rmse_cv(lasso).mean())
from sklearn.linear_model import ElasticNet

elastic_net_model = ElasticNet(alpha=0.0005, l1_ratio=.7, random_state=3)

elastic_net_model.fit(X, Y_train1)

#pred_elastic = elastic_net_model.predict(x_test)

#print(rmse(pred_elastic,y_test))

#print(rmse_cv(elastic_net_model).mean())
#making predictions on test set

y_pred_elastic_net_test_data = np.expm1(elastic_net_model.predict(test_data))

y_pred_lasso_test_data = np.expm1(lasso.predict(test_data))
pred = 0.3*y_pred_elastic_net_test_data + 0.7*y_pred_lasso_test_data
solution = pd.DataFrame({"id":test_data_copy.Id, "SalePrice":pred})

solution.to_csv("housing_pricefinal.csv", index = False)
#y_pred_lasso_test_data