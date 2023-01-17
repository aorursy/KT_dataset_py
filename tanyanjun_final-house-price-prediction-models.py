# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

sns.set_style("darkgrid")

%matplotlib inline



from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

from sklearn import linear_model

from sklearn.neighbors import KNeighborsRegressor

from sklearn.preprocessing import PolynomialFeatures

from sklearn import metrics

from sklearn.model_selection import cross_val_score

import seaborn as sns

from sklearn import preprocessing

from sklearn import linear_model, svm, gaussian_process

from sklearn.ensemble import RandomForestRegressor

import numpy as np





from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV

from sklearn.preprocessing import LabelEncoder, StandardScaler, MaxAbsScaler, QuantileTransformer



#models

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, LinearRegression, Ridge, RidgeCV

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor



#validation libraries

from sklearn.model_selection import KFold, StratifiedKFold

from IPython.display import display

from sklearn import metrics





from mpl_toolkits.mplot3d import Axes3D

import folium

from folium.plugins import HeatMap



from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import SelectKBest,f_regression

from sklearn.model_selection import KFold,cross_val_score

from sklearn.linear_model import LinearRegression,BayesianRidge,ElasticNet,Lasso,SGDRegressor,Ridge

from sklearn.kernel_ridge import KernelRidge

from sklearn.preprocessing import LabelEncoder,Imputer,OneHotEncoder,RobustScaler,StandardScaler,Imputer



from scipy import stats



import warnings

warnings.filterwarnings('ignore')



from IPython.display import HTML, display



import statsmodels.api as sm

from statsmodels.formula.api import ols

from statsmodels.sandbox.regression.predstd import wls_prediction_std



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
housetrain= pd.read_csv("../input/train.csv")

housetest = pd.read_csv("../input/test.csv")

housetrain.head()

housetest.head()

df=pd.DataFrame(housetrain)

df.info()
test=pd.DataFrame(housetest)

test.info()
missing_val_count_by_column = (df.isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 0])
print(missing_val_count_by_column[missing_val_count_by_column > 0].plot(kind='bar'))
# fill up MSZoning with the mode value

df['MSZoning'] = df['MSZoning'].fillna(df['MSZoning'].mode()[0])



# LotFrontage : Since the area of each street connected to the house property most likely have a similar area to other houses in its neighborhood , we can fill in missing values by the median LotFrontage of the neighborhood.

df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))



# from the data description file, NA = No Alley Access

df['Alley'].fillna(0, inplace=True)



# fill up NA values with mode

df['Utilities'] = df['Utilities'].fillna(df['Utilities'].mode()[0])



# since both Exterior1st and 2nd only has 2 missing value, substitute with mode

df['Exterior1st'] = df['Exterior1st'].fillna(df['Exterior1st'].mode()[0])

df['Exterior2nd'] = df['Exterior2nd'].fillna(df['Exterior2nd'].mode()[0])



# fill up MasVnrType with the mode value

df["MasVnrType"] = df["MasVnrType"].fillna(df['MasVnrType'].mode()[0])

df["MasVnrArea"] = df["MasVnrArea"].fillna(df['MasVnrArea'].mode()[0])



# for these columns, NA = No Basement

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    df[col] = df[col].fillna('None')

    

# for these columns, NA is likely to be 0 due to no basement

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    df[col] = df[col].fillna(0)

    

# substitue NA value here with mode

df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])



# substitute NA value with mode

df['KitchenQual'] = df['KitchenQual'].fillna(df['KitchenQual'].mode()[0])



# if no value, assume Typ, typical is also mode value

df['Functional'] = df['Functional'].fillna(df['Functional'].mode()[0])



# NA = No Fireplace

df['FireplaceQu'] = df['FireplaceQu'].fillna('None')



# for these columns, NA = No Garage

for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):

    df[col] = df[col].fillna('None')

    

# as there is no garage, NA value for this column is set to zero

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    df[col] = df[col].fillna(0)

    

# NA = no pool

df['PoolQC'] = df['PoolQC'].fillna('None')



# NA = no fence

df['Fence'] = df['Fence'].fillna('None')



#Misc Feature, NA = None

df['MiscFeature'] = df['MiscFeature'].fillna('None')



#sale type, only have 1 NA value. substitute it with mode value

df['SaleType'] = df['SaleType'].fillna(df['SaleType'].mode()[0])



# checking for any null value left

df.isnull().sum().sum()
# fill up MSZoning with the mode value

test['MSZoning'] = test['MSZoning'].fillna(test['MSZoning'].mode()[0])



# LotFrontage : Since the area of each street connected to the house property most likely have a similar area to other houses in its neighborhood , we can fill in missing values by the median LotFrontage of the neighborhood.

test["LotFrontage"] = test.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))



# from the data description file, NA = No Alley Access

test["Alley"] = test["Alley"].fillna(0)



# fill up NA values with mode

test['Utilities'] = test['Utilities'].fillna(test['Utilities'].mode()[0])



# since both Exterior1st and 2nd only has 2 missing value, substitute with mode

test['Exterior1st'] = test['Exterior1st'].fillna(test['Exterior1st'].mode()[0])

test['Exterior2nd'] = test['Exterior2nd'].fillna(test['Exterior2nd'].mode()[0])



# fill up MasVnrType with the mode value

test["MasVnrType"] = test["MasVnrType"].fillna(df['MasVnrType'].mode()[0])

test["MasVnrArea"] = test["MasVnrArea"].fillna(df['MasVnrArea'].mode()[0])



# for these columns, NA = No Basement

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    test[col] = test[col].fillna('None')

    

# for these columns, NA is likely to be 0 due to no basement

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    test[col] = test[col].fillna(0)

    

# substitue NA value here with mode

test['Electrical'] = test['Electrical'].fillna(test['Electrical'].mode()[0])



# substitute NA value with mode

test['KitchenQual'] = test['KitchenQual'].fillna(test['KitchenQual'].mode()[0])



# if no value, assume Typ, typical is also mode value

test['Functional'] = test['Functional'].fillna(test['Functional'].mode()[0])



# NA = No Fireplace

test['FireplaceQu'] = test['FireplaceQu'].fillna('None')



# for these columns, NA = No Garage

for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):

    test[col] = test[col].fillna('None')

    

# as there is no garage, NA value for this column is set to zero

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    test[col] = test[col].fillna(0)

    

# NA = no pool

test['PoolQC'] = test['PoolQC'].fillna('None')



# NA = no fence

test['Fence'] = test['Fence'].fillna('None')



#Misc Feature, NA = None

test['MiscFeature'] = test['MiscFeature'].fillna('None')



#sale type, only have 1 NA value. substitute it with mode value

test['SaleType'] = test['SaleType'].fillna(test['SaleType'].mode()[0])



# checking for any null value left

test.isnull().sum().sum()




# feature extraction



numeric_cols = [x for x in df.columns if ('Area' in x) | ('SF' in x)] + ['SalePrice','LotFrontage','MiscVal','EnclosedPorch','3SsnPorch','ScreenPorch','OverallQual','OverallCond','YearBuilt']



for col in numeric_cols:

    df[col] = df[col].astype(float)

numeric_cols
categorical_cols = [x for x in df.columns if x not in numeric_cols]



for col in categorical_cols:

    df[col] = df[col].astype('category')

    

categorical_cols
df['above_200k'] = df['SalePrice'].map(lambda x : 1 if x > 200000 else 0) 

df['above_200k'] = df['above_200k'].astype('category')



df.loc[df['SalePrice']>200000,'above_200k'] = 1

df.loc[df['SalePrice']<=200000,'above_200k'] = 0

df['above_200k'] = df['above_200k'].astype('category')
df['LivArea_Total'] = df['GrLivArea'] + df['GarageArea'] + df['PoolArea']

df[['LivArea_Total','GrLivArea','GarageArea','PoolArea']].head()
## concatenating two different fields together in the same row

df['Lot_desc'] = df.apply(lambda val : val['MSZoning'] + val['LotShape'], axis=1)

df[['Lot_desc','MSZoning','LotShape']].head()
from sklearn.preprocessing import LabelEncoder, StandardScaler, MaxAbsScaler, QuantileTransformer



df['LotArea_norm'] = df['LotArea']



ss = StandardScaler()

mas = MaxAbsScaler()

qs = QuantileTransformer()



df['LotArea_norm'] = ss.fit_transform(df[['LotArea']])

df['LotArea_mas'] = mas.fit_transform(df[['LotArea']])

df['LotArea_qs'] = qs.fit_transform(df[['LotArea']])





df[['LotArea_norm','LotArea_mas','LotArea_qs', 'LotArea']].head(5)
small_df = df[['MSZoning','SalePrice']].copy()

small_df['MSZoning'] = small_df['MSZoning'].astype('category')

small_df.head()
pd.get_dummies(small_df).head(5)
small_df = df[['MSSubClass','SalePrice']].copy()

small_df['MSSubClass'] = small_df['MSSubClass'].astype('category')

small_df.head()
le = LabelEncoder()

trf_MSSubClass = le.fit_transform(small_df['MSSubClass'])

trf_MSSubClass
le.classes_
le.inverse_transform(trf_MSSubClass)

feature_cols = [col for col in df.columns if 'Price' not in col]

print(feature_cols)
df['LogSalePrice'] = np.log(df['SalePrice'])



y = df['LogSalePrice']

X = df[feature_cols]

print(X.head(2),'\n\n', X.head(2))
X_train, X_valid, y_train, y_valid = train_test_split(X,y, test_size=0.2)

print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)
X_numerical = pd.get_dummies(X)

X_numerical.head(5)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from boruta import boruta_py

from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression

from pandas import read_csv





def greedy_elim(df):



    # do feature selection using boruta

    X = df[[x for x in df.columns if x!='SalePrice']]

    y = df['SalePrice']

    #model = RandomForestRegressor(n_estimators=50)

    model = GradientBoostingRegressor(n_estimators=50, learning_rate=0.05)

    # 150 features seems to be the best at the moment. Why this is is unclear.

    feat_selector = RFE(estimator=model, step=1, n_features_to_select=150)



    # find all relevant features

    feat_selector.fit_transform(X.as_matrix(), y.as_matrix())



    # check selected features

    features_bool = np.array(feat_selector.support_)

    features = np.array(X.columns)

    result = features[features_bool]

    #print(result)



    # check ranking of features

    features_rank = feat_selector.ranking_

    #print(features_rank)

    rank = features_rank[features_bool]

    #print(rank)



    print(result) 

    print(rank)











# Minimum price of the data

minimum_price = np.amin(df.SalePrice)



# Maximum price of the data

maximum_price = np.amax(df.SalePrice)





# Mean price of the data

mean_price = np.mean(df.SalePrice)





# Median price of the data

median_price = np.median(df.SalePrice)





# Standard deviation of prices of the data

std_price = np.std(df.SalePrice)



    

# Show the calculated statistics

print("Zillow Housing Price Dataset:\n")

print("Minimum price: ${}".format(minimum_price)) 

print("Maximum price: ${}".format(maximum_price))

print("Mean price: ${}".format(mean_price))

print("Median price ${}".format(median_price))

print("Standard deviation of prices: ${}".format(std_price))
sns.distplot(df['SalePrice'])
newhouse_dm=df.copy()

newhouse_dm.head()
# add the age of the buildings when the houses were sold as a new column

newhouse_dm['Age']=newhouse_dm['YrSold'].astype(int)-newhouse_dm['YearBuilt'].astype(int)



# partition the age into bins

bins = [-2,0,5,10,25,50,75,100,100000]

labels = ['<1','1-5','6-10','11-25','26-50','51-75','76-100','>100']

newhouse_dm['age_binned'] = pd.cut(newhouse_dm['Age'].astype(int), bins=bins, labels=labels)





# add the age of the renovation when the houses were sold as a new column

newhouse_dm['age_remodel']=0

newhouse_dm['age_remodel']=newhouse_dm['YrSold'][newhouse_dm['YearRemodAdd']!=0].astype(int)-newhouse_dm['YearRemodAdd'].astype(int)[newhouse_dm['YearRemodAdd']!=0].astype(int)

newhouse_dm['age_remodel'][newhouse_dm['age_remodel'].isnull()]=0





# histograms for the binned columns

f, axes = plt.subplots(1,2,figsize=(25,5))

p1=sns.countplot(newhouse_dm['age_binned'],ax=axes[0])



# partition the age_remodel into bins

bins = [-2,0,5,10,25,50,75,100000]

labels = ['<1','1-5','6-10','11-25','26-50','51-75','>75']

newhouse_dm['age_remodel_binned'] = pd.cut(newhouse_dm['age_remodel'], bins=bins, labels=labels)





for p in p1.patches:

    height = p.get_height()

    p1.text(p.get_x()+p.get_width()/2,height + 50,height,ha="center")   



p2=sns.countplot(newhouse_dm['age_remodel_binned'],ax=axes[1])

sns.despine(left=True, bottom=True)

for p in p2.patches:

    height = p.get_height()

    p2.text(p.get_x()+p.get_width()/2,height + 200,height,ha="center")

    

axes[0].set(xlabel='Age')

axes[0].yaxis.tick_left()

axes[1].yaxis.set_label_position("right")

axes[1].yaxis.tick_right()

axes[1].set(xlabel='Remodel Age');



# transform the factor values to be able to use in the model

newhouse_dm = pd.get_dummies(newhouse_dm, columns=['age_binned'])





# CentralAir

var = 'CentralAir'

data = pd.concat([df['SalePrice'], df[var]], axis=1)

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=350000);
# OverallQual

var = 'OverallQual'

data = pd.concat([df['SalePrice'], df[var]], axis=1)

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=500000);
# YearBuilt

var = 'YearBuilt'

data = pd.concat([df['SalePrice'], df[var]], axis=1)

data.plot.scatter(x=var, y="SalePrice", ylim=(0, 800000))
var = 'Utilities'

data = pd.concat([df['SalePrice'], df[var]], axis=1)

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=350000);
var = 'SaleCondition'

data = pd.concat([df['SalePrice'], df[var]], axis=1)

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=350000);
# Neighborhood

var = 'Neighborhood'

data = pd.concat([df['SalePrice'], df[var]], axis=1)

f, ax = plt.subplots(figsize=(26, 12))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);
#GrLivArea



var  = 'GrLivArea'

data = pd.concat([df['SalePrice'], df[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
# GarageArea 



var  = 'GarageArea'

data = pd.concat([df['SalePrice'], df[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
# LotArea



var  = 'LotArea'

data = pd.concat([df['SalePrice'], df[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
# BedroomAbvGr

var = 'BedroomAbvGr'

data = pd.concat([df['SalePrice'], df[var]], axis=1)

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);
#TotalBsmtSF



var  = 'TotalBsmtSF'

data = pd.concat([df['SalePrice'], df[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
features = ['SalePrice','MSSubClass','LotFrontage','LotArea','OverallQual','OverallCond','YearBuilt',

            'YearRemodAdd','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF',

            '2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath',

            'BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','BsmtHalfBath','GarageYrBlt',

            'GarageCars','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea',

            'MiscVal','MoSold','GarageArea']



mask = np.zeros_like(df[features].corr(), dtype=np.bool) 

mask[np.triu_indices_from(mask)] = True 



f, ax = plt.subplots(figsize=(16, 12))

plt.title('Pearson Correlation Matrix',fontsize=25)



sns.heatmap(df[features].corr(),linewidths=0.25,vmax=0.7,square=True,cmap="BuGn", #"BuGn_r" to reverse 

            linecolor='w',annot=True,annot_kws={"size":8},mask=mask,cbar_kws={"shrink": .9});
from sklearn import preprocessing



f_names = ['CentralAir', 'Neighborhood']

for x in f_names:

    label = preprocessing.LabelEncoder()

    df[x] = label.fit_transform(df[x])

corrmat = df.corr()

f, ax = plt.subplots(figsize=(20, 9))



k  = 11 # 关系矩阵中将显示10个特征

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(df[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, \

                 square=True, fmt='.2f', annot_kws={'size': 10}, cmap='PiYG',yticklabels=cols.values, xticklabels=cols.values)

plt.show()
sns.set()

cols = ['SalePrice','OverallQual','GrLivArea', 'GarageCars','TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt','GarageArea','1stFlrSF']

sns.pairplot(df[cols], size = 2.5)

plt.show()
evaluation = pd.DataFrame({'Model': [],

                           'Details':[],

                           'Mean Squared Error (MSE)':[],

                           'Root Mean Squared Error (RMSE)':[],

                           'R-squared (training)':[],

                           'Adjusted R-squared (training)':[],

                           'R-squared (test)':[],

                           'Adjusted R-squared (test)':[],

                           '5-Fold Cross Validation':[]})
def adjustedR2(r2,n,k):

    return r2-(k-1)/(n-k)*(1-r2)
from sklearn.svm import LinearSVC

from sklearn.feature_selection import RFE

from sklearn import datasets



names = df[['OverallQual','GrLivArea', 'GarageCars','TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt','GarageArea','1stFlrSF']]



target=df[['SalePrice']]



svm = LinearSVC()

# create the RFE model for the svm classifier 

# and select attributes

rfe = RFE(svm, 3)

rfe = rfe.fit(names, target)

# print summaries for the selection of attributes

list(names)

print(rfe.support_)

print(rfe.ranking_)
from sklearn.svm import LinearSVC

from sklearn.feature_selection import RFE

from sklearn import datasets



names = df[['OverallQual','GrLivArea', 'GarageCars','TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt','GarageArea','1stFlrSF']]



target=df[['SalePrice']]



svm = LinearSVC()

# create the RFE model for the svm classifier 

# and select attributes

rfe = RFE(svm, 4)

rfe = rfe.fit(names, target)

# print summaries for the selection of attributes

list(names)

print(rfe.support_)

print(rfe.ranking_)
from sklearn.svm import LinearSVC

from sklearn.feature_selection import RFE

from sklearn import datasets



names = df[['OverallQual','GrLivArea', 'GarageCars','TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt','GarageArea','1stFlrSF']]



target=df[['SalePrice']]



svm = LinearSVC()

# create the RFE model for the svm classifier 

# and select attributes

rfe = RFE(svm, 5)

rfe = rfe.fit(names, target)

# print summaries for the selection of attributes

list(names)

print(rfe.support_)

print(rfe.ranking_)
#####################################  Simple Linear Regression - Lot Size v.s. Price  ##################################### 



import math

#Split data into Train and Test (60%/40%)

train_data,test_data = train_test_split(df,train_size = 0.6,random_state=3)



#Using Linear Regression Model and Train Model with Training Subset

lr = linear_model.LinearRegression()

X_train = np.array(train_data['LotArea'], dtype=pd.Series).reshape(-1,1)

y_train = np.array(train_data['SalePrice'], dtype=pd.Series)



#Fitting Model 1 to Training Data

lr.fit(X_train,y_train)



X_test = np.array(test_data['LotArea'], dtype=pd.Series).reshape(-1,1)

y_test = np.array(test_data['SalePrice'], dtype=pd.Series)



pred = lr.predict(X_test)

msesm = float(format(np.sqrt(metrics.mean_squared_error(y_test,pred)),'.3f'))

rmse=(math.sqrt(msesm))

rtrsm = float(format(lr.score(X_train, y_train),'.3f'))

rtesm = float(format(lr.score(X_test, y_test),'.3f'))

cv = float(format(cross_val_score(lr,df[['LotArea']],df['SalePrice'],cv=5).mean(),'.3f'))



print ("Average Price for Test Data: {:.3f}".format(y_test.mean()))

print('Intercept: {}'.format(lr.intercept_))

print('Coefficient: {}'.format(lr.coef_))



r = evaluation.shape[0]

evaluation.loc[r] = ['Simple Linear Regression - Lot Size v.s. Price','-',msesm,rmse,rtrsm,'-',rtesm,'-',cv]

evaluation
lr.fit(X_train,y_train)

lr.predict([[1000]])
lr.score(X_test,y_test)
sns.set(style="white", font_scale=1)



plt.figure(figsize=(10,9))



plt.scatter(X_train,y_train,color='purple',label="Data", alpha=.1)

plt.plot(X_train,lr.predict(X_train),color="red",label="Predicted Regression Line")



plt.title("Simple Regression House Price Predict Model 1 - Train Set",fontsize=20)

plt.xlabel("LotArea (sqft)", fontsize=15)

plt.ylabel("SalePrice ($)", fontsize=15)



plt.xticks(fontsize=13)

plt.yticks(fontsize=13)

plt.legend()



plt.gca().spines['right'].set_visible(False)

plt.gca().spines['top'].set_visible(False)
sns.set(style="white", font_scale=1)



plt.figure(figsize=(10,9))



plt.scatter(X_test,y_test,color='purple',label="Data", alpha=.1)

plt.plot(X_test,lr.predict(X_test),color="red",label="Predicted Regression Line")



plt.title("Simple Regression House Price Predict Model 1 - Test Set",fontsize=20)

plt.xlabel("LotArea (sqft)", fontsize=15)

plt.ylabel("SalePrice ($)", fontsize=15)



plt.xticks(fontsize=13)

plt.yticks(fontsize=13)

plt.legend()



plt.gca().spines['right'].set_visible(False)

plt.gca().spines['top'].set_visible(False)
#############################################  BedroomAbvGr v.s. Price ###############################################################



train_data_1,test_data_1 = train_test_split(df,train_size = 0.6,random_state=3)



lr = linear_model.LinearRegression()

X_train_1 = np.array(train_data_1['BedroomAbvGr'], dtype=pd.Series).reshape(-1,1)

y_train_1 = np.array(train_data_1['SalePrice'], dtype=pd.Series)

lr.fit(X_train_1,y_train_1)



X_test_1 = np.array(test_data_1['BedroomAbvGr'], dtype=pd.Series).reshape(-1,1)

y_test_1 = np.array(test_data_1['SalePrice'], dtype=pd.Series)



pred = lr.predict(X_test_1)

msesm = float(format(np.sqrt(metrics.mean_squared_error(y_test_1,pred)),'.3f'))

rmse=(math.sqrt(msesm))

rtrsm = float(format(lr.score(X_train_1, y_train_1),'.3f'))

rtesm = float(format(lr.score(X_test_1, y_test_1),'.3f'))

cv = float(format(cross_val_score(lr,df[['LotArea']],df['SalePrice'],cv=5).mean(),'.3f'))



print ("Average Price for Test Data: {:.3f}".format(y_test.mean()))

print('Intercept: {}'.format(lr.intercept_))

print('Coefficient: {}'.format(lr.coef_))



r = evaluation.shape[0]

evaluation.loc[r] = ['Simple Linear Regression - Bedroom Above Grade v.s. Price','-',msesm,rmse,rtrsm,'-',rtesm,'-',cv]

evaluation
sns.set(style="white", font_scale=1)



plt.figure(figsize=(10,9))



plt.scatter(X_train_1,y_train_1,color='darkgreen',label="Data", alpha=.1)

plt.plot(X_train_1,lr.predict(X_train_1),color="red",label="Predicted Regression Line")



plt.title("Simple Regression House Price Predict Model 2 - Train Set",fontsize=20)

plt.xlabel("BedroomAbvGr", fontsize=15)

plt.ylabel("SalePrice ($)", fontsize=15)



plt.xticks(fontsize=13)

plt.yticks(fontsize=13)

plt.legend()



plt.gca().spines['right'].set_visible(False)

plt.gca().spines['top'].set_visible(False)
sns.set(style="white", font_scale=1)



plt.figure(figsize=(10,6))



plt.scatter(X_test_1,y_test_1,color='darkgreen',label="Data", alpha=.1)

plt.plot(X_test_1,lr.predict(X_test_1),color="red",label="Predicted Regression Line")



plt.title("Simple Regression House Price Predict Model 2 - Test Set",fontsize=20)

plt.xlabel("BedroomAbvGr", fontsize=15)

plt.ylabel("SalePrice ($)", fontsize=15)



plt.xticks(fontsize=13)

plt.yticks(fontsize=13)

plt.legend()



plt.gca().spines['right'].set_visible(False)

plt.gca().spines['top'].set_visible(False)
lr.fit(X_train_1,y_train_1)

lr.predict([[6]])
#############################################  GrLivArea v.s. Price ###############################################################



train_data_2,test_data_2 = train_test_split(df,train_size = 0.6,random_state=3)



lr = linear_model.LinearRegression()

X_train_2 = np.array(train_data_2['GrLivArea'], dtype=pd.Series).reshape(-1,1)

y_train_2 = np.array(train_data_2['SalePrice'], dtype=pd.Series)

lr.fit(X_train_2,y_train_2)



X_test_2 = np.array(test_data_2['GrLivArea'], dtype=pd.Series).reshape(-1,1)

y_test_2= np.array(test_data_2['SalePrice'], dtype=pd.Series)



pred = lr.predict(X_test_2)

msesm = float(format(np.sqrt(metrics.mean_squared_error(y_test_2,pred)),'.3f'))

rmse=(math.sqrt(msesm))

rtrsm = float(format(lr.score(X_train_2, y_train_2),'.3f'))

rtesm = float(format(lr.score(X_test_2, y_test_2),'.3f'))

cv = float(format(cross_val_score(lr,df[['LotArea']],df['SalePrice'],cv=5).mean(),'.3f'))



print ("Average Price for Test Data: {:.3f}".format(y_test.mean()))

print('Intercept: {}'.format(lr.intercept_))

print('Coefficient: {}'.format(lr.coef_))



r = evaluation.shape[0]

evaluation.loc[r] = ['Simple Linear Regression - Total Living Room Area v.s. Price','-',msesm,rmse,rtrsm,'-',rtesm,'-',cv]

evaluation
lr.fit(X_train_2,y_train_2)

lr.predict([[1200]])
sns.set(style="white", font_scale=1)



plt.figure(figsize=(10,9))



plt.scatter(X_train_2,y_train_2,color='black',label="Data", alpha=.1)

plt.plot(X_train_2,lr.predict(X_train_2),color="red",label="Predicted Regression Line")



plt.title("Simple Regression House Price Predict Model 3 - Train Set",fontsize=20)

plt.xlabel("GrLivArea", fontsize=15)

plt.ylabel("SalePrice ($)", fontsize=15)



plt.xticks(fontsize=13)

plt.yticks(fontsize=13)

plt.legend()



plt.gca().spines['right'].set_visible(False)

plt.gca().spines['top'].set_visible(False)
sns.set(style="white", font_scale=1)



plt.figure(figsize=(10,6))



plt.scatter(X_test_2,y_test_2,color='black',label="Data", alpha=.1)

plt.plot(X_test_2,lr.predict(X_test_2),color="red",label="Predicted Regression Line")



plt.title("Simple Regression House Price Predict Model 3 - Test Set")

plt.xlabel("GrLivArea", fontsize=15)

plt.ylabel("SalePrice ($)", fontsize=15)



plt.xticks(fontsize=13)

plt.yticks(fontsize=13)

plt.legend()



plt.gca().spines['right'].set_visible(False)

plt.gca().spines['top'].set_visible(False)
df.info()
features=['MSSubClass',

 'LotFrontage',

 'LotArea',

 'OverallQual',

 'OverallCond',

 'YearBuilt',

 'YearRemodAdd',

 'MasVnrArea',

 'BsmtFinSF1',

 'BsmtFinSF2',

 'BsmtUnfSF',

 'TotalBsmtSF',

 '1stFlrSF',

 '2ndFlrSF',

 'LowQualFinSF',

 'GrLivArea',

 'BsmtFullBath',

 'BsmtHalfBath',

 'FullBath',

 'HalfBath',

 'BedroomAbvGr',

 'KitchenAbvGr',

 'TotRmsAbvGrd',

 'Fireplaces',

 'GarageYrBlt',

 'GarageCars',

 'GarageArea',

 'WoodDeckSF',

 'OpenPorchSF',

 'EnclosedPorch',

 '3SsnPorch',

 'ScreenPorch',

 'PoolArea',

 'MiscVal',

 'MoSold',

 'YrSold',

 'SalePrice']



print('this many columns:%d ' % len(df.columns))

df.columns
feature_cols = [col for col in df.columns if 'Price' not in col]
print (feature_cols)
X_numerical = pd.get_dummies(X)

X_numerical.head(5)
train_data_dm,test_data_dm = train_test_split(df,train_size = 0.6,random_state=3)



complex_model_1 = linear_model.LinearRegression()

complex_model_1.fit(train_data_dm[features],train_data_dm['SalePrice'])



print('Intercept: {}'.format(complex_model_1.intercept_))

print('Coefficients: {}'.format(complex_model_1.coef_))



pred = complex_model_1.predict(test_data_dm[features])

msecm = float(format(np.sqrt(metrics.mean_squared_error(test_data_dm['SalePrice'],pred)),'.3f'))

rmse=(math.sqrt(msecm))

rtrcm = float(format(complex_model_1.score(train_data_dm[features],train_data_dm['SalePrice']),'.3f'))

artrcm = float(format(adjustedR2(complex_model_1.score(train_data_dm[features],train_data_dm['SalePrice']),train_data_dm.shape[0],len(features)),'.3f'))

rtecm = float(format(complex_model_1.score(test_data_dm[features],test_data_dm['SalePrice']),'.3f'))

artecm = float(format(adjustedR2(complex_model_1.score(test_data_dm[features],test_data['SalePrice']),test_data_dm.shape[0],len(features)),'.3f'))

cv = float(format(cross_val_score(complex_model_1,df[features],df['SalePrice'],cv=5).mean(),'.3f'))



r = evaluation.shape[0]

evaluation.loc[r] = ['Multiple Regression 1 - All Features','All Features',msecm,rmse,rtrcm,artrcm,rtecm,artecm,cv]

evaluation.sort_values(by = '5-Fold Cross Validation', ascending=False)

############################################# Features 2 ##################################################



train_data_dm,test_data_dm = train_test_split(df,train_size = 0.6,random_state=3)



features_2=['YearBuilt','OverallQual','TotalBsmtSF','1stFlrSF','GrLivArea','FullBath','TotRmsAbvGrd','GarageCars','GarageArea']



complex_model_2 = linear_model.LinearRegression()

complex_model_2.fit(train_data_dm[features_2],train_data_dm['SalePrice'])



print('Intercept: {}'.format(complex_model_2.intercept_))

print('Coefficients: {}'.format(complex_model_2.coef_))

pred = complex_model_2.predict(test_data_dm[features_2])

msecm = float(format(np.sqrt(metrics.mean_squared_error(test_data_dm['SalePrice'],pred)),'.3f'))

rmse=(math.sqrt(msecm))

rtrcm = float(format(complex_model_2.score(train_data_dm[features_2],train_data_dm['SalePrice']),'.3f'))

artrcm = float(format(adjustedR2(complex_model_2.score(train_data_dm[features_2],train_data_dm['SalePrice']),train_data_dm.shape[0],len(features_2)),'.3f'))

rtecm = float(format(complex_model_2.score(test_data_dm[features_2],test_data_dm['SalePrice']),'.3f'))

artecm = float(format(adjustedR2(complex_model_2.score(test_data_dm[features_2],test_data['SalePrice']),test_data_dm.shape[0],len(features_2)),'.3f'))

cv = float(format(cross_val_score(complex_model_2,df[features_2],df['SalePrice'],cv=5).mean(),'.3f'))



r = evaluation.shape[0]

evaluation.loc[r] = ['Multiple Regression 2 - 9 Features v.s. House Price','Selected Features',msecm,rmse,rtrcm,artrcm,rtecm,artecm,cv]

evaluation.sort_values(by = '5-Fold Cross Validation', ascending=False)
features_2=['YearBuilt','OverallQual','TotalBsmtSF','1stFlrSF','GrLivArea','FullBath','TotRmsAbvGrd','GarageCars','GarageArea']
complex_model_2.predict([[1996,7,1000,1200,1200,1,4,1,10]])
############################################# Features 3 ##################################################



train_data_dm,test_data_dm = train_test_split(df,train_size = 0.6,random_state=3)



features_3=['YearBuilt','OverallQual','TotalBsmtSF','1stFlrSF','GrLivArea','FullBath','TotRmsAbvGrd']



complex_model_3 = linear_model.LinearRegression()

complex_model_3.fit(train_data_dm[features_3],train_data_dm['SalePrice'])



print('Intercept: {}'.format(complex_model_3.intercept_))

print('Coefficients: {}'.format(complex_model_3.coef_))



pred = complex_model_3.predict(test_data_dm[features_3])

msecm = float(format(np.sqrt(metrics.mean_squared_error(test_data_dm['SalePrice'],pred)),'.3f'))

rtrcm = float(format(complex_model_3.score(train_data_dm[features_3],train_data_dm['SalePrice']),'.3f'))

rmse=(math.sqrt(msecm))

artrcm = float(format(adjustedR2(complex_model_3.score(train_data_dm[features_3],train_data_dm['SalePrice']),train_data_dm.shape[0],len(features_3)),'.3f'))

rtecm = float(format(complex_model_3.score(test_data_dm[features_3],test_data_dm['SalePrice']),'.3f'))

artecm = float(format(adjustedR2(complex_model_3.score(test_data_dm[features_3],test_data['SalePrice']),test_data_dm.shape[0],len(features_3)),'.3f'))

cv = float(format(cross_val_score(complex_model_3,df[features_3],df['SalePrice'],cv=5).mean(),'.3f'))



r = evaluation.shape[0]

evaluation.loc[r] = ['Multiple Regression 3 - 7 Features','Selected Features',msecm,rmse,rtrcm,artrcm,rtecm,artecm,cv]

evaluation.sort_values(by = '5-Fold Cross Validation', ascending=False)
complex_model_3.predict([[1999,6,700,1200,1050,1.5,3]])
############################################# Features 4 ##################################################



train_data_dm,test_data_dm = train_test_split(df,train_size = 0.6,random_state=3)



features_4=['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces','EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea']



complex_model_4 = linear_model.LinearRegression()

complex_model_4.fit(train_data_dm[features_4],train_data_dm['SalePrice'])



print('Intercept: {}'.format(complex_model_4.intercept_))

print('Coefficients: {}'.format(complex_model_4.coef_))



pred = complex_model_4.predict(test_data_dm[features_4])

msecm = float(format(np.sqrt(metrics.mean_squared_error(test_data_dm['SalePrice'],pred)),'.3f'))

rmse=(math.sqrt(msecm))

rtrcm = float(format(complex_model_4.score(train_data_dm[features_4],train_data_dm['SalePrice']),'.3f'))

artrcm = float(format(adjustedR2(complex_model_4.score(train_data_dm[features_4],train_data_dm['SalePrice']),train_data_dm.shape[0],len(features_4)),'.3f'))

rtecm = float(format(complex_model_4.score(test_data_dm[features_4],test_data_dm['SalePrice']),'.3f'))

artecm = float(format(adjustedR2(complex_model_4.score(test_data_dm[features_4],test_data['SalePrice']),test_data_dm.shape[0],len(features_4)),'.3f'))

cv = float(format(cross_val_score(complex_model_4,df[features_4],df['SalePrice'],cv=5).mean(),'.3f'))



r = evaluation.shape[0]

evaluation.loc[r] = ['Multiple Regression 4 - 12 Features (House Structure) v.s. House Price','Selected Features',msecm,rmse,rtrcm,artrcm,rtecm,artecm,cv]

evaluation.sort_values(by = '5-Fold Cross Validation', ascending=False)
complex_model_4.predict([[2,1,1,2,1.5,2,0,1,0,0,0,1]])
############################################# Features 5 ##################################################



train_data_dm,test_data_dm = train_test_split(df,train_size = 0.6,random_state=3)



features_5=['YearBuilt','YearRemodAdd','GarageYrBlt']



complex_model_5 = linear_model.LinearRegression()

complex_model_5.fit(train_data_dm[features_5],train_data_dm['SalePrice'])



print('Intercept: {}'.format(complex_model_5.intercept_))

print('Coefficients: {}'.format(complex_model_5.coef_))



pred = complex_model_5.predict(test_data_dm[features_5])

msecm = float(format(np.sqrt(metrics.mean_squared_error(test_data_dm['SalePrice'],pred)),'.3f'))

rmse=(math.sqrt(msecm))

rtrcm = float(format(complex_model_5.score(train_data_dm[features_5],train_data_dm['SalePrice']),'.3f'))

artrcm = float(format(adjustedR2(complex_model_5.score(train_data_dm[features_5],train_data_dm['SalePrice']),train_data_dm.shape[0],len(features_5)),'.3f'))

rtecm = float(format(complex_model_5.score(test_data_dm[features_5],test_data_dm['SalePrice']),'.3f'))

artecm = float(format(adjustedR2(complex_model_5.score(test_data_dm[features_5],test_data['SalePrice']),test_data_dm.shape[0],len(features_5)),'.3f'))

cv = float(format(cross_val_score(complex_model_5,df[features_5],df['SalePrice'],cv=5).mean(),'.3f'))



r = evaluation.shape[0]

evaluation.loc[r] = ['Multiple Regression 5 - 3 Features (Building Years) v.s. House Price','Selected Features',msecm,rmse,rtrcm,artrcm,rtecm,artecm,cv]

evaluation.sort_values(by = '5-Fold Cross Validation', ascending=False)
complex_model_5.predict([[1925,1950,1977]])
print([features])
############################################# Features 6 ##################################################



train_data_dm,test_data_dm = train_test_split(df,train_size = 0.6,random_state=3)



features_6=['MSSubClass','OverallQual', 'OverallCond','TotalBsmtSF','GrLivArea','BedroomAbvGr','KitchenAbvGr', 'TotRmsAbvGrd','MiscVal']



complex_model_6 = linear_model.LinearRegression()

complex_model_6.fit(train_data_dm[features_6],train_data_dm['SalePrice'])



print('Intercept: {}'.format(complex_model_6.intercept_))

print('Coefficients: {}'.format(complex_model_6.coef_))



pred = complex_model_6.predict(test_data_dm[features_6])

msecm = float(format(np.sqrt(metrics.mean_squared_error(test_data_dm['SalePrice'],pred)),'.3f'))

rmse=(math.sqrt(msecm))

rtrcm = float(format(complex_model_6.score(train_data_dm[features_6],train_data_dm['SalePrice']),'.3f'))

artrcm = float(format(adjustedR2(complex_model_6.score(train_data_dm[features_6],train_data_dm['SalePrice']),train_data_dm.shape[0],len(features_6)),'.3f'))

rtecm = float(format(complex_model_6.score(test_data_dm[features_6],test_data_dm['SalePrice']),'.3f'))

artecm = float(format(adjustedR2(complex_model_6.score(test_data_dm[features_6],test_data['SalePrice']),test_data_dm.shape[0],len(features_6)),'.3f'))

cv = float(format(cross_val_score(complex_model_6,df[features_6],df['SalePrice'],cv=5).mean(),'.3f'))



r = evaluation.shape[0]

evaluation.loc[r] = ['Multiple Regression 6 - 9 Features (House Structure + Condition) v.s. House Price','Selected Features',msecm,rmse,rtrcm,artrcm,rtecm,artecm,cv]

evaluation.sort_values(by = '5-Fold Cross Validation', ascending=False)
############################################# Features 7 ##################################################



train_data_dm,test_data_dm = train_test_split(df,train_size = 0.6,random_state=3)



features_7=['MSSubClass','OverallQual', 'OverallCond','GrLivArea','BedroomAbvGr','KitchenAbvGr', 'TotRmsAbvGrd']



complex_model_7 = linear_model.LinearRegression()

complex_model_7.fit(train_data_dm[features_7],train_data_dm['SalePrice'])



print('Intercept: {}'.format(complex_model_7.intercept_))

print('Coefficients: {}'.format(complex_model_7.coef_))



pred = complex_model_7.predict(test_data_dm[features_7])

msecm = float(format(np.sqrt(metrics.mean_squared_error(test_data_dm['SalePrice'],pred)),'.3f'))

rmse=(math.sqrt(msesm))

rtrcm = float(format(complex_model_7.score(train_data_dm[features_7],train_data_dm['SalePrice']),'.3f'))

artrcm = float(format(adjustedR2(complex_model_7.score(train_data_dm[features_7],train_data_dm['SalePrice']),train_data_dm.shape[0],len(features_7)),'.3f'))

rtecm = float(format(complex_model_7.score(test_data_dm[features_7],test_data_dm['SalePrice']),'.3f'))

artecm = float(format(adjustedR2(complex_model_7.score(test_data_dm[features_7],test_data['SalePrice']),test_data_dm.shape[0],len(features_7)),'.3f'))

cv = float(format(cross_val_score(complex_model_7,df[features_7],df['SalePrice'],cv=5).mean(),'.3f'))



r = evaluation.shape[0]

evaluation.loc[r] = ['Multiple Regression 7 - 7 Features (House Structure + Condition) v.s. House Price','Selected Features',msecm,rmse,rtrcm,artrcm,rtecm,artecm,cv]

evaluation.sort_values(by = '5-Fold Cross Validation', ascending=False)
features_8=['MSSubClass',

 'LotFrontage',

 'LotArea',

 'OverallQual',

 'OverallCond',

 'YearBuilt',

 'YearRemodAdd',

 'MasVnrArea',

 'BsmtFinSF1',

 'BsmtFinSF2',

 'BsmtUnfSF',

 'TotalBsmtSF',

 '1stFlrSF',

 '2ndFlrSF',

 'LowQualFinSF',

 'GrLivArea',

 'BsmtFullBath',

 'BsmtHalfBath',

 'FullBath',

 'HalfBath',

 'BedroomAbvGr',

 'KitchenAbvGr',

 'TotRmsAbvGrd',

 'Fireplaces',

 'GarageYrBlt',

 'GarageCars',

 'GarageArea',

 'WoodDeckSF',

 'OpenPorchSF',

 'EnclosedPorch',

 '3SsnPorch',

 'ScreenPorch',

 'PoolArea',

 'MiscVal',

 'MoSold',

 'YrSold',

 'SalePrice']
knnreg = KNeighborsRegressor(n_neighbors=5)

knnreg.fit(train_data_dm[features_8],train_data_dm['SalePrice'])

pred = knnreg.predict(test_data_dm[features_8])



mseknn1 = float(format(np.sqrt(metrics.mean_squared_error(y_test,pred)),'.3f'))

rmse1=(math.sqrt(mseknn1))

rtrknn1 = float(format(knnreg.score(train_data_dm[features_8],train_data_dm['SalePrice']),'.3f'))

artrknn1 = float(format(adjustedR2(knnreg.score(train_data_dm[features_8],train_data_dm['SalePrice']),train_data_dm.shape[0],len(features_8)),'.3f'))

rteknn1 = float(format(knnreg.score(test_data_dm[features_8],test_data_dm['SalePrice']),'.3f'))

arteknn1 = float(format(adjustedR2(knnreg.score(test_data_dm[features_8],test_data_dm['SalePrice']),test_data_dm.shape[0],len(features_8)),'.3f'))

cv1 = float(format(cross_val_score(knnreg,df[features_8],df['SalePrice'],cv=5).mean(),'.3f'))



knnreg = KNeighborsRegressor(n_neighbors=11)

knnreg.fit(train_data_dm[features_8],train_data_dm['SalePrice'])

pred = knnreg.predict(test_data_dm[features_8])



mseknn2 = float(format(np.sqrt(metrics.mean_squared_error(y_test,pred)),'.3f'))

rmse2=(math.sqrt(mseknn2))

rtrknn2 = float(format(knnreg.score(train_data_dm[features_8],train_data_dm['SalePrice']),'.3f'))

artrknn2 = float(format(adjustedR2(knnreg.score(train_data_dm[features_8],train_data_dm['SalePrice']),train_data_dm.shape[0],len(features_8)),'.3f'))

rteknn2 = float(format(knnreg.score(test_data_dm[features_8],test_data_dm['SalePrice']),'.3f'))

arteknn2 = float(format(adjustedR2(knnreg.score(test_data_dm[features_8],test_data_dm['SalePrice']),test_data_dm.shape[0],len(features_8)),'.3f'))

cv2 = float(format(cross_val_score(knnreg,df[features_8],df['SalePrice'],cv=5).mean(),'.3f'))



knnreg = KNeighborsRegressor(n_neighbors=17)

knnreg.fit(train_data_dm[features_8],train_data_dm['SalePrice'])

pred = knnreg.predict(test_data_dm[features_8])



mseknn3 = float(format(np.sqrt(metrics.mean_squared_error(y_test,pred)),'.3f'))

rmse3=(math.sqrt(mseknn3))

rtrknn3 = float(format(knnreg.score(train_data_dm[features_8],train_data_dm['SalePrice']),'.3f'))

artrknn3 = float(format(adjustedR2(knnreg.score(train_data_dm[features_8],train_data_dm['SalePrice']),train_data_dm.shape[0],len(features_8)),'.3f'))

rteknn3 = float(format(knnreg.score(test_data_dm[features_8],test_data_dm['SalePrice']),'.3f'))

arteknn3 = float(format(adjustedR2(knnreg.score(test_data_dm[features_8],test_data_dm['SalePrice']),test_data_dm.shape[0],len(features_8)),'.3f'))

cv3 = float(format(cross_val_score(knnreg,df[features_8],df['SalePrice'],cv=5).mean(),'.3f'))



r = evaluation.shape[0]

evaluation.loc[r] = ['KNN Regression','k=5, all features',mseknn1,rmse1,rtrknn1,artrknn1,rteknn1,arteknn1,cv1]

evaluation.loc[r+1] = ['KNN Regression','k=11, all features',mseknn2,rmse2,rtrknn2,artrknn2,rteknn2,arteknn2,cv2]

evaluation.loc[r+2] = ['KNN Regression','k=17, all features',mseknn3,rmse3,rtrknn3,artrknn3,rteknn3,arteknn3,cv3]

evaluation.sort_values(by = '5-Fold Cross Validation', ascending=False)
knnreg = KNeighborsRegressor(n_neighbors=5)

knnreg.fit(train_data_dm[features_2],train_data_dm['SalePrice'])

pred = knnreg.predict(test_data_dm[features_2])



mseknn1 = float(format(np.sqrt(metrics.mean_squared_error(y_test,pred)),'.3f'))

rmse1=(math.sqrt(mseknn1))

rtrknn1 = float(format(knnreg.score(train_data_dm[features_2],train_data_dm['SalePrice']),'.3f'))

artrknn1 = float(format(adjustedR2(knnreg.score(train_data_dm[features_2],train_data_dm['SalePrice']),train_data_dm.shape[0],len(features_2)),'.3f'))

rteknn1 = float(format(knnreg.score(test_data_dm[features_2],test_data_dm['SalePrice']),'.3f'))

arteknn1 = float(format(adjustedR2(knnreg.score(test_data_dm[features_2],test_data_dm['SalePrice']),test_data_dm.shape[0],len(features_2)),'.3f'))

cv1 = float(format(cross_val_score(knnreg,df[features_2],df['SalePrice'],cv=5).mean(),'.3f'))



knnreg = KNeighborsRegressor(n_neighbors=11)

knnreg.fit(train_data_dm[features_2],train_data_dm['SalePrice'])

pred = knnreg.predict(test_data_dm[features_2])



mseknn2 = float(format(np.sqrt(metrics.mean_squared_error(y_test,pred)),'.3f'))

rmse2=(math.sqrt(mseknn2))

rtrknn2 = float(format(knnreg.score(train_data_dm[features_2],train_data_dm['SalePrice']),'.3f'))

artrknn2 = float(format(adjustedR2(knnreg.score(train_data_dm[features_2],train_data_dm['SalePrice']),train_data_dm.shape[0],len(features_2)),'.3f'))

rteknn2 = float(format(knnreg.score(test_data_dm[features_2],test_data_dm['SalePrice']),'.3f'))

arteknn2 = float(format(adjustedR2(knnreg.score(test_data_dm[features_2],test_data_dm['SalePrice']),test_data_dm.shape[0],len(features_2)),'.3f'))

cv2 = float(format(cross_val_score(knnreg,df[features_8],df['SalePrice'],cv=5).mean(),'.3f'))



knnreg = KNeighborsRegressor(n_neighbors=17)

knnreg.fit(train_data_dm[features_2],train_data_dm['SalePrice'])

pred = knnreg.predict(test_data_dm[features_2])



mseknn3 = float(format(np.sqrt(metrics.mean_squared_error(y_test,pred)),'.3f'))

rmse3=(math.sqrt(mseknn3))

rtrknn3 = float(format(knnreg.score(train_data_dm[features_2],train_data_dm['SalePrice']),'.3f'))

artrknn3 = float(format(adjustedR2(knnreg.score(train_data_dm[features_2],train_data_dm['SalePrice']),train_data_dm.shape[0],len(features_2)),'.3f'))

rteknn3 = float(format(knnreg.score(test_data_dm[features_2],test_data_dm['SalePrice']),'.3f'))

arteknn3 = float(format(adjustedR2(knnreg.score(test_data_dm[features_2],test_data_dm['SalePrice']),test_data_dm.shape[0],len(features_2)),'.3f'))

cv3 = float(format(cross_val_score(knnreg,df[features_2],df['SalePrice'],cv=5).mean(),'.3f'))



r = evaluation.shape[0]

evaluation.loc[r] = ['KNN Regression','k=5, selected features',mseknn1,rmse1,rtrknn1,artrknn1,rteknn1,arteknn1,cv1]

evaluation.loc[r+1] = ['KNN Regression','k=11, selected features',mseknn2,rmse2,rtrknn2,artrknn2,rteknn2,arteknn2,cv2]

evaluation.loc[r+2] = ['KNN Regression','k=17, selected features',mseknn3,rmse3,rtrknn3,artrknn3,rteknn3,arteknn3,cv3]

evaluation.sort_values(by = '5-Fold Cross Validation', ascending=False)
df_dm=df

train_data_dm,test_data_dm = train_test_split(df,train_size = 0.6,random_state=3)



features=['LotArea', 'MasVnrArea', 'BsmtFinSF1', 

             'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 

             '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 

             'GrLivArea', 'GarageArea', 'WoodDeckSF', 

             'OpenPorchSF', 'PoolArea', 'LotFrontage', 

             'MiscVal', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 

             'OverallQual', 'OverallCond', 'YearBuilt']





complex_model_R = linear_model.Ridge(alpha=1)

complex_model_R.fit(train_data_dm[features],train_data_dm['SalePrice'])



pred1 = complex_model_R.predict(test_data_dm[features])

msecm1 = float(format(np.sqrt(metrics.mean_squared_error(test_data_dm['SalePrice'],pred1)),'.3f'))

rtrcm1 = float(format(complex_model_R.score(train_data_dm[features],train_data_dm['SalePrice']),'.3f'))

artrcm1 = float(format(adjustedR2(complex_model_R.score(train_data_dm[features],train_data_dm['SalePrice']),train_data_dm.shape[0],len(features)),'.3f'))

rtecm1 = float(format(complex_model_R.score(test_data_dm[features],test_data_dm['SalePrice']),'.3f'))

artecm1 = float(format(adjustedR2(complex_model_R.score(test_data_dm[features],test_data_dm['SalePrice']),test_data_dm.shape[0],len(features)),'.3f'))

cv1 = float(format(cross_val_score(complex_model_R,df_dm[features],df_dm['SalePrice'],cv=5).mean(),'.3f'))



complex_model_R = linear_model.Ridge(alpha=100)

complex_model_R.fit(train_data_dm[features],train_data_dm['SalePrice'])



pred2 = complex_model_R.predict(test_data_dm[features])

msecm2 = float(format(np.sqrt(metrics.mean_squared_error(test_data_dm['SalePrice'],pred2)),'.3f'))

rtrcm2 = float(format(complex_model_R.score(train_data_dm[features],train_data_dm['SalePrice']),'.3f'))

artrcm2 = float(format(adjustedR2(complex_model_R.score(train_data_dm[features],train_data_dm['SalePrice']),train_data_dm.shape[0],len(features)),'.3f'))

rtecm2 = float(format(complex_model_R.score(test_data_dm[features],test_data_dm['SalePrice']),'.3f'))

artecm2 = float(format(adjustedR2(complex_model_R.score(test_data_dm[features],test_data_dm['SalePrice']),test_data_dm.shape[0],len(features)),'.3f'))

cv2 = float(format(cross_val_score(complex_model_R,df_dm[features],df_dm['SalePrice'],cv=5).mean(),'.3f'))



complex_model_R = linear_model.Ridge(alpha=1000)

complex_model_R.fit(train_data_dm[features],train_data_dm['SalePrice'])



pred3 = complex_model_R.predict(test_data_dm[features])

msecm3 = float(format(np.sqrt(metrics.mean_squared_error(test_data_dm['SalePrice'],pred3)),'.3f'))

rtrcm3 = float(format(complex_model_R.score(train_data_dm[features],train_data_dm['SalePrice']),'.3f'))

artrcm3 = float(format(adjustedR2(complex_model_R.score(train_data_dm[features],train_data_dm['SalePrice']),train_data_dm.shape[0],len(features)),'.3f'))

rtecm3 = float(format(complex_model_R.score(test_data_dm[features],test_data_dm['SalePrice']),'.3f'))

artecm3 = float(format(adjustedR2(complex_model_R.score(test_data_dm[features],test_data_dm['SalePrice']),test_data_dm.shape[0],len(features)),'.3f'))

cv3 = float(format(cross_val_score(complex_model_R,df_dm[features],df_dm['SalePrice'],cv=5).mean(),'.3f'))



r = evaluation.shape[0]

evaluation.loc[r] = ['Ridge Regression','alpha=1, all features',msecm1,'-',rtrcm1,artrcm1,rtecm1,artecm1,cv1]

evaluation.loc[r+1] = ['Ridge Regression','alpha=100, all features',msecm2,'-',rtrcm2,artrcm2,rtecm2,artecm2,cv2]

evaluation.loc[r+2] = ['Ridge Regression','alpha=1000, all features',msecm3,'-',rtrcm3,artrcm3,rtecm3,artecm3,cv3]

evaluation.sort_values(by = '5-Fold Cross Validation', ascending=False)
from sklearn.cluster import KMeans



temp = df.select_dtypes(include='object')

dumb = pd.get_dummies(temp)

df2 = df.select_dtypes(exclude='object')

df3 = pd.concat([df2, dumb], axis=1, sort=False)



# df3.LotFrontage = df3.LotFrontage.astype(int)

df4 = df3.fillna(0).astype(int)

df_tr = df4



clmns = list(df4.columns.values)



ks = range(1, 20)

inertias = []

sse = {}



for k in ks:

    # Create a KMeans instance with k clusters: model

    model = KMeans(n_clusters=k)

    

    # Fit model to samples

    model.fit(df4)

    

    # Append the inertia to the list of inertias

    inertias.append(model.inertia_)

    

    #for plot... please work

    sse[k] = model.inertia_

    



plt.figure()

plt.plot(list(sse.keys()), list(sse.values()))

plt.xlabel("Number of clusters")

plt.ylabel("SSE")

plt.show()
model1 = KMeans(n_clusters=4, random_state=0)

df_tr_std = stats.zscore(df_tr[clmns])
model1.fit(df_tr_std)

labels = model1.labels_
df_tr['clusters'] = labels

clmns.extend(['clusters'])
x1 = df_tr[["YearBuilt","OverallQual","TotalBsmtSF","1stFlrSF","GrLivArea","FullBath","TotRmsAbvGrd","clusters"]]

pd.options.display.max_columns = None



print(x1.groupby(['clusters']).mean())
x2 = df_tr[["YearBuilt","OverallQual","TotalBsmtSF","1stFlrSF","GrLivArea","FullBath","TotRmsAbvGrd","clusters"]]

pd.options.display.max_columns = None

print(x1.groupby(['clusters']).std())