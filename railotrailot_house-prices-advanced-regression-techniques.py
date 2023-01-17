#import the packages

import numpy as np

import pandas as pd

import seaborn as sns

from scipy import stats

from scipy.stats import skew

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax

import matplotlib.style as style

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn import preprocessing 

%matplotlib inline
#load the data

data_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv', sep=',')

data_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv', sep=',')



print (f"The train data has the empty values {data_train.isnull().values.sum()} of total {len(data_train.index)} rows")

print (f"The test data has the empty values {data_test.isnull().values.sum()} of total {len(data_test.index)} rows")
def check_dublicates (df):

    unic = len(set(df.Id))

    total = df.shape[0]

    is_dublicates = unic - total

    if is_dublicates:

        return 'There are {is_dublicates} dublicate records'
check_dublicates(data_train)

check_dublicates(data_test)

data_test['SalePrice'] = np.zeros((len(data_test.index),1))

joined_data = pd.concat((data_train, data_test)).reset_index(drop=True)

print (f"The joined data has the empty values {joined_data.isnull().values.sum()} of total {len(joined_data.index)} rows")
joined_data.head(5)
missing_report = joined_data[joined_data.columns[joined_data.isnull().sum()!=0]]

(missing_report.isnull().sum() / missing_report.shape[0]).sort_values(ascending=False)
joined_data = joined_data.drop(['PoolQC', 'MiscFeature', 'Alley'], axis=1)
#[val for val in categorical if val in checking_categorical_data (joined_data )]

import operator

res ={col: joined_data[ joined_data[col]==0 ][col].count() / joined_data[col].shape[0] 

     for col in joined_data if joined_data[ joined_data[col]==0 ][col].count() / joined_data[col].shape[0] > 0.9}

dict(sorted(res.items(), key=operator.itemgetter(1),reverse=True))
joined_data = joined_data.drop([ 'PoolArea', '3SsnPorch', 'LowQualFinSF', 'MiscVal', 'BsmtHalfBath', 'ScreenPorch' ], axis=1)
nominal_var = [

    'MSSubClass','MSZoning','Street',

    'LotShape','LandContour','Utilities',

    'LotConfig','LandSlope','Neighborhood',

    'Condition1','Condition2','BldgType',

    'HouseStyle','RoofStyle','RoofMatl',

    'Exterior1st','Exterior2nd','MasVnrType',

    'Foundation','Heating','CentralAir',

    'BsmtFullBath','FullBath','HalfBath',

    'TotRmsAbvGrd','Functional','Fireplaces',

    'PavedDrive','MoSold','SaleType',

    'SaleCondition','GarageCars',

]

ordinal_var = [

    'ExterQual','ExterCond','BsmtQual','BsmtCond',

    'BsmtExposure','BsmtFinType2','HeatingQC','Electrical',

    'FireplaceQu','GarageType','GarageFinish','GarageQual',

    'GarageCond','Fence','YrSold','YearBuilt','YearRemodAdd',

    'GarageYrBlt','BsmtFinType1', 'BedroomAbvGr', 'KitchenQual',

    'KitchenAbvGr'

]

numeric_var=[

    'LotFrontage','LotArea','OverallQual','OverallCond',

    'MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF',

    'TotalBsmtSF','1stFlrSF','2ndFlrSF','GrLivArea',

    'GarageArea','WoodDeckSF','OpenPorchSF',

    'EnclosedPorch'

]
def checking_categorical_data ( dataframe ):

    list_ord = []

    for feature in dataframe.columns:

        if 1.*dataframe[feature].nunique()/dataframe[feature].count() < 0.05:

            list_ord.append( feature )

    return list_ord



categorical = nominal_var + ordinal_var

res = checking_categorical_data (joined_data )

#list(set(categorical) - set(res)), list(set(joined_data.columns)-set(categorical+numeric_var))
## Replaced all missing values in LotFrontage by imputing the median value of each neighborhood. 

#joined_data['LotFrontage'] = joined_data.groupby('Neighborhood')['LotFrontage'].transform( lambda x: x.fillna(x.mean()))    

#joined_data['MSSubClass'] = joined_data['MSSubClass'].astype(str)

#joined_data['MSZoning'] = joined_data.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))    

#joined_data['YrSold'] = joined_data['YrSold'].astype(str)

#joined_data['MoSold'] = joined_data['MoSold'].astype(str) 

#joined_data['Functional'] = joined_data['Functional'].fillna('Typ') 

#joined_data['Utilities'] = joined_data['Utilities'].fillna('AllPub') 

#joined_data['Exterior1st'] = joined_data['Exterior1st'].fillna(joined_data['Exterior1st'].mode()[0]) 

#joined_data['Exterior2nd'] = joined_data['Exterior2nd'].fillna(joined_data['Exterior2nd'].mode()[0])

#joined_data['KitchenQual'] = joined_data['KitchenQual'].fillna("TA") 

#joined_data['SaleType'] = joined_data['SaleType'].fillna(joined_data['SaleType'].mode()[0])

#joined_data['Electrical'] = joined_data['Electrical'].fillna("SBrkr")     



for var in ordinal_var:

    joined_data[var] = joined_data[var].fillna('NA')

for var in numeric_var:

    joined_data[var] = joined_data[var].fillna(0)

#for var in numeric_var:

#    joined_data[var]=joined_data[var].apply(lambda x: x if x else joined_data[var].mean())

for var in nominal_var:

    joined_data[var] = joined_data[var].fillna('NA')



before_encoding_joined_data = joined_data.copy()

#Check remaining missing values if any 

print(joined_data.isnull().values.sum())
def one_hot_encoding ( dataframe, feature_to_encode ):

    for cat in feature_to_encode:

        dummies = pd.get_dummies( dataframe[ [ cat ] ] )

        dataframe = pd.concat( [ dataframe, dummies ], axis=1 )

        dataframe.drop( cat, axis=1, inplace=True )

    return dataframe 



def labeled_encoding ( dataframe, feature_to_encode ):

    for col in feature_to_encode:

        labelencoder = LabelEncoder()

        dataframe[col+'_Cat'] = labelencoder.fit_transform( dataframe[col].astype(str) )

        dataframe.drop( col, axis=1, inplace=True )

    return dataframe



def normalization_columns ( dataframe, feature_to_encode ):

    for col in feature_to_encode:

        min_max_scaler = preprocessing.MinMaxScaler()

        x = dataframe[[col]].values.astype(float)

        x_scaled = min_max_scaler.fit_transform(x)

        dataframe[col] = pd.DataFrame(x_scaled)

    return dataframe    



joined_data = one_hot_encoding( joined_data, nominal_var ) #one_hot_encoding

joined_data = one_hot_encoding( joined_data, ordinal_var )

#joined_data = normalization_columns ( joined_data, numeric_var )

joined_data.head()
print (f"After handling the joined_data has {len(joined_data.columns)} column of total {len(joined_data.index)} rows")
def show_scatter( otput_col, input_var, df ):

    for col in input_var:

        plt.subplots(figsize = (12,8))

        res = df[df[otput_col] > 0]

        sns.scatterplot(res[otput_col], res[col])

        

def show_resid( otput_col, input_var, df ):

    for col in input_var:

        plt.subplots(figsize = (12,8))

        res = df[df[otput_col] > 0]

        sns.residplot(res[col], res[otput_col])  

        

show_scatter('SalePrice', numeric_var, joined_data)
outlines = {

    'LotFrontage':0.9, 

    'LotArea':0.7, 

    'BsmtFinSF1':0.7, 

    'TotalBsmtSF':0.7, 

    '1stFlrSF':0.7, 

    'GrLivArea':1, 

    'WoodDeckSF':0.9, 

    'OpenPorchSF':0.9, 

    'EnclosedPorch' : 0.9

}



joined_data.reset_index(drop = True, inplace = True)

    

#for key, value in outlines.items():

#    joined_data[key] = joined_data[key].apply( lambda x: x if x < value else joined_data[key].median() )

show_scatter('SalePrice', numeric_var, joined_data)
show_resid('SalePrice', numeric_var, joined_data)
plt.figure(figsize = (24,12))

plt.subplot(121)

sns.distplot(joined_data[joined_data['SalePrice']>0]['SalePrice'])

plt.subplot(122)

stats.probplot(joined_data[joined_data['SalePrice']>0]['SalePrice'],plot=plt)
plt.figure(figsize = (24,12))

plt.subplot(121)

sns.distplot( np.log(joined_data[joined_data['SalePrice']>0]['SalePrice']) )

plt.subplot(122)

stats.probplot( np.log(joined_data[joined_data['SalePrice']>0]['SalePrice']), plot=plt )
plt.figure(figsize = (24,12))

plt.subplot(121)

sns.distplot( (joined_data[joined_data['SalePrice']>0]['SalePrice'])**0.5 )

plt.subplot(122)

stats.probplot( (joined_data[joined_data['SalePrice']>0]['SalePrice'])**0.5, plot=plt )
joined_data['SalePrice'] = np.log(joined_data[joined_data['SalePrice']>0]['SalePrice']) 

show_resid('SalePrice', numeric_var, joined_data)
skewed_feats = joined_data[numeric_var].apply(lambda x: skew(x)).sort_values(ascending=False)

print("\nSkew in numerical parameters: \n")

skewness = pd.DataFrame({'Skew' : skewed_feats})

skewness.head(10)
skewness = skewness[abs(skewness) > 0.75]

skewed_features = skewness.index

lam = 0.15

before_skewed = joined_data.copy()

for col in numeric_var:

    joined_data[col] = boxcox1p(joined_data[col], boxcox_normmax(joined_data[col] + 1))
numeric_var=[

    'LotFrontage','LotArea','OverallQual','OverallCond',

    'MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF',

    'TotalBsmtSF','1stFlrSF','2ndFlrSF','GrLivArea',

    'GarageArea','WoodDeckSF','OpenPorchSF',

    'EnclosedPorch'

]

#joined_data['OverallQual'] = np.log1p(joined_data['OverallQual'])

#joined_data['OverallCond'] = np.log1p(joined_data['OverallCond'])

#joined_data['BsmtUnfSF'] = np.log1p(joined_data['BsmtUnfSF'])

#joined_data['GrLivArea'] = np.log1p(joined_data['GrLivArea'])

#joined_data['GarageArea'] = np.log1p(joined_data['GarageArea'])

#joined_data['WoodDeckSF'] = np.log1p(joined_data['WoodDeckSF'])

#joined_data['OpenPorchSF'] = np.log1p(joined_data['OpenPorchSF'])

#joined_data['EnclosedPorch'] = np.log1p(joined_data['EnclosedPorch'])

#joined_data['KitchenAbvGr'] = np.log1p(joined_data['KitchenAbvGr'])

#joined_data = joined_data.drop([ 'KitchenAbvGr' ], axis=1)

#joined_data[var].apply(lambda x: x if x else joined_data[var].mean())

#joined_data['MasVnrArea'] = joined_data['MasVnrArea'].apply(lambda x: x if x else joined_data['MasVnrArea'].mean())

#print(joined_data[joined_data['MasVnrArea'] == 0]['MasVnrArea'].sum())

#sns.distplot(joined_data['KitchenAbvGr'], kde_kws={"label":"before transformation"})

#sns.distplot(np.log1p(joined_data['KitchenAbvGr']), kde_kws={"label":"after transformation"})

#sns.distplot(joined_data['BsmtFinSF2'])
#Best heatmap, the code below - https://www.kaggle.com/masumrumi/a-detailed-regression-guide-with-house-pricing

## Plot fig sizing. 

style.use('ggplot')

sns.set_style('whitegrid')

plt.subplots(figsize = (30,20))

## Plotting heatmap. 



# Generate a mask for the upper triangle (taken from seaborn example gallery)

mask = np.zeros_like(before_encoding_joined_data.corr(), dtype=np.bool)

mask[np.triu_indices_from(mask)] = True





sns.heatmap(before_encoding_joined_data.corr(), 

            cmap=sns.diverging_palette(20, 220, n=200), 

            mask = mask, 

            annot=True, 

            center = 0, 

           );

## Give title. 

plt.title("Heatmap of all the Features", fontsize = 30);
joined_data['is_garage'] = joined_data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

joined_data['is_bsmt'] = joined_data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

joined_data['is_second_floor'] = joined_data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

from sklearn import ensemble

from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVR

from sklearn import linear_model

from sklearn.linear_model import LinearRegression, RidgeCV, Ridge, Lasso, LassoCV, ElasticNet

from sklearn.model_selection import cross_val_score, KFold

from sklearn.metrics import accuracy_score

from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.kernel_ridge import KernelRidge

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone



train = joined_data[joined_data['SalePrice']>0]

y = train['SalePrice'].values.tolist()

y_tr = train['SalePrice']

train = train.drop('SalePrice', axis=1 )

X = train.values.tolist()



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)







reg_rgl = np.linspace(-0.1, 3, num=100) 

reg_lso = np.linspace(-5, 5, num=1000) 

reg_els = np.linspace(-5, 5, num=1000)

reg_lin = np.linspace(-5, 5, num=100)



score_ridge   = {}

score_lasso   = {}

score_elastic = {}

score_linear  = {}



def get_key(d, value):

    for k, v in d.items():

        if v == value:

            return k





for i in reg_lin:

    linear = LinearRegression()

    linear.fit(X_train, y_train)

    score_linear[i] = mean_squared_error(y_test, linear.predict(X_test))



for i in reg_rgl:

    ridge = Ridge(alpha= i, normalize=True)

    ridge.fit(X_train, y_train)

    score_ridge[i] = mean_squared_error(y_test, ridge.predict(X_test))



for i in reg_lso:

    lasso_reg = Lasso(alpha= i, normalize=True)

    lasso_reg.fit(X_train, y_train)

    score_lasso[i] = mean_squared_error(y_test, lasso_reg.predict(X_test))



for i in reg_els:

    elastic_net = ElasticNet(alpha= i, normalize=True)

    elastic_net.fit(X_train, y_train)

    score_elastic[i] = mean_squared_error(y_test, elastic_net.predict(X_test))    

    

BayRidge = linear_model.BayesianRidge(normalize=True)

BayRidge.fit(X_train, y_train)

BayRidge_res = mean_squared_error(y_test, BayRidge.predict(X_test))

    

svr = SVR(C = 20, epsilon= 0.008, gamma=0.0003,)

svr.fit(X_train, y_train)

SVR_res = mean_squared_error(y_test, y_pred = svr.predict(X_test)) 



KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=1.5)

KRR.fit(X_train, y_train)

KRR_res = mean_squared_error(y_test, y_pred = KRR.predict(X_test)) 



print("Best results for:")

print( f"KernelRidge - loss {KRR_res}")

print( f"Ridge - loss {min(score_ridge.values())}, koeff -{  get_key(score_ridge, min(score_ridge.values())) } ")

print( f"Lasso - loss {min(score_lasso.values())}, koeff -{  get_key(score_lasso, min(score_lasso.values())) } ")

print( f"Elastic - loss {min(score_elastic.values())}, koeff -{  get_key(score_elastic, min(score_elastic.values())) } ")

print( f"svr - loss {SVR_res}")

print( f"BayRidge - loss {KRR_res}")



#print( f"Best results for linear - {min(score_lasso.values())}" )

#print( f"Best results for KernelRidge - {KRR_res} linear - {min(score_linear.values())}, \n ridge - {min(score_ridge.values())}, \n lasso - {min(score_lasso.values())}, \n elastic -  {min(score_elastic.values())} \n bayes_reg -  {BayRidge_res}\n SVR - {SVR_res}")
sub = pd.DataFrame()

test = joined_data[joined_data['SalePrice'].isnull()]

test = test.drop( 'SalePrice', axis=1 )

sub['Id'] = test['Id']

sub["SalePrice"] =  np.exp(lasso_reg.predict(test.values.tolist()))

sub.to_csv('submission.csv',index=False)
