import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from scipy.stats import norm

import os #The OS module in Python provides a way of using operating system dependent functionality

import warnings

warnings.filterwarnings('ignore')

print(os.listdir("../input"))
# Loading  test and train datasets

df_train=pd.read_csv('../input/train.csv')

df_test=pd.read_csv('../input/test.csv')

df_train.head()
# Separate the id columns from both test and train data as id columns

#is needed for submission and also id column does not make any contribution to prediciton



id_train= df_train['Id']

Id_test = df_test['Id']



df_train.drop("Id", axis = 1, inplace = True)

df_test.drop("Id", axis = 1, inplace = True)
# Function to print the basic information of the data

def data_info(df):



    print('Shape of the data: ', df.shape)

    

    print('------------########################------------------')

    print('                                                     ')

    print('Information of the data:')

    print(' ', df.info())

    

    print('------------########################------------------')

    print('                                                     ')

    print('Check the duplication of the data:', df.duplicated().sum())
data_info(df_train)
# Function to find out the Statistical susmmary 

def summary(df):

    print('\n Statistical Summary of Numberical data:\n', df.describe(include=np.number))

    print('------------########################------------------')

    print('\n Statistical Summary of categorical data:\n',df.describe(include='O'))

    

summary(df_train)
# Boxplot for target

plt.figure(figsize=(12,8))

sns.boxplot(df_train['SalePrice'])
# Remove outliers from target variables

df_train=df_train[df_train['SalePrice']<700000]

df_train.head()
# Distribution plot

plt.figure(figsize=(12,8))

sns.distplot(df_train['SalePrice'])
# Distribution plot

plt.figure(figsize=(12,8))

sns.distplot(df_train['SalePrice'] , fit=norm);



# Probability parameter

(mu, sigma) = norm.fit(df_train['SalePrice'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')
#Log tranformation of target column

plt.figure(figsize=(12,8))

df_train["SalePrice"] = np.log1p(df_train["SalePrice"])



#Plot the new distriution

sns.distplot(df_train['SalePrice'] , fit=norm);



# probability parameter for normal distribution

(mu, sigma) = norm.fit(df_train['SalePrice'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)])

plt.ylabel('Frequency')

plt.title('SalePrice distribution')

# Outliers Check

def outlier(df):

    stat=df.describe()

    IQR=stat['75%']-stat['25%']

    upper=stat['75%']+1.5*IQR

    lower=stat['25%']-1.5*IQR

    print('The upper and lower bounds for outliers are {} and {}'.format(upper,lower))
outlier(df_train['SalePrice'])
# Let's separate the numerical and categorical columns

numerical_col=df_train.select_dtypes(include=[np.number])

categorical_col=df_train.select_dtypes(include=[np.object])

num_var=numerical_col.columns.tolist()

cat_var=categorical_col.columns.tolist()
# Function to plot target vs categorical data

def cat_plot(df):

    for col in cat_var:

        f, ax = plt.subplots(figsize=(12, 6))

        sns.boxplot(x=col,y='SalePrice', data=df)

        plt.xlabel(col)

        plt.title('{}'.format(col))



cat_plot(df_train)
# Function to plot target vs numerical data

def num_plot(df):

    for col in num_var:

        f, ax = plt.subplots(figsize=(12, 6))

        plt.scatter(x=col,y='SalePrice', data=df)

        plt.xlabel(col)

        plt.ylabel("SalePrice")

        plt.title('{}'.format(col))
num_plot(df_train)
# Removing suspicious outliers

df_train = df_train.drop(df_train[(df_train['GrLivArea']>4000) & (df_train['SalePrice']<300000)].index).reset_index(drop=True)

df_train=df_train.drop(df_train[(df_train['LotFrontage']>250) & (df_train['SalePrice']<300000)].index).reset_index(drop=True)

df_train=df_train.drop(df_train[(df_train['BsmtFinSF1']>1400) & (df_train['SalePrice']<400000)].index).reset_index(drop=True)

df_train=df_train.drop(df_train[(df_train['TotalBsmtSF']>5000) & (df_train['SalePrice']<300000)].index).reset_index(drop=True)

df_train=df_train.drop(df_train[(df_train['1stFlrSF']>4000) & (df_train['SalePrice']<300000)].index).reset_index(drop=True)
#Categorical variables after removing the outliers

new_cat=['GrLivArea','LotFrontage','BsmtFinSF1','TotalBsmtSF','1stFlrSF']
# Plotting after removing outliers

for col in new_cat:

        f, ax = plt.subplots(figsize=(12, 6))

        plt.scatter(x=col,y='SalePrice', data=df_train)
# merging the data

train_len = len(df_train) # created length of the train data so that after EDA is done we can seperate the train and test data

data= pd.concat(objs=[df_train, df_test], axis=0).reset_index(drop=True)

data.head()
# A function for calculating the missing data

def missing_data(df):

    tot_missing=df.isnull().sum().sort_values(ascending=False)

    Percentage=tot_missing/len(df)*100

    missing_data=pd.DataFrame({'Missing Percentage': Percentage})

    

    return missing_data.head(36)



missing_data(data)
# missing value in test dataset

missing_data(df_test)
# Features with missing value

miss_col1=['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageCond', 'GarageType', 'GarageYrBlt',

           'GarageFinish', 'GarageQual', 'BsmtExposure','BsmtFinType2', 'BsmtFinType1', 'BsmtCond', 'BsmtQual', 

           'MasVnrArea', 'MasVnrType','SaleType','MSZoning','Utilities','Functional','Exterior1st','Exterior2nd',

           'BsmtFinSF1','BsmtFinSF2','TotalBsmtSF','GarageArea','KitchenQual','GarageCars','BsmtFullBath',

           'BsmtHalfBath','BsmtUnfSF']

# Imputing missing value

for col in miss_col1:

    if data[col].dtype=='O':

        data[col]=data[col].fillna("None")

    else:

        data[col]=data[col].fillna(0)
# Imputing missing value with neighborhood value

data['LotFrontage']=data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
# Imputing missing value with mode

data['Electrical']=data['Electrical'].fillna(data['Electrical'].mode()[0])
missing_data(data)
corr= data.corr()

f, ax = plt.subplots(figsize=(16, 10))

sns.heatmap(corr, vmax=.6, square=True)
k = 20 #number of variables for heatmap

cols = corr.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(data[cols].values.T)

f , ax = plt.subplots(figsize = (14,12))

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True,linewidths=0.004, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
# Total area in units of square feet

data['TotSF']=data['TotalBsmtSF']+data['1stFlrSF']+data['2ndFlrSF']

data['TotArea']=data['GarageArea']+data['GrLivArea']
plt.scatter(x='TotArea',y='SalePrice', data=data)
cols=['MSSubClass','OverallCond','YrSold','MoSold']



for col in cols:

    data[col] = data[col].apply(str)
categorical_col=data.select_dtypes(include=[np.object])

new_catcol=categorical_col.columns

new_catcol
ordinal_cat=['OverallCond','KitchenQual','YrSold','MoSold','Fence','PoolQC','FireplaceQu','GarageQual', 

             'GarageCond','LotShape','LandSlope','HouseStyle','ExterQual','ExterCond','BsmtQual', 

             'BsmtCond','BsmtExposure','BsmtFinType1', 'BsmtFinType2','HeatingQC','KitchenQual','CentralAir',

             'MSSubClass']



# label Encoding for ordinal data

from sklearn.preprocessing import LabelEncoder

label_encode=LabelEncoder()



for col in ordinal_cat:

    data[col]=label_encode.fit_transform(data[col])
data.select_dtypes(include=[np.object]).head()
# One hot encoding for nominal data

data=pd.get_dummies(data)
df_target=data['SalePrice']

df_features=data.drop(columns=['SalePrice'])
X_train=df_features[:train_len]

Y_train=df_target[:train_len]

X_test=df_features[train_len:]
# Import MLlibraries

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import cross_val_score, KFold

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.linear_model import LinearRegression

from xgboost import XGBRegressor

from sklearn.utils import shuffle



from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

import lightgbm as lgb

from sklearn.linear_model import ElasticNet, Lasso, Ridge, BayesianRidge, LassoLarsIC
# Function to calculate RMSE

n_folds = 5

def rmse_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X_train)

    rmse= np.sqrt(-cross_val_score(model, X_train, Y_train, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)
lasso=Lasso()

rmse_cv(lasso).mean()
from sklearn.model_selection import GridSearchCV

params = {'alpha': [0.0001,0.0002,0.0003,0.0004,0.0005,0.0006]}

grid_search_cv = GridSearchCV(Lasso(random_state=42), params, n_jobs=-1)

grid_search_cv.fit(X_train, Y_train)



print(grid_search_cv.best_estimator_)

print(grid_search_cv.best_score_)
Random=RandomForestRegressor()

rmse_cv(Random).mean()
params = {'n_estimators': list(range(50, 200, 25)), 'max_features': ['auto', 'sqrt', 'log2'], 

         'min_samples_leaf': list(range(50, 200, 50))}



grid_search_cv = GridSearchCV(RandomForestRegressor(random_state=42), params, n_jobs=-1)

grid_search_cv.fit(X_train, Y_train)



print(grid_search_cv.best_estimator_)

print(grid_search_cv.best_score_)
Enet=ElasticNet()

rmse_cv(Enet).mean()
params = {'alpha': [0.0001,0.0002,0.0003,0.0004,0.0005,0.0006]}



grid_search_cv = GridSearchCV(ElasticNet(random_state=42), params, n_jobs=-1)

grid_search_cv.fit(X_train, Y_train)



print(grid_search_cv.best_estimator_)

print(grid_search_cv.best_score_)
KR=KernelRidge()

rmse_cv(KR).mean()
params = {'alpha': [0.0001,0.0002,0.0003,0.0004,0.0005,0.0006]}



grid_search_cv = GridSearchCV(KernelRidge(), params, n_jobs=-1)

grid_search_cv.fit(X_train, Y_train)



print(grid_search_cv.best_estimator_)

print(grid_search_cv.best_score_)
GBoost = GradientBoostingRegressor()

rmse_cv(GBoost).mean()
params = {'n_estimators': [1000,2000,3000,4000,5000,6000]}



grid_search_cv = GridSearchCV(GradientBoostingRegressor(), params, n_jobs=-1)

grid_search_cv.fit(X_train, Y_train)



print(grid_search_cv.best_estimator_)

print(grid_search_cv.best_score_)
# Models  with best paramenters

lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0004, random_state=42))

ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0006, l1_ratio=.5, random_state=42))

KRR = KernelRidge(alpha=0.0001, kernel='linear', degree=3, coef0=1.0)

GBoost = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.1,

                                   max_depth=3, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5)



RanForest=RandomForestRegressor(min_samples_leaf=50, min_samples_split=2, 

                                n_estimators=150, random_state=42)
Ml_models=[RanForest,lasso,ENet,KRR,GBoost]

def rmse_score(models):

    for model in models:

        print("RMSE and STD for {} are {:4f} and  {:4f} respectively.".format(model,rmse_cv(model).mean(),rmse_cv(model).std()))

        #print("RMSE and STD for lasso are {:4f} and  {:4f} respectively.".format(rmse_cv(lasso).mean(),rmse_cv(lasso).std()))
rmse_score(Ml_models)
LassoFit= lasso.fit(X_train,Y_train)

ENetFit = ENet.fit(X_train,Y_train)

KRRFit = KRR.fit(X_train,Y_train)

GBoostFit = GBoost.fit(X_train,Y_train)

RanForestFit=RanForest.fit(X_train,Y_train)
Final_score= (np.expm1(LassoFit.predict(X_test)) + 

              np.expm1(ENetFit.predict(X_test)) + np.expm1(KRRFit.predict(X_test)) 

              + np.expm1(GBoostFit.predict(X_test))+ np.expm1(RanForestFit.predict(X_test))) / 5

Final_score
test_prediction = pd.Series(Final_score, name="SalePrice")
# Making Submission file

Final_sub= pd.concat([Id_test,test_prediction],axis=1)

Final_sub.to_csv("submission_emrul.csv", index=False)
Final_sub.head()