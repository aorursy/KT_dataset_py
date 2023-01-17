#Importing all the libaries and models that we will be using throughout this Notebook

import numpy as np # linear algebra

import pandas as pd # data processing and reading csv files

import matplotlib.pyplot as plt # visualising libary to visualise data

import seaborn as sns # A libary to visualise data

from scipy import stats

from scipy.stats import norm, skew

#Importing all the models we going to use in this Notebook

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.kernel_ridge import KernelRidge

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import RANSACRegressor

from sklearn import metrics

import re
#Importing the train and test csv into dataframes 

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
#train information

train.info()
#All the stats related information 

train.describe()
#Counting the dtype 

train.get_dtype_counts()
#Checking the variables that are 0.3 and above correlated with the Independant variable which is Saleprice

corrmat = train.corr()

top_f = corrmat.index[abs(corrmat['SalePrice'])>0.3]

top_f

train_id = train['Id'] #Making a variable called train_id which will containg the ID numbers of train

test_id = test['Id'] #Making a variable called test_id which will containg the ID numbers of test

train = train.drop('Id', axis = 1) #Dropping the train ID in our orginal DataFrame

test = test.drop('Id', axis = 1) #Dropping the test ID in our orginal DataFrame
#We are going to use this to seperate the combined train and test dataframe closer to the end 

ntrain = train.shape[0]

ntest = test.shape[0]
#Taking outliers out of the equation to produce better results

plt.scatter(x=train['GrLivArea'],y=train['SalePrice'])

train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

plt.scatter(x=train['GrLivArea'],y=train['SalePrice'])
#Checking the amount of data that is missing through visual representation

train_na = (train.isnull().sum() / len(train))*100

train_na = train_na.drop(train_na[train_na==0].index).sort_values(ascending=False)

plt.xticks(rotation='90')

plt.bar(train_na.index,train_na)
#Code for checking the percentage of the missing data in a dataFrame

missing_data = pd.DataFrame(train_na, columns = ['NAN VALUES in perc'])
#Missing data in percentage

missing_data
train["PoolQC"] = train["PoolQC"].fillna("None") # We will assume that for the NAN values in PoolQC, it indicates that theres no pool

train["MiscFeature"] = train["MiscFeature"].fillna("None")# Assuming theres no MiscFeatures

train["Alley"] = train["Alley"].fillna("None")#Assuming that the house doesnt have an Alley 

train["FireplaceQu"] = train["FireplaceQu"].fillna("None")#Assumuing theres no firplace when the values are NAN

train["Fence"] = train["Fence"].fillna("None")#Assuming theres no fence

for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:

    train[col] = train[col].fillna('None')#Assuming theres no Garage None

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    train[col] = train[col].fillna(0)#Assuming that theres no Garage by using 0 

train['LotFrontage'] = train.groupby("Neighborhood")["LotFrontage"].transform(lambda x:x.fillna(x.median()))

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    train[col] = train[col].fillna(0)#Assuming that theres no 

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    train[col] = train[col].fillna('None')

train["MasVnrType"] = train["MasVnrType"].fillna("None")

train["MasVnrArea"] = train["MasVnrArea"].fillna(0)

train['MSZoning'] = train['MSZoning'].fillna(train['MSZoning'].mode()[0])

train.drop('Utilities',inplace=True,axis=1)

train['Functional'] = train['Functional'].fillna('Typ')

mode_col = ['Electrical','KitchenQual', 'Exterior1st', 'SaleType']

for col in mode_col:

    train[col] = train[col].fillna(train[col].mode()[0])

train["Exterior2nd"] = train["Exterior2nd"].fillna("None")
def checking_nan(df):

    for column in df:

            if (df[column].isnull().sum() > 0):

                print(column)
checking_nan(train)
train.shape
sns.catplot("Fireplaces","SalePrice",data=train,hue="FireplaceQu", kind = 'point')
sns.violinplot(train["GarageCars"],train["SalePrice"])

plt.title("Garage Cars Vs SalePrice ")

plt.ylabel("SalePrice")

plt.xlabel("Number of Garage cars")

plt.show()
sns.stripplot(x="HeatingQC", y="SalePrice",data=train,hue='CentralAir',jitter=True, dodge = True)

plt.title("Sale Price vs Heating Quality")

plt.show()
sns.catplot("KitchenAbvGr","SalePrice",data=train,hue="KitchenQual", kind = 'point')

plt.title("Sale Price vs Kitchen");
plt.figure(figsize = (10,10))

plt.barh(train["Neighborhood"],train["SalePrice"])

plt.title("Sale Price vs Neighborhood");
sns.barplot(train['OverallQual'], train['SalePrice'])

plt.title('Sale Price vs Overall Qual')

plt.show()
#Creating copies of the original variables to manually encode it 

train['WoodDeckSF_Bool']  = train['WoodDeckSF'].copy()

train['OpenPorchSF_Bool']  = train['OpenPorchSF'].copy()

train['MasVnrArea_Bool']  = train['MasVnrArea'].copy()

train['GarageArea_Bool'] = train['GarageArea'].copy()

new_bool = 'WoodDeckSF_Bool', 'OpenPorchSF_Bool', 'MasVnrArea_Bool', 'GarageArea_Bool'
#Manual encoding for these variables. Esentially it indicates if there is a presence for that variable by using 1's and 0's

def changing_to_bool1(column):

    for rows in train[column]:

        if rows > 0:

            train[column] = train[column].replace(rows, 1)
#Iterating through new_bool to apply the function 

for col in new_bool:

    changing_to_bool1(col)
#Manual encoding for these variables. Esentially it indicates if there is a presence for that variable by using 1's and 0's

def changing_to_bool2(column):

    for rows in train[column]:

        if rows == 'Ex':

            train[column] = train[column].replace(rows, 5)

        elif rows == 'Gd':

            train[column] = train[column].replace(rows,4)

        elif rows == 'TA':

            train[column] = train[column].replace(rows,3)

        elif rows == 'Fa':

            train[column] = train[column].replace(rows,2)

        elif rows == 'Po':

            train[column] = train[column].replace(rows,1)

        elif rows == 'None':

            train[column] = train[column].replace(rows,0)
bool_values = ('FireplaceQu', 'ExterQual', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual')
#Iterating through bool_values to apply the function

for col in bool_values:

    changing_to_bool2(col)
#Manual encoding for these variables. Esentially it indicates if there is a presence for that variable by using 1's and 0's

def changing_to_bool3(column):

    for rows in train[column]:

        if rows == 'Y':

            train[column] = train[column].replace(rows,1)

        else:

            train[column] = train[column].replace(rows, 0)
changing_to_bool3('CentralAir')
#Manual encoding for these variables. Esentially it indicates if there is a presence for that variable by using 1's and 0's

def changing_to_bool4(column):

    for rows in train[column]:

        if rows == 'GLQ':

            train[column] = train[column].replace(rows,6)

        elif rows == 'ALQ':

            train[column] = train[column].replace(rows,5)

        elif rows == 'BLQ':

            train[column] = train[column].replace(rows,4)

        elif rows == 'Rec':

            train[column] = train[column].replace(rows,3)

        elif rows == 'LwQ':

            train[column] = train[column].replace(rows,2)

        elif rows == 'Unf':

            train[column] = train[column].replace(rows,1)

        elif rows == 'None':

            train[column] = train[column].replace(rows,0)
changing_to_bool4('BsmtFinType1')
#This a combination of variables to create new ones 

train['TotalSF'] = train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF']

train['TotalBath'] = train['BsmtFullBath'] + (train['BsmtHalfBath'] / 0.5) + train['FullBath'] + (train['HalfBath'] / 0.5)

train['Age'] = train['YrSold'] -train['YearBuilt']

train['TotalFinSF'] = (train['GrLivArea'] + train['BsmtFinSF1'] + train['BsmtFinSF2']) - train['LowQualFinSF'] 

train['OtherAreas'] = train['TotalBsmtSF'] + train['GarageArea'] + train['OpenPorchSF'] + train['WoodDeckSF'] + train['3SsnPorch'] + train['OpenPorchSF'] + train['MasVnrArea']

train['AreaPerRoom'] = train['GrLivArea'] / train['TotRmsAbvGrd']

train['Porches_SF'] = train['WoodDeckSF'] + train['OpenPorchSF'] + train['ScreenPorch']
sns.jointplot(x='Porches_SF', y='SalePrice', data=train, kind='reg', height=10, color= 'green')

#plt.title('Sale Price vs Total Porches SF')

plt.show()
sns.jointplot(x='AreaPerRoom', y='SalePrice', data=train, kind='reg', height=10);

plt.show();
sns.jointplot(x='OtherAreas', y='SalePrice', data=train, kind='reg', height=10, color = 'green');

plt.show();
sns.jointplot(x='TotalFinSF', y='SalePrice', data=train, kind='reg', height=10);

plt.show();
sns.jointplot(x='Age', y='SalePrice', data=train, kind='reg', height=10, color = 'green');

plt.show();
sns.jointplot(x='TotalBath', y='SalePrice', data=train, kind='reg', height=10);

plt.show();
sns.jointplot(x='TotalSF', y='SalePrice', data=train, kind='reg', height=10, color = 'green');

plt.show();
#Correlation between the all the variables versus Sale Price that we manually encoded on top 

plt.figure(figsize = (9,9))

sns.heatmap(train[['FireplaceQu', 'ExterQual', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual','SalePrice']].corr(),annot =True, cmap="YlGnBu")
#Highest variables that correlates with SalePrice 

corrmat = train.corr()

top_f = corrmat.index[abs(corrmat['SalePrice'])>0.6]

top_f

plt.figure(figsize=(10,10))

g = sns.heatmap(train[top_f].corr(),annot=True)
#Importing the train and test csv into dataframes 

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train_id = train['Id'] #Making a variable called train_id which will containg the ID numbers of train

test_id = test['Id'] #Making a variable called test_id which will containg the ID numbers of test

train = train.drop('Id', axis = 1) #Dropping the train ID in our orginal DataFrame

test = test.drop('Id', axis = 1) #Dropping the test ID in our orginal DataFrame
#Taking outliers out of the equation to produce better results

plt.scatter(x=train['GrLivArea'],y=train['SalePrice'])

train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
#This function will check the skewness of the data

def checkskew(col):

    sns.distplot(train[col],fit=norm)

    (mu, sigma) = norm.fit(train[col])

    print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

checkskew('SalePrice')
#We will then use np.log1p to make the SalePrice data evenly distributed 

train['SalePrice'] = np.log1p(train['SalePrice'])

checkskew('SalePrice')
#We will create a new variable called y_train, we will use this as the independant variable 

y_train = train.SalePrice.values

train.drop('SalePrice',axis=1,inplace=True)

y_train.shape
#We are going to use this to seperate the combined train and test dataframe closer to the end 

ntrain = train.shape[0]

ntest = test.shape[0]
train.shape
y_train.shape
train["PoolQC"] = train["PoolQC"].fillna("None") # We will assume that for the NAN values in PoolQC, it indicates that theres no pool

train["MiscFeature"] = train["MiscFeature"].fillna("None")# Asuuming theres no MiscFeatures

train["Alley"] = train["Alley"].fillna("None")#Assuming that the house doesnt have an Alley 

train["FireplaceQu"] = train["FireplaceQu"].fillna("None")#Assumuing theres no firplace when the values are NAN

train["Fence"] = train["Fence"].fillna("None")#Assuming theres no fence

for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:

    train[col] = train[col].fillna('None')#Assuming theres no Garage None

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    train[col] = train[col].fillna(0)#Assuming that theres no Garage by using 0 

train['LotFrontage'] = train.groupby("Neighborhood")["LotFrontage"].transform(lambda x:x.fillna(x.median()))

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    train[col] = train[col].fillna(0)#Assuming that theres no basement by using 0

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    train[col] = train[col].fillna('None')

train["MasVnrType"] = train["MasVnrType"].fillna("None")

train["MasVnrArea"] = train["MasVnrArea"].fillna(0)

train['MSZoning'] = train['MSZoning'].fillna(train['MSZoning'].mode()[0])

train.drop('Utilities',inplace=True,axis=1)

train['Functional'] = train['Functional'].fillna('Typ')

mode_col = ['Electrical','KitchenQual', 'Exterior1st', 'SaleType']

for col in mode_col:

    train[col] = train[col].fillna(train[col].mode()[0])

train["Exterior2nd"] = train["Exterior2nd"].fillna("None")
#Function to check the NAN values

def checking_nan(df):

    for column in df:

            if (df[column].isnull().sum() > 0):

                print(column)
checking_nan(train)
test_na = (test.isnull().sum() / len(train))*100

test_na = test_na.drop(test_na[test_na==0].index).sort_values(ascending=False)

plt.xticks(rotation='90')

plt.bar(test_na.index,test_na)
#Doing the same here for what i did to the train dataset 

test["PoolQC"] = test["PoolQC"].fillna("None")

test["MiscFeature"] = test["MiscFeature"].fillna("None")

test["Alley"] = test["Alley"].fillna("None")

test["FireplaceQu"] = test["FireplaceQu"].fillna("None")

test["Fence"] = test["Fence"].fillna("None")

for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:

    test[col] = test[col].fillna('None')

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    test[col] = test[col].fillna(0)

test['LotFrontage'] = test.groupby("Neighborhood")["LotFrontage"].transform(lambda x:x.fillna(x.median()))

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    test[col] = test[col].fillna(0)

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    test[col] = test[col].fillna('None')

test["MasVnrType"] = test["MasVnrType"].fillna("None")

test["MasVnrArea"] = test["MasVnrArea"].fillna(0)

test['MSZoning'] = test['MSZoning'].fillna(test['MSZoning'].mode()[0])

test.drop('Utilities',inplace=True,axis=1)

test['Functional'] = test['Functional'].fillna('Typ')

mode_col = ['Electrical','KitchenQual', 'Exterior1st', 'SaleType']

for col in mode_col:

    test[col] = test[col].fillna(test[col].mode()[0])

test["Exterior2nd"] = test["Exterior2nd"].fillna("None")
#Function to check the NaN values 

def checking_nan(df):

    for column in df:

            if (df[column].isnull().sum() > 0):

                print(column)
checking_nan(test)
#Joining the test and train dataset to make data (new dataframe)

data = pd.concat((train, test), sort = False).reset_index(drop=True)
data.shape
#This a combination of variables to create new ones 

data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']

data['TotalBath'] = data['BsmtFullBath'] + (data['BsmtHalfBath'] / 0.5) + data['FullBath'] + (data['HalfBath'] / 0.5)

data['Age'] = data['YrSold'] - data['YearBuilt']

data['TotalFinSF'] = (data['GrLivArea'] + data['BsmtFinSF1'] + data['BsmtFinSF2']) - data['LowQualFinSF'] 

data['OtherAreas'] = data['TotalBsmtSF'] + data['GarageArea'] + data['OpenPorchSF'] + data['WoodDeckSF'] + data['3SsnPorch'] + data['OpenPorchSF'] + data['MasVnrArea']

data['AreaPerRoom'] = data['GrLivArea'] / data['TotRmsAbvGrd']

data['Porches_SF'] = data['WoodDeckSF'] + data['OpenPorchSF'] + data['ScreenPorch']
def changing_to_bool2(column):

    for rows in data[column]:

        if rows == 'Ex':

            data[column] = data[column].replace(rows, 5)

        elif rows == 'Gd':

            data[column] = data[column].replace(rows,4)

        elif rows == 'TA':

            data[column] = data[column].replace(rows,3)

        elif rows == 'Fa':

            data[column] = data[column].replace(rows,2)

        elif rows == 'Po':

            data[column] = data[column].replace(rows,1)

        elif rows == 'None':

            data[column] = data[column].replace(rows,0)
bool_values = ('FireplaceQu', 'ExterQual', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual')
data = data.drop([ 'MiscVal', 'MiscFeature'], axis = 1 )
set1 = data.dtypes[data.dtypes != 'object'].index

skew_f = data[set1].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

skewed = pd.DataFrame({'Skew':skew_f})

skewed
skewed = skewed[abs(skewed) > 0.75]

from scipy.special import boxcox1p

skewed_features = skewed.index

lam = 0.15

for feat in skewed_features:

    data[feat] = boxcox1p(data[feat], lam)
mylist = list(data.select_dtypes(include=['object']).columns)
dummies = pd.get_dummies(data[mylist], prefix= mylist)
data.drop(mylist, axis=1, inplace = True)
data = pd.concat([data,dummies], axis =1 )
train = data[:ntrain]

test = data[ntrain:]

train.shape
test.shape
train.shape
y_train.shape
#Creating variables for the Independent and dependent variables 

X = train

y = y_train
#Splitting the train and test 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#This cell will be refered to as "the iterator" if discussed in comments.

#In this cell, various regression models are prepared to take turns being active. Three models are chosen for each run of 

#the function. During a full run, the iterator will fit and measure the mse of 99 iterations of three models. The model's 

#parameters can be set to adjust according to the iteation count, this allows a user to focus in and find the ideal setting for

#a model or combination of models. Outputs are stored in lists and are available for other functions later.

#You can go boil the kettle because this should take almost ten minutes.



import warnings

warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import Lasso

import lightgbm as lgb



lislas = []

output=[]

tree=[]

submission_options = []

aggregate = []

for g in range(10):

    for l in range(10):

        w = str(g) + str(l)

        val = float('0.000' + w)

        def KR():

            KRR = KernelRidge(alpha=float('0'+'.'+str(g)+w), degree=3, coef0=2.9, kernel='polynomial')

            KRR.fit(X_train, y_train)

            KR = KRR.predict(X_test)

            return (KR, KRR)

            

        def GBO():

            GBoost = GradientBoostingRegressor(n_estimators=1030, learning_rate=float('0.014'+w)+0.005,

                                   max_depth=3, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=float('0.142'),#+float('0.0'+w), 

                                   loss='huber', random_state =5)

            GBoost.fit(X_train, y_train)

            GBO = GBoost.predict(X_test)

            return(GBO, GBoost)

            

        def LIG():

            light = lgb.LGBMRegressor(objective='regression',num_leaves=3, 

                                      learning_rate=0.05, n_estimators=550 +int(w),

                                      max_bin = 55, bagging_fraction = 0.8,

                                      bagging_freq = 5, feature_fraction =0.2172,

                                      feature_fraction_seed=9,bagging_seed=9, min_data_in_leaf =3,

                                      min_sum_hessian_in_leaf = 11)

            light.fit(X_train, y_train)

            Lig = light.predict(X_test)

            return (Lig, light)

            

        def ENet():

            ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=float('0.000'+w), l1_ratio=2.222, random_state=3, warm_start=True))

            ENet.fit(X_train, y_train)

            EN = ENet.predict(X_test)

            return (EN, ENet)

        

        def FR():

            rf = RandomForestRegressor(n_estimators=150, criterion='mae', max_features=float('0.0'+w)+0.001, n_jobs=-1)

            rf.fit(X_train,y_train)

            rfr = rf.predict(X_test)

            return (rfr, rf)

        

        def las():

            las = make_pipeline(RobustScaler(), Lasso(alpha =float('0.000' + '49'), random_state=True))

            las.fit(X_train,y_train)

            la = las.predict(X_test)

            return (la, las)



        model_dict = {

        'las':las(),

        'KRR':KR(),

        'ENet':ENet(),

        'GB':GBO(),

        'Lig':LIG(),

        'FR': FR()}

        

        #Assign variables to chosen regression models for use in the following functions.

        model1 = 'las'

        reg1, md1 = model_dict[model1]

        model2 = 'KRR'

        reg2, md2 = model_dict[model2]

        model3 = 'GB'

        reg3, md3 = model_dict[model3]

        

        combined = []

        for e in range(len(reg1)):

            trio = (reg1[e] + reg2[e] + reg3[e])/3

            combined.append(trio) 

        

        #Calculate individual MSEs and also MSE of the average of all three model's predictions.

        mse_reg1 = metrics.mean_squared_error(y_test, reg1)

        mse_reg2 = metrics.mean_squared_error(y_test, reg2)

        mse_reg3 = metrics.mean_squared_error(y_test, reg3)

        mse_trio = metrics.mean_squared_error(y_test, combined)

        

        #Details from each iteration and fitting of models is stored in a list 'lislas'

        lislas.append((int(str(g) + str(l)), mse_reg1, mse_reg2, mse_reg3, mse_trio))

        

        #a seperate ist is made containing the average of all three model's prdictions for each iteration. This data can be used

        #for analysis and tweaking of outputs later on.

        aggregate.append(combined)

        

        #A tuple containing each model's prediction is crated and stored in a list with the iteration number as a form of ID

        trioOut = ((reg1, reg2, reg3))

        submission_options.append(trioOut)

        

        #Results are printe while function runs in order to provide live feedback.

        print(w)

        print('reg1 mse:', mse_reg1)

        print('reg2 mse:', mse_reg2)

        print('reg3 mse:', mse_reg3)

        print('trio mse:', mse_trio)
#BEST BLEND

#Locates and returns the iteration where the three models stacked equally performed best.

def bestBlend():

    ave_val = 99999

    ave_pos = 0

    for t, b, z, h, f in lislas:

        if f < ave_val:

            ave_val = f

            

            ave_pos = t

    return(lislas[t])

#Here bestBlend function is used to return the best iteration for all three models stacked with eqal weighting.

print(bestBlend()[-1], bestBlend()[0])
#BEST reg1

#Here you can find the best prediction by reg1. This allows the user to note the iteration number and use that number to

#fine tune any variable that might have been set to be adjusted by the iteration count ('val' in the iterator ).

def reg1():

    temp = 500

    alpha = 0

    for t, b, z, h, f in lislas:

        if b < temp:

            temp = b

            alpha = t

    return (temp, alpha)       

reg1()
#BEST Reg2

#Here you can find the best prediction by reg2. This allows the user to note the iteration number and use that number to

#fine tune any variable that might have been set to be adjusted by the iteration count ('val' in the iterator ).

def reg2():

    low = 99999

    bow = 0

    for t, b, z, h, f in lislas:

        if z < low:

            low = z

            bow = t

    return (low, bow)

print(reg2())
#BEST reg3

#Here you can find the best prediction by reg1. This allows the user to note the iteration number and use that number to

#fine tune any variable that might have been set to be adjusted by the iteration count ('val' in the iterator ).

def reg3():

    tr = 100

    tr_pos = 0

    for t, b, z, h, f in lislas:

        if h < tr:

            tr = h

            tr_pos = t

    return (tr, tr_pos)          

print(reg3())    
def run(lis, loc, g, t, first, second):

    #This function receives the output list from the iterator. It then uses the the parameters 'g' and 't' to

    #create a new list that is the product of their weights according to paramters 'first' and 'second'. The parameters

    #'first' states what percentage to take from the  list 'g' and 'second' is the percentage taken from list 't'.

    #This function will be called three hundred times (Three permutations of picking teo lists from the three stored

    #in 'solution_options', and this is done for each percent).

    temp=[]

    ng = lis[loc][g]

    nt = lis[loc][t]

    ng = ng * first

    nt = nt * second

    for i in range(len(ng)):

        ng[i] = ng[i] + nt[i]

    temp = [str(g) + ' AND ' + str(t),  ng, str(first) + ' x ' + str(g), str(second) + ' x ' + str(t), first, second]

    return temp
def run2(lis, loc, g, t, first, second):

    #This function receives the output list from the iterator. It then uses the the parameters 'g' and 't' to

    #create a new list that is the product of their weights according to paramters 'first' and 'second'. The parameters

    #'first' states what percentage to take from the  list 'g' and 'second' is the percentage taken from list 't'.

    #This function will be called three hundred times (Three permutations of picking teo lists from the three stored

    #in 'solution_options', and this is done for each percent).

    a = 3-(g+t)

    temp=[]

    ng = lis[loc][g]

    nt = lis[loc][t]

    na = lis[loc][a]

    ng = ng * first

    nt = nt * second

    na = na * first

    

    for w in range(len(nt)):

        nt[w] = nt[w] + ng[w]

    no = nt * second    



    for i in range(len(na)):

        na[i] = na[i] + no[i]



    temp = [str(g) + ' AND ' + str(t),  na, str(first) + ' x ' + str(g), str(second) + ' x ' + str(t), first, second]

    return temp
#This function receives the output list from the iterator. It then 

def weight(inflow, loc, first, second, flag):

    #This function divides the three lists in 'possible_solutions' into three permutations where each list is ppartnered with 

    #both other lists once. The function then passes 'possible_solutions' to the function 'run' with instructions on which two 

    #lists to process along with the percentage weightings for each list.

    j=[]

    g=[[],[],[]]

    a = [[0, 1], [1,2], [2,0]]

    if flag ==1:

        for u, d in enumerate(a):

            a,b,c,d,e,f = run(inflow, loc, d[0], d[1], first, second)

            g[u] = [a,b,c,d,e,f]

    if flag == 2:

        for u, d in enumerate(a):

            a,b,c,d,e,f = run2(inflow, loc, d[0], d[1], first, second)

            g[u] = [a,b,c,d,e,f]

        

    return(g)     



def evaluateWeight(df, loc, flag):

    #This function iterates through percentages (one to a hundred) and provides the function 'weight' with two complimentry 

    #percentages that will be used to combine the predictions in 'possible_solutions' lists in one percent increments.

    theThree=[]

    for q in range(0, 101):

        q = q/100

        k = 1 - q

        theThree.append(weight(df, loc, q, k, flag))

    return (theThree)   
def seeker():

    #This function goes through all three hundred combinations and weightings of predictions to find the combination that

    #produced the best MSE. Once it has compared all entries, it outputs the iteration ID information along with some other 

    #information. The ID is used to calculate the settings that were used by 'the iterator' to achieve this MSE. These settings

    #will later be used to fit the models in order to predict the y values for the test set. 

    

    retainer = []

    check1 = [bestBlend()[-1]]

    check2 = [bestBlend()[-1]]

    temp_mse = 500



    for p in range(len(submission_options)):

        options = evaluateWeight(submission_options, p, 1)

        options2 = evaluateWeight(submission_options, p, 2)

        

        #second index: list of combination permutations with varying weights/blends in a list.

        for s in range(len(options)):

            for yi in range(3):

                cross_v = (metrics.mean_squared_error(y_test, options[s][yi][1]),1)

      #         print(options[s][yi][1][200])

                cross_vi = (metrics.mean_squared_error(y_test, options2[s][yi][1]),2)

      #          print(options2[s][yi][1][200], 'vi', 'p is:', p, 'yi is', yi)

                

                if cross_v[0] < temp_mse:

                    temp_mse = cross_v[0]

                    check1.append(cross_v[0])

                    retainer.append((options[s][yi][0], options[s][yi][2],options[s][yi][3], cross_v[0],  'ID:', p, 'Yi:', yi, cross_v[1], options[s][yi][4],options[s][yi][5]))

                if cross_vi[0] < temp_mse:

                    temp_mse = cross_vi[0]

                    check2.append(cross_vi[0])

                    retainer.append((options2[s][yi][0], options2[s][yi][2],options2[s][yi][3], cross_vi[0],  'ID:', p, 'Yi:', yi, cross_vi[1],options[s][yi][4], options[s][yi][5]))



    

    if len(retainer) == 0:

        return bestBlend()

    print(check1[-1], check2[-1])

    return retainer[-1]



seeker = seeker()    



#A printout below is read and information used in callibrating the models used ofr the final prediction.

#The information in the output is as follows.

#Lists used in combination (predictions by 'reg1', 'reg2' and 'reg3' in 'the iterator')

seeker
#Parameters for refitting models to train set with new weights are assigned to variables.

firs = seeker[9]

sec = seeker[10]

permutations = [[0, 1], [1,2], [2,0]]

model_dicti = {

    1:md1,

    2:md2,

    3:md3}



#This option is initiated if the best mse was achieved with a blend of outputs directly from the models.

if seeker[8] == 1:

    modelDetails = seeker()

    models = modelDetails[0]

    modelApercent = modelDetails[1]

    modelBpercent = modelDetails[2]

    

    modelA = int(models[0])+1

    modelB = int(models[-1])+1

    print(modelB, 'modelB')

    selected_A = model_dicti[modelA] 

    selected_B = model_dicti[modelB]

    

    #Extract percentage Float from string output provided by 'seeker' function

    modelApe = re.findall(r"[-+]?\d*\.\d+|\d+", modelApercent)

    modelApercent = round(float(modelApe[0]), 3)

    modelBpe = re.findall(r"[-+]?\d*\.\d+|\d+", modelBpercent)

    modelBpercent = round(float(modelBpe[0]), 3)



#This option is activated if the best mse was achieved through a combination of blends of outputs from the models - still under construction.

if seeker[8] == 2:

    submission_options = np.array(submission_options)

    print(type(submission_options))

    modelB = int(models[0])+1

    modelC = int(models[-1])+1

    modelA = 6-(modelB+modelC)

    if modelA == 0:

        one = np.array(submission_options[0] * fir * sec)

        two = np.array(submission_options[1] * sec * sec)

        three = np.array(submission_options[2] * sec * sec)

    if modelA == 1:

        one = np.array(submission_options[1] * fir * sec)

        two = np.array(submission_options[2] * sec * sec)

        three = np.array(submission_options[2] * fir * sec)

        four = np.array(submission_options[0] * sec * sec)

    if modelA == 2:

        one = np.array(submission_options[2] * fir * sec)

        two = np.array(submission_options[0] * sec * sec)

        three = np.array(submission_options[0] * fir * sec)

        four = np.array(submission_options[1] * sec * sec)

        

    

    modelApercent = modelDetails[1]

    modelBpercent = modelDetails[2]

    modelCpercent = modelDetails[1]

    

    #Extract percentage Float from string output provided by 'seeker' function

    modelApe = re.findall(r"[-+]?\d*\.\d+|\d+", modelApercent)

    modelApercent = round(float(modelApe[0]), 3)

    modelBpe = re.findall(r"[-+]?\d*\.\d+|\d+", modelBpercent)

    modelBpercent = round(float(modelBpe[0]), 3)

    modelCpe = re.findall(r"[-+]?\d*\.\d+|\d+", modelCpercent)

    modelCpercent = round(float(modelBpe[0]), 3)

    

    

    selected_A = model_dicti[modelA]

    selected_B = model_dicti[permutations]

    selected_C = model_dicti[modelC]

    print('selected_A', selected_A, modelApercent)

    print('selected_B', selected_B, modelBpercent)

    print('selected_C', selected_C, modelCpercent)

    

#Fit models on new settings

if seeker[-1] == 1:

    model_A = selected_A.fit(X, y)

    model_B = selected_B.fit(X, y)

    A_toAdjust = model_A.predict(test)

    B_toAdjust = model_B.predict(test)

    

    #Apply weights to each model's output and add them together to make a full complement

    A_out = np.expm1(A_toAdjust) * 0.88

    B_out = np.expm1(B_toAdjust) * 0.12

    final_out = A_out + B_out

    

if seeker[-1] == 2:

    print(modelApercent)

    print(modelBpercent)

    print(modelCpercent)

    model_A = selected_A.fit(X, y)

    model_B = selected_B.fit(X, y)

    model_C = selected_C.fir(X, y)

    A_toAdjust = model_A.predict(test)

    B_toAdjust = model_B.predict(test)

    C_toAdjust = model_C.predict(test)

    

    #Apply weights to each model's output and add them together to make a full complement

    A_out = np.expm1(A_toAdjust) * modelApercent

    B_out = np.expm1(B_toAdjust) * modelBpercent

    C_out = np.expm1(C_toAdjust) * modelCpercent

    final_out = A_out + B_out

pd.DataFrame({'Id': test_id, 'SalePrice': final_out}).to_csv('Yhat.csv', index =False)
from IPython.display import HTML



def create_download_link(title = "Download Test file", filename = "Yhat.csv"):  

    html = '<a href={filename}>{title}</a>'

    html = html.format(title=title,filename=filename)

    return HTML(html)

create_download_link(filename='Yhat.csv')
pd.read_csv('Yhat.csv')