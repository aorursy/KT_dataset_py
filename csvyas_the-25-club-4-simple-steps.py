import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# import both the dataset

train_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')



# shapes of both the dataset

print(f'train shape : {train_data.shape} , test data : {test_data.shape}')
# Run some test on our Hero

train_data.SalePrice.describe()
# let's visualize and see what the sale price distribution looks like

plt.figure(figsize=(7,5))

sns.distplot(train_data['SalePrice'],bins = 25)

plt.ticklabel_format(style='sci', axis='x', scilimits=(0,1))

plt.xlabel("House Sales Price (10^5) in USD")

plt.ylabel("Number of Houses")

plt.title("House Sales Price Distribution")



# the skewness and the kurtosis

print(f'Skewness : {train_data.SalePrice.skew()}')

print(f'Kurtosis : {train_data.SalePrice.kurt()}')
# pairplot

sns.set()

numerical_cols = ['LotArea','TotalBsmtSF','GrLivArea','TotRmsAbvGrd','PoolArea','MiscVal','SalePrice']

sns.pairplot(train_data[numerical_cols], height= 2.5)

plt.show()
#boxplot

categorical_col = ['YearBuilt','OverallQual','MSZoning','Neighborhood','BsmtQual']



for col in categorical_col:

    f, ax = plt.subplots(figsize=(12, 5))

    sns.boxenplot(x=col, y='SalePrice', data = train_data)

    plt.show()
# correlation heatmap

corr = train_data.corr()

plt.figure(figsize = (40, 30))

sns.heatmap(corr, annot = True)

plt.show()
# scatter plot for outliar's detection

var = train_data['TotalBsmtSF']

sns.set()

sns.scatterplot(train_data['SalePrice'], var)

plt.title("Before Removing Outliar's")

plt.show()



# remove outliar from TotalBsmtSF

Indx = train_data[((train_data.TotalBsmtSF>5000))].index

train_data.drop(Indx,inplace=True)



# see weather outliar gone

var = train_data['TotalBsmtSF']

sns.set()

sns.scatterplot(train_data['SalePrice'], var)

plt.title("After Removing Outliar's")

plt.show()
# scatter plot for outliar's detection

var = train_data['GrLivArea']

sns.set()

sns.scatterplot(train_data['SalePrice'], var)

plt.title("Before Removing Outliar's")

plt.show()



# remove outliar from GrLivArea

Index = train_data[((train_data.GrLivArea>4000))].index

for ind in Index:

    train_data.drop(ind, inplace=True)



# scatter plot for outliar's detection

var = train_data['GrLivArea']

sns.set()

sns.scatterplot(train_data['SalePrice'], var)

plt.title("After Removing Outliar's")

plt.show()
# scatter plot for outliar's detection

var = train_data['GarageCars']

sns.set()

sns.scatterplot(train_data['SalePrice'], var)

plt.show()
# scatter plot for outliar's detection

var = train_data['OverallQual']

sns.set()

sns.scatterplot(train_data['SalePrice'], var)

plt.show()
# scatter plot for outliar's detection

var = train_data['YearBuilt']

sns.set()

sns.scatterplot(train_data['SalePrice'], var)

plt.title("Before Removing Outliar's")

plt.show()



# remove outliar from YearBuilt

Indx = train_data[((train_data.SalePrice>400000)&(train_data.SalePrice<500000)&(train_data.YearBuilt<1900))].index

train_data.drop(Indx,inplace=True)



# scatter plot for outliar's detection

var = train_data['YearBuilt']

sns.set()

sns.scatterplot(train_data['SalePrice'], var)

plt.title("After Removing Outliar's")

plt.show()
# split dependent and independent feature from dataset

x = train_data.iloc[:,0:80]

y = train_data.iloc[:,-1]



# combine train and test data

df = pd.concat([x,test_data],axis = 0)



# drop the Id colum from df

df = df.drop(['Id'], axis=1)
# dropping all the unnecessary features

drop_features = ['MSSubClass','OverallCond','BsmtFinType2','BsmtFinSF2','LowQualFinSF','BsmtHalfBath','PoolArea','MiscVal','MoSold', 'YrSold','GarageArea',

                '1stFlrSF','TotRmsAbvGrd','GarageYrBlt','LotArea','Alley','LandContour','LandSlope','Heating','Electrical','BsmtExposure','BldgType',

                'Utilities','Functional']

df.drop(drop_features,axis=1,inplace=True)

# search for missing data

def missing_values(data):

    values = data.isnull().sum().sort_values(ascending = False)

    percentage = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)

    missing_df = pd.concat([values,percentage], axis=1, keys=['Values', 'Percentage%'])

    return missing_df



missing_df  = missing_values(df)

missing_df.head(25)
# plot the missing data

sns.set()

plt.figure(figsize = (10, 5))

sns.barplot(missing_df.index[0:26],missing_df['Percentage%'].head(26))

plt.xticks(rotation=90)

plt.show()
# remove the features with more than 50% missing data

df.drop([ 'PoolQC','MiscFeature','Fence','FireplaceQu'], axis=1, inplace=True)



# numerical column imputation

num_colums = ['LotFrontage','MasVnrArea','BsmtFullBath','BsmtFinSF1','TotalBsmtSF','BsmtUnfSF','GarageCars']

for col in num_colums:

    df[col].fillna(df[col].mean(), inplace = True)

    

# categorical column imputation

catg_colums = ['GarageCond','GarageFinish','GarageQual','GarageType','BsmtCond','BsmtQual','BsmtFinType1','MasVnrType','MSZoning','Exterior1st','Exterior2nd',

              'SaleType','KitchenQual']

for col in catg_colums:

    df[col].fillna(df[col].value_counts().index[0], inplace=True)
# normalize the target variable

from scipy import stats

from scipy.stats import norm



# visualize the taarget variable

sns.distplot(y, fit =norm);

fig = plt.figure()

prob = stats.probplot(y, plot = plt)

plt.show()



print("so we see that the SalePrice is show some positive skewness and peakedness also in probability plot we clearly see that it is deviate from it's diagnol line \n but we can handle it by doing the log transformation, so let's see how it handle it")



# apply log transformation

y = np.log(y)



# again visualize it

sns.distplot(y, fit =norm);

fig = plt.figure()

prob = stats.probplot(y, plot = plt)

plt.show()
# normalize the GrLivArea

col = 'GrLivArea'

    

# visualize the taarget variable

sns.distplot(df[col], fit =norm);

fig = plt.figure()

prob = stats.probplot(df[col], plot = plt)

plt.show()



print("so we see that the GrLivArea is show some positive skewness and peakedness also in probability plot we clearly see that it is deviate from it's diagnol line \n but we can handle it by doing the log transformation, so let's see how it handle it")



# apply log transformation

df[col] = np.log(df[col])

    

# again visualize it

sns.distplot(df[col], fit =norm);

fig = plt.figure()

prob = stats.probplot(df[col], plot = plt)

plt.show()

    
# get dummy variables

df = pd.get_dummies(df, drop_first=True)

df.head()
# feature scaling

from sklearn.preprocessing import MinMaxScaler

minmax = MinMaxScaler()

normal_df = pd.DataFrame(minmax.fit_transform(df))
# split the train and test data

train = normal_df.iloc[:1455,:].values

test = normal_df.iloc[1455:,:].values
# import all necessary libraries

from sklearn.linear_model import LinearRegression,BayesianRidge,Ridge,Lasso,HuberRegressor,ElasticNet

from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,BaggingRegressor,GradientBoostingRegressor

from catboost import CatBoostRegressor

from sklearn.svm import SVR

from lightgbm import LGBMRegressor

from xgboost import XGBRegressor

from mlxtend.regressor import StackingCVRegressor

import tensorflow as tf

from sklearn.model_selection import cross_val_score, KFold

from sklearn.metrics import mean_squared_error,make_scorer,r2_score

from sklearn.model_selection import RandomizedSearchCV



# DL libraries

from keras.callbacks import ModelCheckpoint

from keras.models import Sequential

from keras.layers import Dense, Activation, Flatten
# apply multiple Algo

import time

from sklearn.utils.testing import ignore_warnings

from sklearn.exceptions import ConvergenceWarning



@ignore_warnings(category=ConvergenceWarning)

def apply_algorithm(train,label):



    start_time = time.time()

    model_score = {} # list that contain the model scores

        

    # build the model and return error

    def model(reg):

        '''we will pass our regressor in this function and '''

        cross_val = KFold(n_splits=10 ,shuffle= True , random_state= 0)

        rmsle = make_scorer(r2_score)

        validation = cross_val_score(reg ,train, label, cv=cross_val, scoring=rmsle,n_jobs=-1)

        return validation.mean()

    

    # Linear Regression

    linearReg = LinearRegression()

    model_score['LinearReg'] = model(linearReg)

    

    # randomForest

    randForest = RandomForestRegressor(n_estimators=150,n_jobs=-1)

    model_score['Rndom Forest'] = model(randForest)

    

    # adboost

    adaboost = AdaBoostRegressor(n_estimators=100)

    model_score['AdaBoost'] = model(adaboost)

    

    # bayseian Ridge

    bayaeRidge = BayesianRidge()

    model_score['Bayesian Ridge'] = model(bayaeRidge)

    

    # Ridge

    ridge = Ridge()

    model_score['Ridge'] = model(ridge)

    

    # lasso

    lasso = Lasso()

    model_score['Lasso'] = model(lasso)

    

    # Hubber

    hubber = HuberRegressor()

    model_score['Hubber'] = model(hubber)

    

    # SVM

    svr = SVR()

    model_score['SVR'] = model(svr)

    

    # Elastic-net

    elastinet = ElasticNet()

    model_score['Elatic-Net'] = model(elastinet)

    

    # gradient boosting

    GDboost = GradientBoostingRegressor()

    model_score['Gradient Boost'] = model(GDboost)

    

    # LightGBM

    lightgb = LGBMRegressor()

    model_score['Light GBM'] = model(lightgb)

    

    # Xgboost

    xgboost = XGBRegressor()

    model_score['XG Boost'] = model(xgboost)

    

    # CatBoost

    catboost = CatBoostRegressor()

    model_score['CatBoost'] = model(catboost)

    

    # visualize in Dataframe

    final_scores = pd.DataFrame.from_dict(model_score,orient='index')

    final_scores.columns = ['R2 Scores']

    final_scores = final_scores.sort_values('R2 Scores',ascending=False)

    

    # visualize in plots

    final_scores.plot(kind = 'bar',title = 'Model Accoracies')

    axes = plt.gca()

    axes.set_ylim([0 ,1])

    

    # time taken

    end_time = time.time()

    time_taken = end_time - start_time

    print(f'Time taken to apply all Algorithms  {time_taken} Seconds')

    

    return final_scores



apply_algorithm(train,y)
# initialize the base models



# apply hyperParameter optimization on ridge regressiom

parameters = {'alpha':[0.001,0.01,0.1,1,2,3,40,50,100,200, 230, 250,265, 270, 275, 290, 300, 500]}

ridge_1 = Ridge()



# apply randomized searchcv

randomSearch = RandomizedSearchCV(ridge_1, param_distributions=parameters, n_jobs=-1, cv=10)

randomSearch.fit(train,y)



# best parametrs 

bestpara = randomSearch.best_params_



# build the model and return error

def model(reg,train,label):

    '''we will pass our regressor in this function and '''

    cross_val = KFold(n_splits=10 ,shuffle= True , random_state= 0)

    rmsle = make_scorer(r2_score)

    validation = cross_val_score(reg ,train, label, cv=cross_val, scoring=rmsle,n_jobs=-1)

    return validation.mean()



# base model 1

model_1 = BayesianRidge()

score_1 = model(model_1,train,y)





# base model 2

model_2 = CatBoostRegressor()

score_2 = model(model_2,train,y)





# meta regressor

ridge = Ridge(alpha=1)

score = model(ridge,train,y)



# apply stacking 

''' Stack up models and optimize using Ridge'''

stack_model = StackingCVRegressor(regressors= (model_1,model_2), meta_regressor=ridge)

final_score = model(stack_model,train,y)

stack_gen = stack_model.fit(train,y)
# accuracies

print('Bayesian Ridge Accuracy = {0:.2f}%'.format(score_1*100))

print('CatBoost Accuracy = {0:.2f}%'.format(score_2*100))

print('Ridge Accuracy = {0:.2f}%'.format(score*100))

print('Stack Accuracy = {0:.2f}%'.format(final_score*100))
# predict the scores

test_pred = stack_gen.predict(test)

submission = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')

submission['SalePrice'] = np.floor(np.expm1(test_pred)) # inversing and flooring log scaled saleprice to see orignal prediction

submission = submission[['Id', 'SalePrice']]



# create submission file

submission.to_csv('blend_submission.csv',index=False)

print('submission file is created')