import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px





#filter out warnings

import warnings 

warnings.filterwarnings('ignore')



#To style plots

plt.style.use('fivethirtyeight')



#cycle the colors

from itertools import cycle

color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])



#Get the kaggle input

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

train.head()

print(train.shape)

print(test.shape)
train.isnull().sum().sort_values(ascending=False)[0:20]
sns.heatmap(train.isnull(),yticklabels=False,cbar='BuPu')



train.info()


'''train['LotFrontage'] = train['LotFrontage'].fillna(train['LotFrontage'].mean)

train.drop(['Alley'], axis = 1, inplace=True)



train['BsmtCond']=train['BsmtCond'].fillna(train['BsmtCond'].mode()[0])

train['BsmtQual']=train['BsmtQual'].fillna(train['BsmtQual'].mode()[0])

train.drop(['PoolQC'], axis = 1, inplace=True)

train.drop(['Fence'], axis = 1, inplace=True)    

train.drop(['MiscFeature'], axis = 1, inplace=True)'''





#clean the train data

for i in list(train.columns):

    dtype = train[i].dtype

    values = 0

    if(dtype == float or dtype == int):

        method = 'mean'

    else:

        method = 'mode'

    if(train[i].notnull().sum() / 1460 <= .5):

        train.drop(i, axis = 1, inplace=True)

    elif method == 'mean':

        train[i]=train[i].fillna(train[i].mean())



    else:

        train[i]=train[i].fillna(train[i].mode()[0])

        print(train[i])





#clean the test data

for i in list(test.columns):

    dtype = test[i].dtype

    values = 0

    if(dtype == float or dtype == int):

        method = 'mean'

    else:

        method = 'mode'

    if(test[i].notnull().sum() / 1460 <= .5):

        test.drop(i, axis = 1, inplace=True)

    elif method == 'mean':

        test[i]=test[i].fillna(test[i].mean())



    else:

        test[i]=test[i].fillna(test[i].mode()[0])





train.head()

test.shape
test.drop(columns=['Id'], inplace=True)

train.dropna(inplace=True)



train.drop(columns=['Id'], inplace=True)

print(train.shape)

print(test.shape)
train.isnull().any().any()

train.head()

#df1=pd.get_dummies(train['MSZoning'],drop_first=True)

#print(df1)
plt.figure(figsize=(15,5))

plt.plot(train.SalePrice,linewidth=2,color=next(color_cycle))

plt.title('Distribution Plot for Sales Prices')

plt.ylabel('Sales Price');
#sort the values

plt.figure(figsize=(15,5))

plt.plot(train.SalePrice.sort_values().reset_index(drop=True),color=next(color_cycle))

plt.title('Distribution Plot for Sales Prices')

plt.ylabel('Sales Price');
fig = px.scatter(train,x=train.index, y='SalePrice', labels={'x':'Index'},

                 color=train.MSZoning, template="seaborn",

                 title='Sale Price distriution of MSZoning')

fig.show()
fig = px.scatter(train,x=train.index, y='SalePrice', labels={'x':'Index'},

                 color=train.Street, template="seaborn",

                 title='Sale Price distriution ---> Street')

fig.show()
plt.figure(figsize=(20,10))



plt.subplot(2,2,1)

plt.scatter(x=train[train.LotConfig == 'FR3'].index,

           y=train[train.LotConfig == 'FR3'].SalePrice,color=next(color_cycle))

plt.title('SalePrice distribution of FR3 value of LotConfig')



plt.subplot(2,2,2)

plt.scatter(x=train[train.LotConfig == 'CulDSac'].index,

           y=train[train.LotConfig == 'CulDSac'].SalePrice,color=next(color_cycle))

plt.title('SalePrice distribution of CulDSac value of LotConfig')



plt.subplot(2,2,3)

plt.scatter(x=train[train.LotConfig == 'Corner'].index,

           y=train[train.LotConfig == 'Corner'].SalePrice,color=next(color_cycle))

plt.title('SalePrice distribution of Corner value of LotConfig')



plt.subplot(2,2,4)

plt.scatter(x=train[train.LotConfig == 'FR2'].index,

           y=train[train.LotConfig == 'FR2'].SalePrice,color=next(color_cycle))

plt.title('SalePrice distribution of FR2 value of  LotConfig');
columns = ['MSZoning', 'Street',

       'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',

       'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 

       'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',

       'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',

       'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',

       'Heating', 'HeatingQC', 'CentralAir', 'Electrical',

       'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish',

       'GarageQual', 'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition']


train_test_data = pd.concat([train, test], axis = 0)

print(test.shape)

print(train.shape)



train_test_data.head()

train_test_data.shape
def One_hot_encoding(columns):

    df_final=train_test_data

    i=0

    for fields in columns:

        df1=pd.get_dummies(train_test_data[fields],drop_first=True)

        

        train_test_data.drop([fields],axis=1,inplace=True)

        if i==0:

            df_final=df1.copy()

        else:           

            df_final=pd.concat([df_final,df1],axis=1)

        i=i+1

       

        

    df_final=pd.concat([train_test_data,df_final],axis=1)

        

    return df_final
train_test_data = One_hot_encoding(columns)

print(train_test_data.shape)

train_test_data.head()
train_test_data.columns.duplicated()

train_test_data =train_test_data.loc[:,~train_test_data.columns.duplicated()]



train_test_data.shape
from scipy.stats import norm, skew

from scipy import stats
sns.distplot(train['SalePrice'], fit=norm);

fig = plt.figure()

res = stats.probplot(train['SalePrice'], plot=plt)
train["SalePrice"] = np.log(train["SalePrice"])



res = stats.probplot(train['SalePrice'], plot=plt)
df_Train=train_test_data.iloc[:1460,:]

df_Test=train_test_data.iloc[1460:,:]

print(df_Test.shape)



df_Test.head()
df_Test.drop(['SalePrice'],axis=1,inplace=True)

X_train_final=df_Train.drop(['SalePrice'],axis=1)

y_train_final=df_Train['SalePrice']

X_train_final.shape
from sklearn.preprocessing import StandardScaler

#make the data into a normal distrubition

#mean = 0 

X_std = StandardScaler().fit_transform(X_train_final)



my_columns = X_train_final.columns

new_df = pd.DataFrame(X_std, columns=my_columns)
from sklearn.decomposition import PCA

pca = PCA(n_components = 2)

df_pca = pca.fit_transform(new_df)

print(df_pca)
plt.figure(figsize =(8, 6))

plt.scatter(df_pca[:, 0], df_pca[:, 1], c = y_train_final, cmap ='plasma')

# labeling x and y axes

plt.xlabel('First Principal Component')

plt.ylabel('Second Principal Component');
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

#split the data set into train and test

X_train, X_test, y_train, y_test = train_test_split(X_train_final, y_train_final)

linreg = LinearRegression()

linreg.fit(X_train, y_train)



#Accuracy

print("R-Squared Value for Training Set: ",linreg.score(X_train, y_train))

print("R-Squared Value for Test Set: ",linreg.score(X_test, y_test))
from sklearn.neighbors import KNeighborsRegressor



knnreg = KNeighborsRegressor(n_neighbors = 2)

knnreg.fit(X_train, y_train)



print('R-squared train score:',knnreg.score(X_train, y_train))

print('R-squared test score: ',knnreg.score(X_test, y_test))
from sklearn.linear_model import Ridge

ridge = Ridge()

ridge.fit(X_train, y_train)

print('R-squared train score:',ridge.score(X_train, y_train))

print('R-squared test score: ',ridge.score(X_test, y_test))
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()



X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)



ridge = Ridge(alpha=20)

ridge.fit(X_train_scaled, y_train)





print('R-squared score (training): {:.3f}'.format(ridge.score(X_train_scaled, y_train)))

print('R-squared score (test): {:.3f}'.format(ridge.score(X_test_scaled, y_test)))
from sklearn.linear_model import Lasso



lasso = Lasso(max_iter = 10000)

lasso.fit(X_train, y_train)



print('R-squared score (training): {:.3f}'.format(lasso.score(X_train, y_train)))

print('R-squared score (test): {:.3f}'.format(lasso.score(X_test, y_test)))



lasso = Lasso(alpha=100, max_iter = 10000)

lasso.fit(X_train_scaled, y_train)



print('R-squared score (training): {:.3f}'.format(lasso.score(X_train_scaled, y_train)))

print('R-squared score (test): {:.3f}'.format(lasso.score(X_test_scaled, y_test)))

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler



lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))

lasso.fit(X_train_scaled, y_train)

print('R-squared score (training): {:.3f}'.format(lasso.score(X_train_scaled, y_train)))

print('R-squared score (test): {:.3f}'.format(lasso.score(X_test_scaled, y_test)))

df_Test.head()
from sklearn.ensemble import GradientBoostingRegressor

GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5)

GBoost.fit(X_train_final, y_train_final)



y_pred = GBoost.predict(df_Test)

y_pred
from sklearn.ensemble import RandomForestClassifier



regressor = RandomForestClassifier()

regressor.fit(X_train_scaled, y_train)

print('R-squared score (training): {:.3f}'.format(regressor.score(X_train_scaled, y_train)))

print('R-squared score (test): {:.3f}'.format(regressor.score(X_test_scaled, y_test)))

'''from sklearn.model_selection import RandomizedSearchCV



n_estimators = [100,200, 300, 500,700, 900]

criterion = ['gini', 'entropy']

depth = [3,5,10,15]

min_split=[2,3,4]

min_leaf=[2,3,4]

bootstrap = ['True', 'False']

verbose = [5]



hyperparameter_grid = {

    'n_estimators': n_estimators,

    'max_depth':depth,

    'criterion':criterion,

    'bootstrap':bootstrap,

    'verbose':verbose,

    'min_samples_split':min_split,

    'min_samples_leaf':min_leaf

    }



random_cv = RandomizedSearchCV(estimator=regressor,

                               param_distributions=hyperparameter_grid,

                               cv=5, 

                               scoring = 'neg_mean_absolute_error',

                               n_jobs = 4, 

                               return_train_score = True,

                               random_state=42)'''

X_train_final
#random_cv.fit(X_train_final,y_train_final)
#random_cv.best_estimator_#
regressor = RandomForestClassifier(bootstrap='False', max_depth=15, min_samples_leaf=4,n_estimators=900, verbose=5)

#regressor.fit(X_train_final,y_train_final)
#y_pred = regressor.predict(df_Test)

#y_pred
#prediction=pd.DataFrame(y_pred)

#samp = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

#print(samp['Id'])

#sub = pd.concat([samp['Id'],prediction], axis=1)

#sub.columns=['Id','SalePrice']

#print(sub)

#sub.to_csv('My_submission.csv',index=False)

import xgboost

regressor=xgboost.XGBRegressor()
'''n_estimators = [100, 500, 900, 1100, 1500]

max_depth = [2, 3, 5, 10, 15]

booster=['gbtree','gblinear']

learning_rate=[0.05,0.1,0.15,0.20]

min_child_weight=[1,2,3,4]

base_score=[0.25,0.5,0.75,1]



# Define the grid of hyperparameters to search

hyperparameter_grid = {

    'n_estimators': n_estimators,

    'max_depth':max_depth,

    'learning_rate':learning_rate,

    'min_child_weight':min_child_weight,

    'booster':booster,

    'base_score':base_score

    }

random_cv = RandomizedSearchCV(estimator=regressor,

            param_distributions=hyperparameter_grid,

            cv=5, n_iter=50,

            scoring = 'neg_mean_absolute_error',n_jobs = 4,

            verbose = 5, 

            return_train_score = True,

            random_state=42)'''
#random_cv.fit(X_train_final,y_train_final)

#andom_cv.best_estimator_
regressor = xgboost.XGBRegressor(base_score=0.25, booster='gbtree', colsample_bylevel=1,

             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,

             importance_type='gain', interaction_constraints='',

             learning_rate=0.1, max_delta_step=0, max_depth=2,

             min_child_weight=1, missing=None, monotone_constraints='()',

             n_estimators=900, n_jobs=0, num_parallel_tree=1, random_state=0,

             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,

             tree_method='exact', validate_parameters=1, verbosity=None)
regressor.fit(X_train_final,y_train_final)
#y_pred = regressor.predict(df_Test)

#y_pred
pred=pd.DataFrame(y_pred)

samp = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

sub = pd.concat([samp['Id'],pred], axis=1)

sub.columns=['Id','SalePrice']
sub

sub.to_csv('My_sub1.csv',index=False)