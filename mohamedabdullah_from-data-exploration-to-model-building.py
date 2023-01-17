#import libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from fancyimpute import KNN

from sklearn.preprocessing import Imputer

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import SGDRegressor 

from sklearn.linear_model import Ridge 

from sklearn.linear_model import Lasso

from sklearn.linear_model import LinearRegression

from sklearn.svm import SVR

from sklearn.tree import DecisionTreeRegressor

from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error 

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import GridSearchCV



import warnings

warnings.filterwarnings('ignore')

%matplotlib inline



print("libraries loaded successfully")
#load data

data_train  = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

print("Data loaded successfully")
#exploar data

print("data shape : ",data_train.shape)

data_train.describe()
#get total count of data including missing data

total = data_train.isnull().sum().sort_values(ascending=False)



#get percent of missing data relevant to all data

percent = (data_train.isnull().sum()/data_train.isnull().count()).sort_values(ascending=False)



missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(25)
#drop PoolQC, MiscFeature, Alley, Fence columns

data_train = data_train.drop(['PoolQC','MiscFeature','Alley','Fence'], axis=1)



#after drop thes columns data shape will be (1460, 77) insted of (1460, 81)

print("data shape : ",data_train.shape)
#get continuous features

colnames_numerics_only = data_train.iloc[:,1:-1].select_dtypes(include=np.number).columns.tolist()

print('numerical features')

print(colnames_numerics_only)



print("----------------------------------------")



print("number of numerics features = ",len(colnames_numerics_only))
#impute missing values of continuous features using KNN

data_train[colnames_numerics_only] = KNN(k=5).fit_transform(data_train[colnames_numerics_only])

print('missing values of continuous features imputed successfully')
#get categorical features

colnames_categorical_only = data_train.iloc[:,1:-1].select_dtypes(include='object').columns.tolist()

print('categorical features')

print(colnames_categorical_only)



print("----------------------------------------")



print("number of categorical features = ",len(colnames_categorical_only))
for categorical_col in colnames_categorical_only:

    most_frequent = data_train[categorical_col].value_counts().idxmax()

    hasCol       = 'Has'+categorical_col

    

    #create new col 

    data_train[hasCol] = pd.Series(len(data_train[categorical_col]), index=data_train.index)

    

    #set new col = 1

    data_train[hasCol] = 1

    

    #set new col = 0 if data_train[categorical_col] not empty

    data_train.loc[data_train[categorical_col].isnull(),hasCol] = 0

    

    #set data_train[categorical_col] = most_frequent if new col = 0

    #if location of new col = 0 this mean that data_train[categorical_col] in this location is empty

    data_train.loc[data_train[hasCol] == 0,categorical_col] = most_frequent

    

    #drop new col

    data_train = data_train.drop(hasCol, axis=1)

    

print('missing values of categorical features imputed successfully')    
#print max count number of null values

print('Number of missing values = ',data_train.isnull().sum().max())
#box plot

cols = ['MSSubClass','LotFrontage','LotArea','OverallQual']

for col in cols:

    plt.figure()

    ax = sns.boxplot(x=data_train[col])
Q1 = data_train[colnames_numerics_only].quantile(0.25)

Q3 = data_train[colnames_numerics_only].quantile(0.75)

IQR = Q3 - Q1



hasOutlier = (data_train[colnames_numerics_only] < (Q1 - 1.5 * IQR)) | (data_train[colnames_numerics_only] > (Q3 + 1.5 * IQR))

hasOutlier
num_data = data_train[colnames_numerics_only]



for numeric_col in colnames_numerics_only: 

    data_train = data_train.drop(data_train.loc[hasOutlier[numeric_col]].index)
#after drop thes raws which contain outliers data raws will be less than 1460 raw 

print("data raws number : ",data_train.shape[0])
data_train = data_train.drop('Id', axis=1)

print('Id column deleted successfully')
#correlation matrix

corrmat = data_train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True)
#scatterplot

sns.set()

cols = ['LotFrontage','LotArea','OverallQual','YearBuilt','YearRemodAdd','MasVnrArea','TotalBsmtSF','1stFlrSF',

        'GrLivArea','FullBath','TotRmsAbvGrd','GarageYrBlt','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF']



group_1 = cols[0:8]

group_1.insert(0, "SalePrice")



#draw scatter plot of first group

sns.pairplot(data_train[group_1], size = 2)

plt.show();
group_2 = cols[8:]

group_2.insert(0, "SalePrice")



#draw scatter plot of first group

sns.pairplot(data_train[group_2], size = 2)

plt.show();
# all numerical features in our data

allNumericalFeatures = colnames_numerics_only



# numerical features which we use it in our model

selectedNumericalFeatures = cols



# numerical features that we will drop it

deletedFeatures =  list(set(allNumericalFeatures) - set(selectedNumericalFeatures))



print("data shape before delete features = ",data_train.shape)



# delete unwanted features

data_train = data_train.drop(deletedFeatures, axis=1)



print("data shape after delete features = ",data_train.shape)



print("unwanted features deleted successfully")
#convert categorical variable into lables

labelEncoder = LabelEncoder()



for categorical_col in colnames_categorical_only:

    data_train[categorical_col] =  labelEncoder.fit_transform(data_train[categorical_col])

    

print("categorical columns converted successfully")
print(colnames_categorical_only)
#data scaling

scaler = StandardScaler(copy=True, with_mean=True, with_std=True)

data_train[selectedNumericalFeatures] = scaler.fit_transform(data_train[selectedNumericalFeatures])



print("data scaling successfully")

data_train.describe()
X = data_train.drop('SalePrice', axis=1)

y = data_train['SalePrice']



X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.40, random_state=55, shuffle =True)

print('data splitting successfully')
#model bulding

SGDRRegModel = SGDRegressor(random_state=55,loss = 'squared_loss')

SelectedParameters = {

                      'alpha':[0.1,0.5,0.01,0.05,0.001,0.005],

                      'max_iter':[100,500,1000,5000,10000],

                      'tol':[0.0001,0.00001,0.000001],

                      'penalty':['l1','l2','none','elasticnet']

                      }



GridSearchModel = GridSearchCV(SGDRRegModel,SelectedParameters, cv = 5,return_train_score=True)

GridSearchModel.fit(X_train,y_train)



SGDRRegModel = GridSearchModel.best_estimator_

SGDRRegModel.fit(X_train,y_train)



print("stochastic gradient model run successfully")
RidgeRegModel = Ridge(random_state= 55, copy_X=True)

SelectedParameters = {

                      'alpha':[0.1,0.5,0.01,0.05,0.001,0.005],

                      'normalize':[True,False],

                      'max_iter':[100,500,1000,5000,10000],

                      'tol':[0.0001,0.00001,0.000001],

                      'solver':['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg']

                      }



GridSearchModel = GridSearchCV(RidgeRegModel,SelectedParameters, cv = 5,return_train_score=True)

GridSearchModel.fit(X_train,y_train)



RidgeRegModel = GridSearchModel.best_estimator_

RidgeRegModel.fit(X_train,y_train)



print("Ridge model run successfully")
LassoRegModel = Lasso(random_state= 55 ,copy_X=True)

SelectedParameters = {

                      'alpha':[0.1,0.5,0.01,0.05,0.001,0.005],

                      'normalize':[True,False],

                      'tol':[0.0001,0.00001,0.000001],

                      }



GridSearchModel = GridSearchCV(LassoRegModel,SelectedParameters, cv = 5,return_train_score=True)

GridSearchModel.fit(X_train,y_train)



LassoRegModel = GridSearchModel.best_estimator_

LassoRegModel.fit(X_train,y_train)



print("lasso model run successfully")
linearRegModel = LinearRegression(copy_X=True)

linearRegModel.fit(X_train,y_train)

print("Linear regression model run successfully")
decisionTreeModel = DecisionTreeRegressor(random_state=55)



SelectedParameters = {

                      'criterion': ['mse','friedman_mse','mae'] ,

                      'max_depth': [None,2,3,4,5,6,7,8,9,10],

                      'splitter' : ['best','random'],

                      'min_samples_split':[2,3,4,5,6,7,8,9,10],

                      }



GridSearchModel = GridSearchCV(decisionTreeModel,SelectedParameters, cv = 5,return_train_score=True)

GridSearchModel.fit(X_train,y_train)



decisionTreeModel = GridSearchModel.best_estimator_

decisionTreeModel.fit(X_train,y_train)



print("decision Tree Regressor model run successfully")
XGBRModel = XGBRegressor(n_jobs = 4)



SelectedParameters = {

                      'n_estimators': [100,1000,10000] ,

                      'learning_rate': [0.1,0.5,0.01,0.05],

                      }



GridSearchModel = GridSearchCV(XGBRModel,SelectedParameters, cv = 5,return_train_score=True)

GridSearchModel.fit(X_train,y_train)



XGBRModel = GridSearchModel.best_estimator_

XGBRModel.fit(X_train,y_train)



print("Xgboost Regressor model run successfully")
#evaluation Details

models = [SGDRRegModel, RidgeRegModel, LassoRegModel, linearRegModel, decisionTreeModel,XGBRModel]



for model in models:

    print(type(model).__name__,' Train Score is   : ' ,model.score(X_train, y_train))

    print(type(model).__name__,' Test Score is    : ' ,model.score(X_test, y_test))

    print('--------------------------------------------------------------------------')
#predict

for model in models:

    print(type(model).__name__," error metrics")

    print('---------------------------------------------------------')

    y_pred = model.predict(X_test)



    MAE = mean_absolute_error(y_test,y_pred)

    print("mean absolute error = ",MAE)



    MSE = mean_squared_error(y_test,y_pred)

    print("mean squared error = ",MSE) 



    RMSE = np.sqrt(mean_squared_error(y_test,y_pred))

    print("root mean squared error = ",RMSE) 

    print()