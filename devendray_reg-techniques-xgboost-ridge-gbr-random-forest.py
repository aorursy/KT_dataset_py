import numpy as np

import pandas as pd



# Ignore warnings

import warnings

warnings.filterwarnings('ignore')



from sklearn.feature_extraction.text import TfidfVectorizer

# For preprocessing--conversion of categorical to dummy

from sklearn.preprocessing import LabelEncoder, OneHotEncoder 

#  For splitting arrays into train/test

from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor

#  ridge regression 

from sklearn.linear_model import Ridge



from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA



#XGboot

import xgboost as xgb

# lightgbm

import lightgbm as lgb

#Visualization 

import matplotlib.pyplot as plt

import seaborn as sns 





import time

from scipy import sparse

from sklearn import metrics

# To plot pretty figures

%matplotlib inline


#Import Data from File

data_train = pd.read_csv("../input/train.csv");

data_test = pd.read_csv("../input/test.csv");



#to play with train data we'll create a copy

data_raw = data_train.copy(deep = True)



#however passing by reference is convenient, because we can clean both datasets at once

data_cleaner = [data_train , data_test]
#Preview Data

data_train.shape

data_train.sample(10)
data_train.describe(include = 'all')
# Get a list of columns having NAs

def ColWithNAs(x):            

    z = x.isnull()

    df = np.sum(z, axis = 0)       # Sum vertically, across rows

    col = df[df > 0].index.values 

    return (col)



def numericalCol(x):            

     return x.select_dtypes(include=[np.number]).columns.values



# Fill in missing values

def MissingValuesAsOther(t , filler = "other"):

    return(t.fillna(value = filler))





def MissingValuesAsMean(t ):

    return(t.fillna(t.mean(), inplace = True))



def MissingValuesAsZero(t ):

     return(t.fillna(value=0, inplace = True))





# Delete the columns

def ColDelete(x,drop_column):

    x.drop(drop_column, axis=1, inplace = True)



# Convert categorical features to dummy

def DoDummy(x):

   #data_dummy =pd.get_dummies(x)

   # Try: le.fit_transform(list('abc'))

    le = LabelEncoder()

    # Apply across all columns of x

    y = x.apply(le.fit_transform)

    # Try:  enc.fit_transform([[1,2],[2,1]]).toarray()

    enc = OneHotEncoder(categorical_features = "all")  # ???all???: All features are treated as categorical.

    enc.fit(y)

    trans = enc.transform(y)

    return(trans)   



def RegressionEvaluationMetrics(regr,X_test_sparse,y_test_sparse,title):

    predictions=regr.predict(X_test_sparse)

    plt.figure(figsize=(8,6))

    plt.scatter(predictions,y_test_sparse,cmap='plasma')

    plt.title(title)

    plt.show()

    print('MAE:', metrics.mean_absolute_error(y_test_sparse, predictions))

    print('MSE:', metrics.mean_squared_error(y_test_sparse, predictions))

    print('RMSE:', np.sqrt(metrics.mean_squared_error(np.log1p(y_test_sparse), np.log1p(predictions))))



def Regression(regr,X_test_sparse,y_test_sparse):

    start = time.time()

    regr.fit(X_train_sparse,y_train_sparse)

    end = time.time()

    rf_model_time=(end-start)/60.0

    print("Time taken to model: ", rf_model_time , " minutes" ) 

    

def SaveResult(SalePrice,test_ids,file):

    OutputRF = pd.DataFrame(data=SalePrice,columns = ['SalePrice'])

    OutputRF['Id'] = test_ids

    OutputRF = OutputRF[['Id','SalePrice']]

    OutputRF.to_csv(file,index=False)

# Assign 'Saleprice' column to a variable and drop it from tr

#     Also drop train_id/test_id columns

y = data_train['SalePrice']             # This is also the target variable

test_ids = data_test['Id']



data_train.drop( ['SalePrice', 'Id'], inplace = True, axis = 'columns')

data_test.drop( ['Id'], inplace = True, axis = 'columns')    



print('Train columns with null values:\n', data_train.isnull().sum().sort_values())

print("-"*20)



print('Test/Validation columns with null values:\n', data_train.isnull().sum().sort_values())

print("-"*20)

#delete the columns Alley/PoolQC/MiscFeature in train dataset , has all NA values

drop_column = ['MiscFeature','PoolQC', 'Alley','Fence','FireplaceQu']

    

###COMPLETING: complete or delete missing values in train and test/validation dataset

for dataset in data_cleaner:   

    ColDelete(dataset,drop_column)

    columns_replace_with_mean_value = numericalCol(dataset)

    dataset[columns_replace_with_mean_value] = MissingValuesAsMean(dataset[columns_replace_with_mean_value])

    columns_with_na = ColWithNAs(dataset)

    print("Columns with NA :",columns_with_na,"\n")

    dataset[columns_with_na] = MissingValuesAsOther(dataset[columns_with_na])

    print('columns with null values:\n', dataset.isnull().sum().sort_values())

    print("-"*20)



preProcessData = pd.concat(data_cleaner, axis = 'index')

preProcessData.shape
#Let's create some simple plots to check out the data!



sns.distplot(y )

sns.pairplot(data_raw[['SalePrice','YearBuilt','LotArea','GrLivArea','BsmtFinSF1','GarageArea']])



print("-"*30)



sns.pairplot(data_raw[['SalePrice','YearRemodAdd','TotalBsmtSF', '1stFlrSF','BedroomAbvGr']])



#rint('columns with null values:\n', preProcessData[numerical_columns_name].isnull().sum().sort_values())

plt.figure(figsize=(15,10))

sns.heatmap(preProcessData.corr())

#   Categorical and Numerical Columns 

#   Number of unique values per column. Maybe some columns

#   are actually categorical in nature

numerical_columns_name = numericalCol(preProcessData)



categorical_columns_name = preProcessData.columns.difference(numerical_columns_name)

preProcessData[categorical_columns_name].head(10)





scaler = StandardScaler()

scaler.fit(preProcessData[numerical_columns_name])

scaled_data = scaler.transform(preProcessData[numerical_columns_name])

scaled_data.shape

type(scaled_data)





model_pca = PCA()

pca = PCA(n_components=4)

pca.fit(scaled_data)

num_pca = pca.transform(scaled_data)



num_pca.shape

type(num_pca)

sparse_num_pca=sparse.csr_matrix(num_pca)
plt.figure(figsize=(13,10))

plt.scatter(num_pca[:data_raw.shape[0],0],num_pca[:data_raw.shape[0],1],num_pca[:data_raw.shape[0],2],c=data_raw['SalePrice'],cmap='jet')

plt.show()
print(pca.components_)



df_comp = pd.DataFrame(pca.components_,columns=numerical_columns_name)



plt.figure(figsize=(15,10))

sns.heatmap(df_comp,cmap='plasma',)
PCADataDF = pd.DataFrame(num_pca, columns = ['P1','P2','P3','P4'])

#PCADataDF
#len(preProcessData.select_dtypes(include=[np.number]).columns.values)

Nunique = preProcessData[categorical_columns_name].nunique()

Nunique= Nunique.sort_values()

Nunique
# Convert categorical to dummy

start = time.time()

categorical_data = DoDummy(preProcessData[categorical_columns_name])

end = time.time()  

dummy_time = (end-start)/60.0

print("Time taken to convert categorical to dummy features: ", dummy_time , " minutes" )

categorical_data.shape
# Concatenate Categorical + Numerical Data

df_sp = sparse.hstack([categorical_data,sparse_num_pca], format = 'csr')



df_sp.shape

type(df_sp)

##  Unstack train and test, sparse matrices

df_train = df_sp[ : data_train.shape[0] , : ]

df_test = df_sp[data_train.shape[0] :, : ]

df_train.shape

df_test.shape



#  PArtition datasets into train + validation

#y_train = np.log1p(y)    # Criterion is rmsle

y_train = y

X_train_sparse, X_test_sparse, y_train_sparse, y_test_sparse = train_test_split(

                                     df_train, y_train,

                                     test_size=0.30,

                                     random_state=42

                                     )
# Instantiate a RandomRegressor object

MAXDEPTH = 50

regr = RandomForestRegressor(n_estimators=1000,       # No of trees in forest

                             criterion = "mse",       # Can also be mae

                             max_features = "sqrt",  # no of features to consider for the best split

                             max_depth= MAXDEPTH,    #  maximum depth of the tree

                             min_samples_split= 2,   # minimum number of samples required to split an internal node

                             min_impurity_decrease=0, # Split node if impurity decreases greater than this value.

                             oob_score = True,       # whether to use out-of-bag samples to estimate error on unseen data.

                             n_jobs = -1,            #  No of jobs to run in parallel

                             random_state=0,

                             verbose = 10            # Controls verbosity of process

                             )





#  Do regression

Regression(regr,X_test_sparse,y_test_sparse)

 

#  Prediction and performance Of RandomForestRegressor

RegressionEvaluationMetrics(regr,X_test_sparse,y_test_sparse,"Random Forest Model")



# Prediction on test Data

SalePrice = regr.predict(df_test)



## Save Result

#SaveResult(SalePrice,test_ids,"Random_Forest_Regression_output.csv")
# Linear Models

# Ridge Regression

model = Ridge(alpha = 40.0,            # Regularization strength. Try 0.0 and 40

              solver = "lsqr",        # auto,svd,cholesky,lsqr,sparse_cg,sag,saga

              fit_intercept=False     # Data is already normalized and centered

              )



#  Do regression

Regression(model,X_test_sparse,y_test_sparse)

 

#  Prediction and performance Of RandomForestRegressor

RegressionEvaluationMetrics(model,X_test_sparse,y_test_sparse,"Ridge Regression Model")



# Prediction on test Data

SalePrice = model.predict(df_test)



## Save Result

#SaveResult(SalePrice,test_ids,"Ridge_Regression_output.csv")

## Gradient Boosting Model





# Fit regression model

paramsLs = {'n_estimators': 5000, 

          'max_depth': 3, 

          'min_samples_split': 3,

          'learning_rate': 0.01}

clf = GradientBoostingRegressor(**paramsLs)

#  Do regression

Regression(clf,X_test_sparse,y_test_sparse)

 

#  Prediction and performance Of RandomForestRegressor

RegressionEvaluationMetrics(clf,X_test_sparse,y_test_sparse,"Gradient Boosting Model")



# Prediction on test Data

SalePrice = clf.predict(df_test)



## Save Result

#SaveResult(SalePrice,test_ids,"GradientBoosting_Regression_output.csv")





### xgboost



##

#from sklearn.ensemble import GradientBoostingClassifier

#gbm = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=500,min_samples_leaf=50,max_depth=8,max_features='sqrt',subsample=0.8,random_state=10)

#gbm = xgb.XGBClassifier(nthread=4)

max_depth = 3

min_child_weight = 10

subsample = 0.5

colsample_bytree = 0.6

objective = 'reg:linear'

num_estimators = 5000

learning_rate = 0.01



gbm = xgb.XGBRegressor(max_depth=max_depth,

                min_child_weight=min_child_weight,

                colsample_bytree=colsample_bytree,

                objective=objective,

                n_estimators=num_estimators,

                learning_rate=learning_rate)

#gbm.fit(X_test_sparse, y_test_sparse)

#  Do regression

Regression(gbm,X_test_sparse,y_test_sparse)

 



#  Prediction and performance Of RandomForestRegressor

RegressionEvaluationMetrics(gbm,X_test_sparse,y_test_sparse,"XGBoost Model")



# Prediction on test Data

SalePrice = gbm.predict(df_test)



## Save Result

#SaveResult(SalePrice,test_ids,"XGBoost_Regression_output.csv")




