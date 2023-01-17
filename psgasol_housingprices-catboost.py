# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import random

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.




       



gridSearch = False # [False, True] If False, no gridSearch is performed

random_state = random.randint(1,100000) # Everytime a new random_state is given for separating the dataset, if good, it is printed somewhere to be seen



TrainTestSklearn = True # If True, the function from sklearn is used without already defaulted values

DataOwnSplit = False # If data DataOwnSplit = True, then percTest and Obsxgroup are important, so that i will use range of SalePrice to appear for each prediction

percTrain = 0.66 # a value between 0,1 ; if wanted, it splits the dataset in that percentage taking into account groups created by order of y predictor

obsxgroup = 40 # Nº of observations x group on the regression



corr_elim = True #[False, True] If corr elimination is set to True, then the correlations above corr_threshold are taken into account

corr_threshold = 0.95 # Variables that are correlated over that number are eliminated of the dataset



# RandomSeedListAppend=[]

# DataName = "GoodSeeds"

# with open(DataName, "w") as file:

#     file.write(str(RandomSeedListAppend))



DataName = "GoodSeeds"

with open( DataName, "r") as file:

    RandomSeedListAppend = eval(file.readline())





ColumnNames = np.array(['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',

       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',

       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',

       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',

       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',

       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',

       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',

       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',

       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',

       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',

       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',

       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',

       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',

       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',

       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',

       'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',

       'SaleCondition', 'SalePrice'])#'Id',



ColumnTypes = np.array([

               1, 1,0,0,1,0,0,0,0,0,

               0,0,0,0,0,0,0,1,1,1,

               1,0,0,0,0,0,0,0,0,0,

               0,0,0,0,1,0,1,1,1,0,

               0,0,0,1,1,1,1,1,1,1,

               1,1,1,0,1,0,1,0,0,0,

               0,1,1,0,0,0,1,1,1,1,

               1,1,0,0,0,1,1,1,0,0,

               1])# Id is removed and the value is 1 # 1 == num, 0== string



num_cols=ColumnNames[ColumnTypes ==1]

cat_cols=ColumnNames[ColumnTypes ==0]
# Se lee el documento

# Se importan las librerías

import catboost as cb

import sklearn 

from sklearn.model_selection import train_test_split

from catboost import CatBoostRegressor, CatBoostClassifier



sample_submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")



submission = pd.DataFrame()

submission['Id'] = test['Id']



# Set X_train, Y_train

Target = 'SalePrice'

OrderColumnForSplit = "Order"



predictors = [predictor for predictor in train.columns if predictor not in [Target]]





if DataOwnSplit ==True:

    # Adding a column to create my train test split

    train[OrderColumnForSplit]= np.argsort(train[Target])



    Y_to_split = "Y_to_split"



    def NumValuesbyGroup(df, y, obs_per_group, nameCol ):

        n_obs = len(train[y])

        df[nameCol] = [str(int(val/obs_per_group)) for val in df[y]]

        df.drop(y, axis=1, inplace=True)

        return(df.copy())



    train= NumValuesbyGroup(train,OrderColumnForSplit, obsxgroup,Y_to_split )





    ### Start My split Code



    print("myrandom value is "+ str(random_state))



    Mytrain_ids =[] 

    uniquevals =train[Y_to_split].unique()

    for value in uniquevals:

        mysample =train[train[Y_to_split]==str(value)].Id

        MyIdForTrainByGroup =np.random.choice(mysample,int(len(mysample)*percTest),replace=False)

        Mytrain_ids= np.concatenate([Mytrain_ids, MyIdForTrainByGroup])

    

    ids_train_int =  Mytrain_ids.copy()



    # this are my set of row ids for train and test

    ids_train = set(ids_train_int)

    ids_val = set(range(train.shape[0]))-set(Mytrain_ids)

    ### End My split Code



Y_train = train[Target]

X_train = train[predictors]





num_cols =X_train._get_numeric_data().columns

cat_cols = list(set(predictors) - set(num_cols))



corr_matrix = X_train[num_cols].corr().abs()



if corr_elim == True:

    # Select upper triangle of correlation matrix

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))



    # Find index of feature columns with correlation greater than 0.95

    to_drop = [column for column in upper.columns if any(upper[column] > corr_threshold)]



    X_train.drop(to_drop, axis=1, inplace=True)

    print("the variables being droped are "+ str(to_drop))

    test.drop(to_drop, axis=1, inplace=True)
# stupid_vars =['Id']

# X_train.drop(stupid_vars, axis=1, inplace=True)

# test.drop(stupid_vars, axis=1, inplace=True)



# Get categorical indexes to pass it to the function

cat_index =  [idx for idx, i in enumerate(cat_cols) if i in X_train.columns]







from pandas.api.types import is_string_dtype

from pandas.api.types import is_numeric_dtype

treatment = ""

# In Need to recognize Nans that are not recognized as such

if treatment == "":

#     print("The general one is done !")

#     print(X_train.isnull().sum()[X_train.isnull().sum()>0])

#     print(test.isnull().sum()[test.isnull().sum()>0])

    DatasetList = [X_train, test]

    for df in DatasetList:

        for colWithNan in df.columns[df.isnull().sum()>0]:

            if ColumnTypes[np.where(ColumnNames==colWithNan)] ==0: 

                changeto ='' 

            elif ColumnTypes[np.where(ColumnNames==colWithNan)] ==1:

                changeto =-2000000

            df[colWithNan][df[colWithNan].isnull()] =changeto

            df[cat_cols] = df[cat_cols].astype(str)

                                

    print("The null in train data are:")

    print(X_train.isnull().sum()[X_train.isnull().sum()>0])

    print("The null in test data are:")    

    print(test.isnull().sum()[test.isnull().sum()>0])





if treatment != "":

    ColumnsWithMissings = train.columns[train.isnull().sum()>0]

    #for each columnWith Missing

    #Id_ RowsMissings = 



    # 1. Choosing the column to be the target

    mytarget= ColumnsWithMissings[1]



    # 2. Choosing all X variables to be predictors

    predictors = [predictor for predictor in train.columns if predictor not in [mytarget]]



    # 3. Creating my X, Y datasets

    X_train_Nan = train[predictors]

    Y_train_Nan = train[mytarget]





    print("The nº of columns in train is "+ str(train.shape[1])+ " AND IT IS USED")

    #print("The nº of columns in X_train is "+ str(X_train.shape[1]) + " but it is not used")

    print("The nº of columns in X_train_Nan is "+ str(X_train_Nan.shape[1]))

    print("The nº of columns in cat_cols is "+ str(len(cat_cols)))

    print("The nº of columns in num_cols is "+ str(len(num_cols)))

    # 4. Selecting indexes where Y are np.nan. They will be used as val for prediction using the logic of our code

    IdxNan = [i for i in range(len(Y_train_Nan)) if pd.isnull(np.array([Y_train_Nan[i]]))==True]  

    IdxNoNan =  [i for i in range(len(Y_train_Nan)) if pd.isnull(np.array([Y_train_Nan[i]]))==False]  





    # 6. Setting the indexes and names of the columns that are numeric and categorical

    num_cols =X_train_Nan._get_numeric_data().columns

    cat_cols = list(set(predictors) - set(num_cols))

    cat_indexes =  [int(idx) for idx, i in enumerate(X_train_Nan.columns) if i in cat_cols]



    # 5. All other np.nan from X are set to ''. Here the doubt is the order of the columns to pass for Nans to be optimized

    print(len(X_train_Nan.columns[X_train_Nan.isnull().sum()>0]))

    X_train_Nan.iloc[:,cat_indexes] = X_train_Nan.iloc[:,cat_indexes].replace(np.nan, '', regex=True)# replacing the other variables in the dataset as nan for string

    X_train_Nan.loc[:,num_cols] = X_train_Nan.loc[:,num_cols].replace(np.nan, -2000000, regex=True)# replacing the other variables in the dataset as the min value invented by me which is -20000000



    print(X_train_Nan)

    print(len(X_train_Nan.columns[X_train_Nan.isnull().sum()>0]))

    # 6. Setting the indexes and names of the columns that are numeric and categorical

    num_cols =X_train_Nan._get_numeric_data().columns

    cat_cols = list(set(predictors) - set(num_cols))

    cat_indexes =  [int(idx) for idx, i in enumerate(X_train_Nan.columns) if i in cat_cols]

    print(len(X_train_Nan.columns[X_train_Nan.isnull().sum()>0]))



    print(cat_cols)

    # 7. Setting X_train_train and X_train_test to check how well the model perform later. Then the ones being Nans as X_train_val will be predicted if good

    X_train_train, X_train_test, Y_train_train, Y_train_test = train_test_split(X_train_Nan.loc[IdxNoNan], Y_train_Nan[IdxNoNan], test_size=0.33, random_state=random_state)

    #print(Y_train_Nan[IdxNoNan].unique())

    X_train_val = X_train_Nan.loc[IdxNan] # The Nan Values from this paper





    # 8. Depending on whether the variable to predict is numerical or categorical or multiclass I choose a different Catboost.

    if is_string_dtype(Y_train_Nan[IdxNoNan])==True:

        print("It is string")



        if len(Y_train_Nan[IdxNoNan].unique()) ==2:

            lossfunc = "Logloss"

        else: 

            lossfunc = "F1"

        model=CatBoostClassifier(depth=7, 

                            learning_rate=0.01,  

                            l2_leaf_reg=1,

                            subsample=0.7,

    #                        random_strength=2,

                            colsample_bylevel=0.7 ,

                            loss_function=lossfunc, 

                            od_type="Iter",

                           od_wait=100, 

                            border_count=250,

                           iterations=25000

                            )

    elif is_numeric_dtype(Y_train_Nan[IdxNoNan]) ==True:

        print("It is numeric")

        model=CatBoostRegressor(depth=7, 

                        learning_rate=0.01,  

                        l2_leaf_reg=1,

                        subsample=0.7,

    #                        random_strength=2,

                        colsample_bylevel=0.7 ,

                        loss_function='RMSE', 

                        od_type="Iter",

                       od_wait=100, 

                        border_count=250,

                       iterations=25000

                        )

    # 9 Fitting the model

    model.fit(X_train_train, Y_train_train,cat_features=cat_indexes,

              eval_set=(X_train_test,Y_train_test),

              plot=True)

import random



if gridSearch == False and TrainTestSklearn ==False and DataOwnSplit ==True:

    print("data own split")

    X_train2= X_train.loc[list(ids_train),:]

    X_val =  X_train.loc[list(ids_val),:]

    Y_train2 = Y_train[list(ids_train)]

    Y_val =  Y_train[list(ids_val)]

elif gridSearch == False and TrainTestSklearn ==True:

    print("train_test_used")

    X_train2, X_val, Y_train2, Y_val = train_test_split(X_train, Y_train, test_size=0.33, random_state=random_state)





else: 

    print("A copy is set")

    X_train2 = X_train.copy()

    Y_train2 = Y_train.copy()

    
grid = {'learning_rate': [0.03, 0.1],

        'l2_leaf_reg': [0.5, 1, 2, 3, 5, 7],

        "depth":[3,4,5,6,7,8,10,12] ,

        "subsample":[0.5,0.7,0.9],

        "colsample_bylevel": [0.5,0.7,0.9],

        #"loss_function":['RMSE'],

        "od_type":["Iter"],

        "od_wait":[100], 

        "border_count":[250],

        "iterations":[10000]

        

       }
if gridSearch == True:

    model=CatBoostRegressor(cat_features= cat_index,)

    grid_search_result = model.grid_search(grid, 

                                           X=X_train2, 

                                           y= Y_train2,

                                           train_size=0.66,

                                           partition_random_seed=2,

                                        cv=3,

                                        refit=True,

                                        shuffle=True,

                                           plot=True)
print(X_train2[num_cols].shape)

print(Y_train2[num_cols].shape)

print(X_val.shape)

print(Y_val.shape)



model=CatBoostRegressor(depth=7, 

                        learning_rate=0.01,  

                        l2_leaf_reg=1,

                        subsample=0.7,

#                        random_strength=2,

                        colsample_bylevel=0.7 ,

                        loss_function='RMSE', 

                        od_type="Iter",

                       od_wait=100, 

                        border_count=250,

                       iterations=25000

                        )

model.fit(X_train2, Y_train2,

          cat_features=cat_index,

          eval_set=(X_val,Y_val),

          plot=True)



# first seed is not good

if list(model.best_score_["validation"].values())[0] < 20000:

    RandomSeedListAppend.append(random_state)

    with open(DataName, "w") as file:

        file.write(str(RandomSeedListAppend))
print(random_state)


submission['SalePrice'] = model.predict(test) 
submission
submission.to_csv('submission.csv', index=False)




