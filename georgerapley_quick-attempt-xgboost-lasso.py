# -- NOTES ---

'''

DATASETS - 



- df                       -->   main dataset containing both eval and train data

- evaluationFeatures       -->   prepared eval data (i.e. 'test.csv' post clean/feature gen)



- trainingFeaturesFull     -->   prepared train data (i.e. 'train.csv' post clean/feature gen, pre-split)

- trainData                -->   random sampled trainingFeaturesFull data for training model

- testData                 -->   random sampled trainingFeaturesFull data for model validation



FUNCTIONS -



- LoadLabelAndUnion           -->   loads (csv), labels and unions raw train and test data

- DataCleaner()               -->   conducts cleaning (drops, fillNAs, anomalies etc)

- FeatureGenerator()          -->   generates features (mostly converts categorical vars to dummies)

- DropUnusedCols()            -->   drops list of cols from df (mostly catgorical vars)



- getPreds()                  -->   gets predictions from model

- getRMSLE()                  -->   gets Root Means Squared Log Error (RMSLE)

- getModelRMSLE()             -->   gets RMSLE for a given model

- getErrorScores()            -->   prints train and test RMSL error values

- getFeatureImportanceInfo()  -->   flexible function to output various degrees of feature importance info

- outputEvalPrediction()      -->   outputs evaluation predictions to CSV

'''

# -- IMPORTS & SETTINGS

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



# models

from xgboost import XGBRegressor

from sklearn import linear_model # LinearRegr, BayesianRidge, Lasso



from sklearn.cross_validation import train_test_split

from sklearn.metrics import mean_squared_error



pd.set_option('display.max_rows', 50)

pd.set_option('display.max_columns', 100)
# -- FUNCTIONS

categoricalColumns = ['Alley','BsmtCond','BsmtExposure','BsmtQual','BsmtFinType1','BsmtFinType2','Fence','FireplaceQu',

           'GarageCond','GarageFinish','GarageQual','GarageType','MasVnrType','MiscFeature','PoolQC',

           'BldgType','Condition1','Condition2','Electrical','ExterCond','ExterQual','Exterior1st',

           'Exterior2nd','Foundation','Functional','Heating','HeatingQC','HouseStyle','KitchenQual','LandContour',

           'LandSlope','LotConfig','LotShape','MSZoning','Neighborhood','PavedDrive','RoofMatl','RoofStyle',

           'SaleCondition','SaleType','Street','Utilities']



def getRMSLE(actual, predicted):

    return np.sqrt(mean_squared_error(np.log(actual), np.log(predicted)))



def getPreds(model, data, target):

    return model.predict(data.drop(target, axis=1))



def getModelRMSLE(model, data, target):

    preds = getPreds(model, data, target)

    actual = data[target]

    return np.sqrt(mean_squared_error(np.log(actual), np.log(preds)))



def LoadLabelAndUnionData(trainDataPath, evaluationDataPath):

    rawTrainDF = pd.read_csv(trainDataPath)

    rawTrainDF['Train'] = 1

    

    rawTestDF = pd.read_csv(evaluationDataPath)

    rawTestDF['Train'] = 0

    

    return rawTrainDF.append(rawTestDF)



def DataCleaner(df):

    

    # -- FILLNA = 'None'

    df.update(df[['Alley','BsmtCond','BsmtExposure','BsmtQual','BsmtFinType1','BsmtFinType2',

              'Fence','FireplaceQu','GarageCond','GarageFinish','GarageQual','GarageType',

              'MasVnrType','MiscFeature','PoolQC']].fillna('None'))

    

    # -- FILLNA = 0

    df.update(df[['BsmtFinSF1','BsmtFinSF2','BsmtFullBath','BsmtHalfBath','BsmtUnfSF',

              'GarageArea','GarageCars','LotFrontage','MasVnrArea','TotalBsmtSF']].fillna(0))



    # -- OTHER FILLNAs

    df['Electrical'] = df['Electrical'].fillna('SBrkr') # -- assume modal class

    df['Functional'] = df['Functional'].fillna('Typ') # -- assume 'Typical' (also modal class)

    df['KitchenQual'] = df['KitchenQual'].fillna('TA') # -- assume 'TA' (typical/average) (also modal class)

    df['MSZoning'] = df['MSZoning'].fillna('RL') # -- assume modal class

    df['SaleType'] = df['SaleType'].fillna('WD') # -- assume modal class

    df['Utilities'] = df['Utilities'].fillna('Unknown') # -- assume modal class



    # -- EXTERIOR 1st / 2nd

    df['Exterior1st'] = df['Exterior1st'].fillna('Other')

    df['Exterior2nd'] = df['Exterior2nd'].fillna('Other')

    

    # -- CENTRAL AIR

    df['CentralAir'] = df['CentralAir'].astype(str)

    df['CentralAir'] = np.where(df['CentralAir'] == 'Y', 1, 0)

    df['CentralAir'] = df['CentralAir'].astype(float)

    

    # -- GarageYrBlt

    df['GarageYrBlt'] = np.where(df['GarageYrBlt'] == 2207, df['YearBuilt'], df['GarageYrBlt'])

    df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['YearBuilt'])

    return df



def FeatureGenerator(df):

    # -- EXTERIOR 1st / 2nd

    df['Exterior1st'] = np.where(df['Exterior1st'].isin(['BrkComm','Stone','CBlock','AsphShn','ImStucc']),'Other',df['Exterior1st'])

    df['Exterior2nd'] = np.where(df['Exterior2nd'].isin(['Stone','CBlock','AsphShn']),'Other',df['Exterior1st'])

    

    # -- GET DUMMIES

    df = pd.concat([df, pd.get_dummies(df[categoricalColumns])], axis=1)  

    return df



def DropUnusedCols(df, cols):

    return df.drop(cols, axis=1)



def getErrorScores(model, trainDF, testDF, target):

    print ('TRAIN ERROR:', getModelRMSLE(model, trainDF.drop(['Id'],axis=1), target),

           ' | ', 

           'TEST ERROR:', getModelRMSLE(model, testDF.drop(['Id'],axis=1), target))



def getFeatureImportanceInfo(model, training_data, Y, figSize=(80,30), labelRot=90, plot=True, 

                             return_importance_data=False, start=0, end=100000, axisFontSize=50):

    feats = {}

    featImp = list(zip(training_data.drop(Y, axis=1).columns, model.feature_importances_))

    featImp = pd.DataFrame(featImp).sort_values(1, ascending=False)[start : min([end,len(featImp)+1])]

    featImp = featImp.rename(columns={0:'Feature', 1:'Importance'})

    featImp = featImp.set_index(featImp.Feature, drop=True)

    featImp = featImp.drop('Feature',axis=1)

    

    if plot:

        featImp.plot(kind='bar',rot=labelRot, figsize=figSize)

        plt.tick_params(axis='both', which='major', labelsize=axisFontSize)

        plt.show()

    if return_importance_data:

        return featImp.sort_values(by='Importance', ascending=False)

    

def getEnsembleData(data, primary_key, target):

    ensembleData = pd.DataFrame({

        str(primary_key) : data[primary_key], # PK

        str(target) : data[target], # target var

        'GradientBoostedTreePrediction' : getPreds(gbt, data.drop(primary_key,axis=1), target),

        'lassoPrediction' : getPreds(lasso, data.drop(primary_key,axis=1), target)

    })

    return ensembleData    



def outputEvalPredictions(model, data, output_filename):

    predictions__evaluation = pd.DataFrame(data['Id'])

    predictions__evaluation['SalePrice'] = model.predict(data.drop(['Id','SalePrice'], axis=1))

    predictions__evaluation.to_csv(output_filename, index=False)
# -- MAIN -- 

rawUniDF = LoadLabelAndUnionData("../input/train.csv", "../input/test.csv")



df = DataCleaner(rawUniDF) # -- cleans data

df = FeatureGenerator(df) # -- generates features & dummies



df = DropUnusedCols(df, categoricalColumns) # -- removes unused columns



# -- drop (dummy) cols not in eval data

df = DropUnusedCols(df, ['Exterior2nd_Other','HouseStyle_2.5Fin', 'RoofMatl_Membran', 'RoofMatl_Metal', 'RoofMatl_Roll',

                'Condition2_RRAe', 'Condition2_RRAn', 'Condition2_RRNn', 'GarageQual_Ex', 'PoolQC_Fa',

                'Heating_Floor', 'Heating_OthW', 'Electrical_Mix', 'MiscFeature_TenC'])



df.apply(pd.to_numeric) # -- converts all col dtypes to float or int



evaluationFeatures = df[df['Train'] == 0].drop('Train', axis=1) # -- separate back the evaluation dataset

trainingFeaturesFull = df[df['Train'] == 1].drop('Train', axis=1) # - prepped pre-split training data



trainData, testData = train_test_split(trainingFeaturesFull, train_size = 0.7, random_state = 20)
# 1 -- LASSO

lasso = linear_model.Lasso(alpha=20, fit_intercept=True, normalize=True, precompute=False, copy_X=True, 

                           max_iter=2000, tol=0.00001, warm_start=True, positive=False, random_state=None, 

                           selection='cyclic')



lasso.fit(trainData.drop(['SalePrice','Id'],axis=1), trainData['SalePrice'])



getErrorScores(lasso, trainData, testData, 'SalePrice') # BEST: 0.120221455665
# 2 -- GRADIENT BOOSTED TREE REGRESSOR

gbt = XGBRegressor(max_depth=4, learning_rate=0.05, n_estimators=900, objective='reg:linear',

                   booster='gbtree', n_jobs=-1, gamma=0, min_child_weight=3, max_delta_step=0, subsample=0.7)



gbt.fit(trainData.drop(['SalePrice','Id'],axis=1), trainData['SalePrice'])



getErrorScores(gbt, trainData, testData, 'SalePrice') # BEST: 0.109517
# -- JOIN OUTPUTS OF MODELS & GENERATE TEST/TRAIN SPLIT FOR ENSEMBLE

ensembleData = getEnsembleData(trainingFeaturesFull, 'Id', 'SalePrice')



ensembleTrainData, ensembleTestData = train_test_split(ensembleData, train_size = 0.75, random_state = None)



# -- ENSEMBLE (Linear Regr)

lnr = linear_model.LinearRegression(fit_intercept=True)

lnr.fit(ensembleTrainData.drop(['Id','SalePrice'], axis=1), ensembleTrainData['SalePrice'])



getErrorScores(lnr, ensembleTrainData, ensembleTestData, 'SalePrice') # BEST: 0.0296039
# -- FEATURE IMPORTANCE

getFeatureImportanceInfo(model=gbt, training_data=trainData.drop(['Id'],axis=1),

                         Y='SalePrice', figSize=(80,20), labelRot=90,

                         plot=True, return_importance_data=False,

                         start=0, end=50, axisFontSize = 60)
# -- ENSEMBLE to CSV

ensembleEvalData = getEnsembleData(evaluationFeatures, 'Id', 'SalePrice')

outputEvalPredictions(lnr, ensembleEvalData, 'xgb_lasso_ensemble_predictions__evaluation.csv')