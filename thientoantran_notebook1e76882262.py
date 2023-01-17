



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))





import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.linear_model import LinearRegression

from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor

from xgboost import XGBRegressor

from sklearn.svm import SVR



from sklearn.preprocessing import *



from sklearn.metrics import mean_squared_log_error



from sklearn.decomposition import PCA
df_train=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

df_test=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

df_submitssion = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
df_train_len = len(df_train)

def cleanUpNaN(inputDF):

    for col in inputDF.columns:

        countNaN = inputDF[col].isnull().sum(axis = 0)

        if (countNaN >= df_train_len * 0.7):

            print('dropping {} due to too many NaN'.format(col))

            inputDF.drop([col],axis=1,inplace=True)

        elif (countNaN > 0):

            print('filling NaN for column ' + col)

#             inputDF[col] = inputDF[col].fillna(inputDF[col].mode()[0])

            inputDF[col] = inputDF[col].fillna(0)
cleanUpNaN(df_train)
cleanUpNaN(df_test)
columns=['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood',

         'Condition2','BldgType','Condition1','HouseStyle','SaleType','SaleCondition','ExterCond',

         'ExterQual','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',

        'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Heating','HeatingQC',

         'CentralAir','Electrical','KitchenQual','Functional',

         'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive']
df_all = pd.concat([df_train, df_test], ignore_index=True, axis=0, sort=False)

df_all.drop(['Id'],axis=1,inplace=True)
def translate_categoricalData_to_columns(df_input, columns):

#     df_final=final_df

    df_extended = pd.DataFrame()

    df_output = df_input

    i=0

    for col in columns:

        df_dummies = pd.get_dummies(df_input[col],prefix= col)

        df_output.drop([col],axis=1,inplace=True)

        df_output = pd.concat([df_output,df_dummies],axis=1)



    return df_output
df_extended = translate_categoricalData_to_columns(df_all, columns)
# df_extended2 = df_extended.loc[:,~df_extended.columns.duplicated()]

df_extended2 = df_extended
df_extended.columns.to_list()
trainRecords = (int)(len(df_train)*.7)

df_extended_train = df_extended2.iloc[:trainRecords,:]

df_extended_validate = df_extended2.iloc[trainRecords:len(df_train),:]

df_extended_test = df_extended2.iloc[len(df_train):,:]



print(df_extended_train.info())

print(df_extended_validate.info())

print(df_extended_test.info())
df_extended_train_y_orig = df_extended_train['SalePrice'].copy()

df_extended_train_y = np.log(df_extended_train['SalePrice'].copy())

df_extended_train.drop(['SalePrice'], axis=1,inplace=True)
df_extended_train_y
df_extended_validate_y = np.log(df_extended_validate['SalePrice'].copy())

df_extended_validate.drop(['SalePrice'], axis=1,inplace=True)

df_extended_validate_y
df_extended_test.drop(['SalePrice'], axis=1,inplace=True)
# pipelines = []

# pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsRegressor())])))

# pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeRegressor())])))

# pipelines.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()),('GBM', GradientBoostingRegressor())])))

# pipelines.append(('ScaledXGB', Pipeline([('Scaler', StandardScaler()),('XGB', XGBRegressor())])))

# pipelines.append(('ScaledSVR', Pipeline([('Scaler', StandardScaler()),('SVR', SVR())])))



# pipelines.append(('ScaledPCAGBM', Pipeline([('Scaler', StandardScaler()),('PCA', PCA(.9)),('GBM', GradientBoostingRegressor())])))



# pipelines.append(('ScaledKNN', Pipeline([('Scaler', Norma()),('KNN', KNeighborsRegressor())])))

# pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeRegressor())])))

# pipelines.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()),('GBM', GradientBoostingRegressor())])))

# pipelines.append(('ScaledXGB', Pipeline([('Scaler', StandardScaler()),('XGB', XGBRegressor())])))

# pipelines.append(('ScaledSVR', Pipeline([('Scaler', StandardScaler()),('SVR', SVR())])))
scalerList = [StandardScaler(), Normalizer(), MinMaxScaler()]

pcaList = []

pcaRatios = [.5,.75,.9,.95]

for x in pcaRatios:

    pcaList.append(PCA(x))

regressorList = [KNeighborsRegressor(),DecisionTreeRegressor(),GradientBoostingRegressor(),SVR()]



pipelines = []

for scaler in scalerList:

    for regressor in regressorList:

        name = '\n'+str(scaler)+'\n'+ str(regressor)+'\n'

        pipelines.append((name, make_pipeline(scaler, regressor)))

for scaler in scalerList:

    for pca in pcaList:

        for regressor in regressorList:

            name = '\n'+str(scaler)+'\n'+ str(pca) +'\n'+ str(regressor)+'\n'

            pipelines.append((name, make_pipeline(scaler, pca, regressor)))



def fit_and_predict(model, train_x, train_y, x):

    model.fit(train_x, train_y)

    return model.predict(x)

def fit_predict_score_rmsle(model, train_x, train_y, x, y):

    predict = fit_and_predict(model, train_x, train_y, x)

    return model.score(train_x, train_y), np.sqrt(mean_squared_log_error(y, predict))
metrics = []



for name, model in pipelines:

    tmpScore, tmpRmsle = fit_predict_score_rmsle(model, df_extended_train, df_extended_train_y, df_extended_validate, df_extended_validate_y)

    metrics.append(tmpRmsle)

    

#     print('{}: \n\tscore = {}\n\tRMSLE= {}'.format(name, tmpScore, tmpRmsle))



index_leaseError = np.argmin(metrics)



# print('Best regressor is {} (index {}), mse = {}'.format(pipelines[index_bestMse][0], index_bestMse, mse[index_bestMse]))

print('Best regressor is {} (index {}), RMSLE = {}'.format(pipelines[index_leaseError][0], index_leaseError, metrics[index_leaseError]))
# index_leaseError=30
df_extended_train = df_extended2.iloc[:len(df_train),:]

df_extended_train_y = np.log(df_extended_train['SalePrice'].copy())

df_extended_train.drop(['SalePrice'], axis=1,inplace=True)



# df_extended_test.drop(['SalePrice'], axis=1,inplace=True)

df_extended_train_y
model_final = pipelines[index_leaseError][1]

predict = fit_and_predict(model_final, df_extended_train, df_extended_train_y, df_extended_test )

predict
df_predict = np.exp(pd.DataFrame(predict))

df_submitssion = pd.concat([df_submitssion['Id'],df_predict],axis=1)

df_submitssion.columns = ['Id','SalePrice']

df_submitssion.to_csv('submission.csv', index=False)
df_submitssion