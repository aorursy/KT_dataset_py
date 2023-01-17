import pandas as pd

import numpy as np

import xgboost as xgb

import sklearn.ensemble  as ens

from sklearn.metrics import median_absolute_error

from sklearn.metrics import mean_squared_log_error

from sklearn import preprocessing

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold

from scipy.special import inv_boxcox

from sklearn.compose import TransformedTargetRegressor

import matplotlib.pyplot as plt

import seaborn as sns

import lightgbm as lgb

from mlxtend.regressor import StackingCVRegressor

sns.set()

%matplotlib inline

import time



validPercent = 0.1      



dropLowImportantFeatures = False # drop low import feature (value ~ zero)



doesHyperparameter = False # test to find the best parameters



removeLowCorr2Price = False     # remove features if they have low correlation with Sale Price (<0.2) --> not good, don't apply

lowCorrValue = 0.2



removeHighCorr = True     #remove 1 of 2 features if they have high correlation each other(>0.8) --> ok and apply

highCorrValue = 0.8



removeMissingValue = True #remmove feature if they have too many missing values (>80% values = NA)

missingValuePercent = 0.8



dropOutlier = False      #drop outlier --> not good for this data, don't apply

coefSigma = 3 

dfTest = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

dfTrainAll = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

                    

print("Train data: "+str(dfTrainAll.shape))

print("Test data: "+str(dfTest.shape))

print(set(dfTrainAll.columns.values) - set(dfTest.columns.values))

print("---*********************----")

print(dfTrainAll.dtypes.value_counts())

dfTrainAll.head()

#print(dfTest.columns)

#split training data to training data and validate data

from sklearn.model_selection import train_test_split

if (validPercent>0):

    dfTrain, dfValid  = train_test_split(dfTrainAll, test_size = validPercent, random_state = 100)

else:

    dfTrain = dfTrainAll

    dfValid = dfTrainAll.copy(deep=True)
print("Train data: "+str(dfTrain.shape))

print("Valid data: "+str(dfValid.shape))

print("Test data: "+str(dfTest.shape))
lstFeature = pd.DataFrame([

  ['LotArea', 'Lot size in square feet'],

  ['TotalBsmtSF', 'Total square feet of basement area'],

  ['GarageCars', 'Size of garage in car capacity'],

  ['MSZoning', 'Identifies the general zoning classification of the sale'],

  ['Street', 'Type of road access to property'],

  ['LotFrontage', 'Linear feet of street connected to property'],

  ['Neighborhood', 'Physical locations within Ames city limits'],

  ['HouseStyle', 'Style of dwelling'],

  ['YearBuilt', 'Original construction date'],

],

  columns=['Column names','Description'

  ])

print(lstFeature)
def plotFeature(df,lstFeature):

    #plt.style.use('dark_background')

    fig, axes = plt.subplots(nrows= 3,ncols = 3, figsize=(24,17))

    for index,row in lstFeature.iterrows():

        j = index % 3

        i = index // 3

        axes[i,j].scatter(df[row['Column names']], df['SalePrice'], marker = 'x', color='red')

        axes[i,j].set_title(row['Description'])

        axes[i,j].set_xlabel(row['Column names'])

        axes[i,j].set_ylabel('Sale price')

    return True

plotFeature(dfTrain,lstFeature)   
plt.figure(figsize=(24,8))

plt.scatter(dfTrain['Neighborhood'], dfTrain['SalePrice'],color='red', alpha=1)

plt.show()
corrMatrix = dfTrain.corr() # for numerical feature values only

f, ax = plt.subplots(figsize=(30, 12))

sns.heatmap(corrMatrix, vmax=0.9, vmin=0.05, annot=True);
#find the high correlation feature to each other

for index,row in corrMatrix.iterrows():  #iterate through all dataframe rows

    sTemp = corrMatrix.loc[(corrMatrix[index]>highCorrValue) & (corrMatrix[index]<1),index] #filter all correlation values > 0.7 & < 1

    if (sTemp.shape[0]>0):

        print('~~~~~')

        for index1,value1 in sTemp.iteritems(): #interate through all series values

            print(" - "+index + " --- " + index1+" : "+str(value1))
#find the low correlation feature to SalePrice

index = 'SalePrice'

sTemp = corrMatrix.loc[(corrMatrix[index]<lowCorrValue) & (corrMatrix[index]>-lowCorrValue),index]

if (sTemp.shape[0]>0):

    lstLowCorr = [index1 for index1,value1 in sTemp.iteritems()]

    lstLowCorr.remove('Id')

    for index1,value1 in sTemp.iteritems(): #interate through all series values

        print(" - "+index + " --- " + index1+" : "+str(value1))
#sort by correlation value with house price (column 38)

corrMatrix['SalePrice'].sort_values(ascending = False).head(31)
#update the list of "important" feature and take a look at them

lstFeature = pd.DataFrame([

  ['OverallQual', 'Rates the overall material and finish of the house'],

  ['GarageArea', 'Size of garage in square feet'],

  ['TotalBsmtSF', 'Total rooms above grade (does not include bathrooms)'],

  ['FullBath', 'Full bathrooms above grade'],

  ['GrLivArea', 'Above grade (ground) living area square feet'],

  ['YearBuilt', 'Original construction date'],

  ['LotFrontage', 'Linear feet of street connected to property'],

  ['YearRemodAdd', 'Remodel date (same as construction date if no remodeling or additions)'],

],columns=['Column names','Description'])

#then re-plot to see values of some new "important" features

plotFeature(dfTrain,lstFeature)
fig, ax = plt.subplots(nrows= 1,ncols = 1, figsize=(10,5))

ax.hist(dfTrain['SalePrice'], bins=30)
fig, axes = plt.subplots(nrows= 3,ncols = 3, figsize=(24,17))

for index,row in lstFeature.iterrows():

    j = index % 3

    i = index // 3

    axes[i,j].hist(dfTrain[row['Column names']],bins=20)

    axes[i,j].set_title(row['Description'])
colTypes = dfTrain.dtypes

uniqueCount = dfTrain.nunique()

uniqueCountObject = uniqueCount[colTypes[colTypes == 'object'].index].sort_values(ascending = False)

print(uniqueCountObject.head(20))

#print(uniqueCount[colTypes[colTypes == 'int'].index].sort_values(ascending = False))
print(dfTrain.Neighborhood.value_counts().head(20))

#print(dfTrain.Exterior2nd.value_counts())

#print(dfTrain.Exterior1st.value_counts())

#print(dfTrain.SaleType.value_counts())

#print(dfTrain.Condition1.value_counts())
#Show all unique values of all categorical features --> use this informaiotn for aggregateion 

for colName in uniqueCountObject.index:

    print(colName +" --- " +str(dfTest[colName].unique()))
#too many NAN valuse should be removed

def findTopMissingValueFeature(dfTrain,dfTest,percentNA,NAOnly):

    if ('SalePrice' in dfTrain.columns):

        dfTotal = pd.concat([dfTrain.drop('SalePrice', axis=1), dfTest],axis=0) #drop remove column from dataframe, concat --> stack 2 dataframe

    else:

        dfTotal = pd.concat([dfTrain, dfTest],axis=0) #drop remove column from dataframe, concat --> stack 2 dataframe

    if (NAOnly):

        null_cols = dfTotal.isnull().sum()

    else:

        null_cols = dfTotal.isnull().sum() +(dfTotal==0).sum() #caculate all zero value

    print(len(null_cols[null_cols > 0])) #number of columns having NAN value

    print(null_cols[null_cols > 0].sort_values(ascending = False).head(20)) #sort list list by number of NAN value, DESC

    return (null_cols[null_cols > (percentNA*dfTotal.shape[0])].sort_values(ascending = False).index)



lstFeatureMissingValue = findTopMissingValueFeature(dfTrain,dfTest,missingValuePercent,True)

print(lstFeatureMissingValue)
dfTrain.describe().T.head(20)
descTrain = dfTrain.describe().T

#check for 3 sigma: mean - 3*std -- mean + 3*std

print("Features having values > mean + 3*std")

print(descTrain[descTrain['max'] > descTrain['mean'] + descTrain['std'] * 3].index)

print("Features having values < mean - 3*std")

print(descTrain[descTrain['min'] < descTrain['mean'] - descTrain['std'] * 3].index)
dfTrainFinal = dfTrain.copy(deep=True)

dfValidFinal = dfValid.copy(deep=True)

dfTestIDs = dfTest['Id']

dfValidIDs = dfValid['Id']

dfTestFinal =dfTest.copy(deep=True)

dfTrainFinal.drop(columns=['Id'], axis=1, inplace=True)

dfValidFinal.drop(columns=['Id'], axis=1, inplace=True)

dfTestFinal.drop(columns=['Id'], axis=1, inplace=True)

#remove High Correlation features

#coding optimization: move to 3.1 Feature Engineering/Aggregations



#remove feature having too many NAN (> 80%)

if (removeMissingValue):

    dfTrainFinal.drop(columns=lstFeatureMissingValue, axis=1, inplace=True)

    dfValidFinal.drop(columns=lstFeatureMissingValue, axis=1, inplace=True)

    dfTestFinal.drop(columns=lstFeatureMissingValue, axis=1, inplace=True)

    

print(dfTrainFinal.shape)

print(dfValidFinal.shape)

print(dfTestFinal.shape)

def dropColOutliers(df, colName, coef):

    mean = df[col].mean()

    std = df[col].std()

    df.drop(df[df[col] > (mean + coef * std)].index, inplace=True)

    df.drop(df[df[col] < (mean - coef * std)].index, inplace=True)

    return df
print(dfTrainFinal.shape)

print(dfValidFinal.shape)

print(dfTestFinal.shape)



#drop outlier for some features but RSMLE increase --> don't apply (data set is small)

if (dropOutlier):

    for col in dfTrainFinal.columns:

        if (dfTrainFinal.dtypes[col] != 'object'):

            dfTrainFinal = dropColOutliers(dfTrainFinal,col,coefSigma)



#dfTrainFinal = dfTrainFinal.drop(dfTrainFinal[(dfTrainFinal['GrLivArea']>4000) & (dfTrainFinal['SalePrice']<300000)].index)

#dfTrainFinal = dfTrainFinal.drop(dfTrainFinal[(dfTrainFinal['GrLivArea']>4000) & (dfTrainFinal['SalePrice']>700000)].index)



print(dfTrainFinal.shape)

print(dfTestFinal.shape)

plotFeature(dfTrainFinal,lstFeature)
lstNA = ['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','BsmtExposure','BsmtCond','BsmtQual','BsmtFinType1','BsmtFinType2','MSZoning']

lstZero =['LotFrontage','GarageYrBlt','GarageArea','GarageCars','TotalBsmtSF','MasVnrArea','BsmtHalfBath','BsmtFullBath','BsmtUnfSF','BsmtFinSF1','BsmtFinSF2']

lstMode =['KitchenQual','Exterior1st','Exterior2nd']

lstMean =[]

lstCols =['MasVnrType','Electrical','Functional','Utilities','SaleType']

lstColsText =['None','Mix','Typ','AllPub','Oth']

lstAllMode = dfTrainFinal.mode()
def fillNAValues(df):

    for col in lstNA:

        if col in df.columns:

            df[col].fillna('NA', inplace=True)

    for col in lstZero:

        if col in df.columns:

            df[col].fillna(0, inplace=True)

    for col in lstMode:

        if col in df.columns:

            df[col].fillna(lstAllMode[col][0], inplace=True)    

    for i in range(len(lstCols)):

        if col in df.columns:

            df[lstCols[i]].fillna(lstColsText[i], inplace=True) 
#fill NA values

fillNAValues(dfTrainFinal)

fillNAValues(dfValidFinal)

fillNAValues(dfTestFinal)
print(dfTrainFinal.isnull().any().any())

print(dfValidFinal.isnull().any().any())

print(dfTestFinal.isnull().any().any())
dfTarget = dfTrainFinal['SalePrice']

dfTargetValid = dfValidFinal['SalePrice']

dfTrainFinal.drop('SalePrice', axis=1, inplace=True)

dfValidFinal.drop('SalePrice', axis=1, inplace=True)



print(dfTrainFinal.isnull().any().any())

print(dfValidFinal.isnull().any().any())

print(dfTestFinal.isnull().any().any())
def aggregation(df):

    #map value for categorical features

    df['Neighborhood'] = df['Neighborhood'].map({'MeadowV': 0,'IDOTRR': 0,'BrDale': 0,'BrkSide': 0,'Edwards': 0,'OldTown': 0,'Sawyer': 0,'Blueste': 0,

        'SWISU': 0,'NPkVill': 0,'NAmes': 0,'Mitchel': 0,'SawyerW': 0,'NWAmes': 0,'Gilbert': 0,'Blmngtn': 0,'CollgCr': 0,'Crawfor': 1,'ClearCr': 1,

        'Somerst': 1,'Veenker': 1,'Timber': 1,'StoneBr': 2,'NridgHt': 2,'NoRidge': 2})

  

    df['MSZoning'] = df['MSZoning'].map({'NA': 0, 'C (all)': 0,'FV': 1,'I': 0,'RH': 1,'RL': 1,'RP': 1,'RM': 1})

      

    df['HouseStyle_stories'] = df['HouseStyle'].map({'1Story': 1,'1.5Fin': 1.5,'1.5Unf': 1.5,'2Story': 2,'2.5Fin': 2.5,'2.5Unf': 2.5,'SFoyer': 1.5,'SLvl': 1.5})



    df['HouseStyle_fin'] = df['HouseStyle'].map({'1Story': 1,'1.5Fin': 1,'1.5Unf': 0,'2Story': 1,'2.5Fin': 1,'2.5Unf': 0,'SFoyer': 1,'SLvl': 1})

    

    #combine feature

    df['TotFullBath'] = (df['BsmtFullBath'] + df['FullBath']*0.8)+(df['BsmtHalfBath'] + df['HalfBath']*0.8)*0.2

    df.drop(columns=['BsmtFullBath','FullBath','BsmtHalfBath','HalfBath'], axis=1, inplace=True)

    df['TotalBsmtSF'] = df['TotalBsmtSF'] + df['1stFlrSF']+ df['2ndFlrSF']

    

    df['Total_porch_sf'] = df['OpenPorchSF'] + df['3SsnPorch'] +  df['EnclosedPorch'] + df['ScreenPorch'] + df['WoodDeckSF']

    df = df.drop(['OpenPorchSF', '3SsnPorch', 'EnclosedPorch', 'ScreenPorch','WoodDeckSF'], axis=1)

    

    #df['YearBuilt'] = df['YearBuilt'] + df['GarageYrBlt']*0.3

    #df['GrLivArea'] = df['GrLivArea'] + df['TotRmsAbvGrd']*0.5

    #df['GarageArea'] = df['GarageArea'] + df['GarageCars']*100

    

    #remove High Correlation features

    if (removeHighCorr):

     #YearBuilt --- GarageYrBlt : 0.825228078796012

     #TotalBsmtSF --- 1stFlrSF : 0.821105896760658

     #GrLivArea --- TotRmsAbvGrd : 0.819525144369214

     #TotRmsAbvGrd --- GrLivArea : 0.819525144369214

     #GarageArea --- GarageCars : 0.8870237927614472

        df.drop(columns=['GarageYrBlt', '1stFlrSF', 'TotRmsAbvGrd', 'GarageCars','2ndFlrSF'], axis=1, inplace=True)
dfTestFinal['MSZoning'].unique()
aggregation(dfTrainFinal)

aggregation(dfTestFinal)

aggregation(dfValidFinal)

print(dfTrainFinal.isnull().any().any())

print(dfValidFinal.isnull().any().any())

print(dfTestFinal.isnull().any().any())
null_cols = dfTestFinal.isnull().sum() 

print(len(null_cols[null_cols > 0])) #number of columns having NAN value

null_cols[null_cols > 0].sort_values(ascending = False)
dfTrainFinal['Neighborhood'].unique()
# try to remove features having low correlation with SalePrice but RSMLE increase --> don't apply

if (removeLowCorr2Price):

    dfTrainFinal.drop(columns=lstLowCorr, axis=1, inplace=True)

    dfValidFinal.drop(columns=lstLowCorr, axis=1, inplace=True)

    dfTestFinal.drop(columns=lstLowCorr, axis=1, inplace=True)


print(dfTrainFinal.shape)

print(dfTestFinal.shape)

col_types = dfTrainFinal.dtypes

dfTrainFinal = pd.get_dummies(dfTrainFinal, columns=col_types[col_types == 'object'].index.values, drop_first=True)

dfValidFinal = pd.get_dummies(dfValidFinal, columns=col_types[col_types == 'object'].index.values, drop_first=True)

dfTestFinal = pd.get_dummies(dfTestFinal, columns=col_types[col_types == 'object'].index.values, drop_first=True)

print(dfTrainFinal.shape)

print(dfTestFinal.shape)
def adapt_columns(train_columns, df):

    # Add missing columns

    for column in train_columns:

        if (column not in df.columns):

            df[column] = 0



    # Delete columns that don't exist in train

    for column in df.columns:

        if (column not in train_columns):

            df.drop(column, axis=1, inplace=True)

    return df
adapt_columns(dfTrainFinal.columns, dfTestFinal)

adapt_columns(dfTrainFinal.columns, dfValidFinal)

print(dfTrainFinal.shape)

print(dfValidFinal.shape)

print(dfTestFinal.shape)
print(dfTrainFinal.dtypes.value_counts())

print(dfTrainFinal.isnull().any().any())

print(dfTestFinal.isnull().any().any())
def normalization(df):

    array_val = df.values 

    min_max_scaler = preprocessing.MinMaxScaler()

    array_norm = min_max_scaler.fit_transform(array_val)

    return pd.DataFrame(data=array_norm, columns=df.columns.values)
dfTrainFinal = normalization(dfTrainFinal)

dfValidFinal = normalization(dfValidFinal)

dfTestFinal = normalization(dfTestFinal)

print(dfTrainFinal.isnull().any().any())

print(dfValidFinal.isnull().any().any())

print(dfTestFinal.isnull().any().any())
xgb_params = {'n_estimators': 3000,'learning_rate': 0.01,'max_depth': 3, 'min_child_weight': 0,'gamma': 0.0,'colsample_bytree': 0.7,"colsample_bylevel": 0.5,'subsample': 0.7,'reg_alpha': 0.00006,'reg_lambda': 0.00006}
max_depth = range(1,15,2)

min_child_weight = range(1,10,2)

dfa1 = pd.DataFrame.from_records([(x,y) for x in max_depth for y in min_child_weight], columns = ['max_depth', 'min_child_weight'])



splits = 2

kf = KFold(n_splits=splits, shuffle=True, random_state=142)

def model_eval(max_depth, min_child_weight):

    mae = 0

    rmsle = 0

    xgb_params_temp = xgb_params.copy()

    xgb_params_temp['n_estimators'] = 500

    xgb_params_temp['learning_rate'] = 0.1

    xgb_params_temp['max_depth'] = max_depth

    xgb_params_temp['min_child_weight'] = min_child_weight

    model = xgb.XGBRegressor(params=xgb_params_temp)

    for train_index, test_index in kf.split(dfTrainFinal):

        X_train_k, X_test_k = dfTrainFinal.values[train_index], dfTrainFinal.values[test_index]

        y_train_k, y_test_k = dfTarget.values[train_index], dfTarget.values[test_index]          

        model.fit(X_train_k, y_train_k)

        y_pred_k = model.predict(X_test_k)

        np.round(y_pred_k)

        #mae = mae + median_absolute_error(y_test_k, y_pred_k)

        rmsle = rmsle + np.sqrt(mean_squared_log_error(y_test_k, y_pred_k))

    return (rmsle/splits)



if (doesHyperparameter):

    dfa1['rmsle'] = dfa1.apply(lambda x: model_eval(max_depth = int(x[0]), min_child_weight = x[1]), axis = 1).values

    #fig, ax = plt.subplots()

    #dfa1.groupby(by = 'max_depth').plot(x= 'min_child_weight', y = 'rmsle', ax = ax,  kind = 'line', figsize=(4,10) )

    #ax.legend(max_depth)

    print(dfa1.sort_values(by = 'rmsle', ascending=False).head(20))
reg_alpha = reg_lambda = [0, 0.001, 0.01, 0.1, 0.2, 0.4, 0.8, 1]

dfa2 = pd.DataFrame.from_records([(x,y) for x in reg_alpha for y in reg_lambda], columns = ['reg_alpha', 'reg_lambda'])



splits = 2

kf = KFold(n_splits=splits, shuffle=True, random_state=42)



def model_eval2(reg_alpha, reg_lambda):

    mae = 0

    rmsle = 0

    xgb_params_temp = xgb_params.copy()

    xgb_params_temp['n_estimators'] = 500

    xgb_params_temp['learning_rate'] = 0.1

    xgb_params_temp['reg_alpha'] = reg_alpha

    xgb_params_temp['reg_lambda'] = reg_lambda

    model = xgb.XGBRegressor(params=xgb_params_temp)

    for train_index, test_index in kf.split(dfTrainFinal):

        X_train_k, X_test_k = dfTrainFinal.values[train_index], dfTrainFinal.values[test_index]

        y_train_k, y_test_k = dfTarget.values[train_index], dfTarget.values[test_index]          

        model.fit(X_train_k, y_train_k)

        y_pred_k = model.predict(X_test_k)

        np.round(y_pred_k)

        #mae = mae + median_absolute_error(y_test_k, y_pred_k)

        rmsle = rmsle + np.sqrt(mean_squared_log_error(y_test_k, y_pred_k))

    return (rmsle/splits)

    

if (doesHyperparameter):

    dfa2['rmsle'] = dfa2.apply(lambda x: model_eval2(reg_alpha = x[0],reg_lambda = x[1]), axis = 1).values

    print(dfa2.sort_values(by = 'rmsle', ascending=False).head(20))
n_estimators = [200, 500, 800, 1000, 1500, 2000, 4000]

learning_rate = [0.1, 0.05, 0.01, 0.005, 0.001]

dfa5 = pd.DataFrame.from_records([(x,y) for x in n_estimators for y in learning_rate], columns = ['n_estimators', 'learning_rate'])



splits = 2

kf = KFold(n_splits=splits, shuffle=True, random_state=412)



def model_eval5(n_estimators, learning_rate ):

    mae = 0

    rmsle = 0

    xgb_params_temp = xgb_params.copy()

    xgb_params_temp['n_estimators'] = n_estimators

    xgb_params_temp['learning_rate'] = learning_rate

    model = xgb.XGBRegressor(params=xgb_params_temp)

    for train_index, test_index in kf.split(dfTrainFinal):

        X_train_k, X_test_k = dfTrainFinal.values[train_index], dfTrainFinal.values[test_index]

        y_train_k, y_test_k = dfTarget.values[train_index], dfTarget.values[test_index]          

        model.fit(X_train_k, y_train_k)

        y_pred_k = model.predict(X_test_k)

        np.round(y_pred_k)

        #mae = mae + median_absolute_error(y_test_k, y_pred_k)

        rmsle = rmsle + np.sqrt(mean_squared_log_error(y_test_k, y_pred_k))

    return (rmsle/splits)

    

if (doesHyperparameter):

    dfa5['rmsle'] = dfa5.apply(lambda x: model_eval5(n_estimators = x[0],learning_rate = x[1]), axis = 1).values

    print(dfa5.sort_values(by = 'rmsle', ascending=False).head(20))
X_train, X_test, y_train, y_test = train_test_split(dfTrainFinal, dfTarget, test_size=0.33, random_state=7)
from sklearn.metrics import accuracy_score

from sklearn.feature_selection import SelectFromModel



model = xgb.XGBRegressor(params=xgb_params)

model.fit(X_train, y_train)

# make predictions for test data and evaluate

y_pred = model.predict(X_test)

rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))

print("RMSLE: %.5f" % (rmsle))

thresholds = np.sort(model.feature_importances_)

thresholds = thresholds[thresholds>0]

print(thresholds)
for thresh in thresholds:

    # select features using threshold

    selection = SelectFromModel(model, threshold=thresh, prefit=True)

    select_X_train = selection.transform(X_train)

    # train model

    selection_model = xgb.XGBRegressor(params=xgb_params)

    selection_model.fit(select_X_train, y_train)

    # eval model

    select_X_test = selection.transform(X_test)

    y_pred = selection_model.predict(select_X_test)

    rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))

    print("Thresh=%.5f, n=%d, RMSLE: %.5f" % (thresh, select_X_train.shape[1], rmsle))
# plot feature importance

fig, ax = plt.subplots(figsize=(16, 14))

xgb.plot_importance(model,importance_type='weight',max_num_features=60,ax=ax).set_yticklabels(dfTrainFinal.columns)

plt.show()

print(model.feature_importances_)
mae = 0

rmsle = 0

splits = 5



kf = KFold(n_splits=splits, shuffle=True, random_state=42)

model = xgb.XGBRegressor(params=xgb_params)    



for train_index, test_index in kf.split(dfTrainFinal):

    X_train_k, X_test_k = dfTrainFinal.values[train_index], dfTrainFinal.values[test_index]

    y_train_k, y_test_k = dfTarget.values[train_index], dfTarget.values[test_index]      

    

    #model_k = xgb.XGBRegressor(params=xgb_params)    

    model.fit(X_train_k, y_train_k)

    y_pred_k = model.predict(X_test_k)

    

    np.round(y_pred_k)

    mae = mae + median_absolute_error(y_test_k, y_pred_k)

    rmsle = rmsle + np.sqrt(mean_squared_log_error(y_test_k, y_pred_k))



print('MAE: ' + '{:.2f}'.format(mae/splits)) 

print('Train data RMSLE: ' + '{:.4f}'.format(rmsle/splits)) 





if (validPercent>0):

    y_predValid = model.predict(dfValidFinal.values)

    mae = median_absolute_error(dfTargetValid.values, y_predValid)

    rmsle = np.sqrt(mean_squared_log_error(dfTargetValid.values, y_predValid))

    print('MAE: ' + '{:.2f}'.format(mae))

    print('Validate data RMSLE: ' + '{:.4f}'.format(rmsle))
if (False):

    mae = 0

    rmsle = 0

    splits = 10



    kf = KFold(n_splits=splits, shuffle=True, random_state=42)

    modelLGBM = lgb.LGBMRegressor(objective='regression', 

                                           num_leaves=4,

                                           learning_rate=0.01, 

                                           n_estimators=3000,

                                           max_bin=200, 

                                           bagging_fraction=0.75,

                                           bagging_freq=5, 

                                           bagging_seed=7,

                                           feature_fraction=0.2,

                                           feature_fraction_seed=7,

                                           verbose=-1,

                                           )

    for train_index, test_index in kf.split(dfTrainFinal):

        X_train_k, X_test_k = dfTrainFinal.values[train_index], dfTrainFinal.values[test_index]

        y_train_k, y_test_k = dfTarget.values[train_index], dfTarget.values[test_index]      



        #print(modelLGBM)

        ts = time.time()

        modelLGBM.fit(X_train_k, y_train_k)

        #print(time.time()-ts)

        y_pred_k = modelLGBM.predict(X_test_k)



        np.round(y_pred_k)

        mae = mae + median_absolute_error(y_test_k, y_pred_k)

        rmsle = rmsle + np.sqrt(mean_squared_log_error(y_test_k, y_pred_k))

        print('Train data RMSLE: ' + '{:.4f}'.format(rmsle)) 





    print('MAE: ' + '{:.2f}'.format(mae/splits)) 

    print('Train data RMSLE: ' + '{:.4f}'.format(rmsle/splits)) 

if (False):

    mae = 0

    rmsle = 0

    splits = 10



    kf = KFold(n_splits=splits, shuffle=True, random_state=42)

    modelGBR = ens.GradientBoostingRegressor(n_estimators=3000, learning_rate=0.01, max_depth=4, max_features='sqrt', min_samples_leaf=15, min_samples_split=10, loss='huber', random_state =42)

    for train_index, test_index in kf.split(dfTrainFinal):

        X_train_k, X_test_k = dfTrainFinal.values[train_index], dfTrainFinal.values[test_index]

        y_train_k, y_test_k = dfTarget.values[train_index], dfTarget.values[test_index]      



        #print(modelLGBM)

        ts = time.time()

        modelGBR.fit(X_train_k, y_train_k)

        #print(time.time()-ts)

        y_pred_k = modelGBR.predict(X_test_k)



        np.round(y_pred_k)

        mae = mae + median_absolute_error(y_test_k, y_pred_k)

        rmsle = rmsle + np.sqrt(mean_squared_log_error(y_test_k, y_pred_k))

        print('Train data RMSLE: ' + '{:.4f}'.format(rmsle)) 





    print('MAE: ' + '{:.2f}'.format(mae/splits)) 

    print('Train data RMSLE: ' + '{:.4f}'.format(rmsle/splits)) 

if (False):

    mae = 0

    rmsle = 0

    splits = 10



    kf = KFold(n_splits=splits, shuffle=True, random_state=42)

    modelStackCV = StackingCVRegressor(regressors=(model, modelLGBM,modelGBR),

                                    meta_regressor=model,

                                    use_features_in_secondary=True)

    for train_index, test_index in kf.split(dfTrainFinal):

        X_train_k, X_test_k = dfTrainFinal.values[train_index], dfTrainFinal.values[test_index]

        y_train_k, y_test_k = dfTarget.values[train_index], dfTarget.values[test_index]      



        #print(modelLGBM)

        ts = time.time()

        modelStackCV.fit(X_train_k, y_train_k)

        #print(time.time()-ts)

        y_pred_k = modelStackCV.predict(X_test_k)



        np.round(y_pred_k)

        mae = mae + median_absolute_error(y_test_k, y_pred_k)

        rmsle = rmsle + np.sqrt(mean_squared_log_error(y_test_k, y_pred_k))

        print('Train data RMSLE: ' + '{:.4f}'.format(rmsle)) 





    print('MAE: ' + '{:.2f}'.format(mae/splits)) 

    print('Train data RMSLE: ' + '{:.4f}'.format(rmsle/splits)) 

model = xgb.XGBRegressor(params=xgb_params)

model.fit(dfTrainFinal.values, dfTarget.values)

y_pred = model.predict(dfTestFinal.values)

if (dropLowImportantFeatures):

    selection = SelectFromModel(model, threshold= 0.0001, prefit=True)

    #show all droped feature

    print(dfTrainFinal.columns[selection.get_support()== False])

    select_X_train = selection.transform(dfTrainFinal)

    selection_model = xgb.XGBRegressor(params=xgb_params)

    selection_model.fit(select_X_train, dfTarget)

    select_X_test = selection.transform(dfTestFinal)

    y_pred = selection_model.predict(select_X_test)

    

if (False):

    modelLGBM.fit(dfTrainFinal.values, dfTarget.values)

    modelGBR.fit(dfTrainFinal.values, dfTarget.values)

    #modelStackCV.fit(dfTrainFinal.values, dfTarget.values)

    y_predLGBM = modelLGBM.predict(dfTestFinal.values)

    y_predGBR = modelGBR.predict(dfTestFinal.values)

    #y_predStack = modelStackCV.predict(dfTestFinal.values)

    #y_pred = y_pred*0.5 + y_predLGBM*0.5 + y_predGBR*0.0

    y_pred = y_predGBR



df_pred = pd.DataFrame(data=y_pred, columns=['SalePrice'])

df_pred = pd.concat([dfTestIDs, df_pred['SalePrice']], axis=1)

df_pred.SalePrice = df_pred.SalePrice.round(0)

df_pred.to_csv('submission.csv', sep=',', encoding='utf-8', index=False)
dfTest['SalePrice']=df_pred['SalePrice']

def plotFeatureTest(df,lstFeature):

    #plt.style.use('dark_background')

    fig, axes = plt.subplots(nrows= 3,ncols = 3, figsize=(24,17))

    for index,row in lstFeature.iterrows():

        j = index % 3

        i = index // 3

        axes[i,j].scatter(df[row['Column names']], df['SalePrice'], marker = 'x', color='red')

        axes[i,j].set_title(row['Description'])

        axes[i,j].set_xlabel(row['Column names'])

        axes[i,j].set_ylabel('Sale price')

    return True

plotFeatureTest(dfTest,lstFeature)  