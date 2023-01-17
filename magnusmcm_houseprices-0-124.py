import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import datetime
import seaborn as sns
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from scipy.stats import norm, skew 
from scipy.special import inv_boxcox
from scipy.stats import boxcox
from sklearn.decomposition import PCA
from math import ceil
from functools import reduce
import statistics as st

# xgboost libraries
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

learning_rate = 0.01 #0.01
training_epochs = 200000
display_step = 1000
keep_prob = 1
train_split_factor = 0.8
hidden_units_multiplier = 1 #2.0 
num_hidden_layers = 5
max_delay_between_best_epochs = 5000 #00
beta1=0.9 #0.9
beta2=0.999 #0.999
lambda_regularization_parameter=0.001 # 0.002
random_sample_test_data = False
max_train_clean_cost_in_best_epoch=99
max_dev_clean_min_cost_for_submission=8.39
round_cost_pass_threshold= 5.2
max_round_attempts = 15

hyperparameters = {"learning_rate": learning_rate , "training_epochs": training_epochs, "keep_prob": keep_prob,\
                   "hidden_units_multiplier": hidden_units_multiplier, "max_delay_between_best_epochs": max_delay_between_best_epochs,\
                   "lambda_regularization_parameter": lambda_regularization_parameter,"random_sample_test_data": random_sample_test_data, \
                   "num_hidden_layers": num_hidden_layers}

# Turn off TensorFlow warning messages in program output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
# RMSLE Error function
def tf_rmsle(y, y0):
    return tf.sqrt(tf.reduce_mean(tf.squared_difference(tf.log1p(y), tf.log1p(y0))))

# metric for evaluation
def rmsle(y_true, y_pred):    
    return np.sqrt(np.sum((np.log(y_true) - np.log(y_pred))**2)/len(y_pred))
#    return np.sqrt(np.sum((y_true - y_pred)**2)/len(y_pred))

train = pd.read_csv("../input/train.csv", skiprows=0)
test = pd.read_csv("../input/test.csv", skiprows=0)

features = [
    
    # Useless Features ==> "treatment": "ignore"
    { "col_name": "Id", "data_type": "int64", "treatment": "ignore", "na_treatment": "zero", "engineered_feature": False   },
    { "col_name": "Utilities", "data_type": "object", "treatment": "ignore", "na_treatment": "mode", "engineered_feature": False  },
    { "col_name": "RoofMatl", "data_type": "object", "treatment": "ignore", "na_treatment": "zero", "engineered_feature": False },
    { "col_name": "Heating", "data_type": "object", "treatment": "ignore", "na_treatment": "zero", "engineered_feature": False },    
    
    # Treatment: encode
    { "col_name": "ExterQual", "data_type": "object", "treatment": "encode", "na_treatment": "zero", "engineered_feature": False },
    { "col_name": "ExterCond", "data_type": "object", "treatment": "encode", "na_treatment": "zero", "engineered_feature": False },
    { "col_name": "BsmtQual", "data_type": "object", "treatment": "encode", "na_treatment": "None", "engineered_feature": False },
    { "col_name": "BsmtCond", "data_type": "object", "treatment": "encode", "na_treatment": "None", "engineered_feature": False },
    { "col_name": "BsmtExposure", "data_type": "object", "treatment": "encode", "na_treatment": "None", "engineered_feature": False },    
    { "col_name": "HeatingQC", "data_type": "object", "treatment": "encode", "na_treatment": "zero", "engineered_feature": False },
    { "col_name": "GarageQual", "data_type": "object", "treatment": "encode", "na_treatment": "None", "engineered_feature": False },    
    { "col_name": "GarageCond", "data_type": "object", "treatment": "encode", "na_treatment": "None", "engineered_feature": False },
    { "col_name": "PoolQC", "data_type": "object", "treatment": "encode", "na_treatment": "NoPool", "engineered_feature": False },
    { "col_name": "GarageFinish", "data_type": "object", "treatment": "encode", "na_treatment": "NoGarage", "engineered_feature": False },
    { "col_name": "Functional", "data_type": "object", "treatment": "encode", "na_treatment": "Typ", "engineered_feature": False },
    { "col_name": "PavedDrive", "data_type": "object", "treatment": "encode", "na_treatment": "mode", "engineered_feature": False },
    
    # These pairs can probably be grouped !
    { "col_name": "Exterior1st", "data_type": "object", "treatment": "one_hot", "na_treatment": "mode", "engineered_feature": False },
    { "col_name": "Exterior2nd", "data_type": "object", "treatment": "one_hot", "na_treatment": "mode", "engineered_feature": False },
    
    { "col_name": "Condition1", "data_type": "object", "treatment": "one_hot", "na_treatment": "zero", "engineered_feature": False },
    { "col_name": "Condition2", "data_type": "object", "treatment": "one_hot", "na_treatment": "zero", "engineered_feature": False },

    { "col_name": "Street", "data_type": "object", "treatment": "one_hot", "na_treatment": "zero", "engineered_feature": False },
    { "col_name": "Alley", "data_type": "object", "treatment": "one_hot", "na_treatment": "NoAccess", "engineered_feature": False }, # encode?
    
    # Ignored (Grouped into engineered features)
    { "col_name": "BsmtFinType1", "data_type": "object", "treatment": "ignore", "na_treatment": "None", "engineered_feature": False },
    { "col_name": "BsmtFinType2", "data_type": "object", "treatment": "ignore", "na_treatment": "None", "engineered_feature": False },
    { "col_name": "BsmtFinSF1", "data_type": "int64", "treatment": "input_col", "na_treatment": "zero", "engineered_feature": False },
    { "col_name": "BsmtFinSF2", "data_type": "int64", "treatment": "input_col", "na_treatment": "zero", "engineered_feature": False },
    { "col_name": "2ndFlrSF", "data_type": "int64", "treatment": "input_col", "na_treatment": "zero", "engineered_feature": False },
    { "col_name": "1stFlrSF", "data_type": "int64", "treatment": "input_col", "na_treatment": "median", "engineered_feature": False},
 
    { "col_name": "GarageArea", "data_type": "int64", "treatment": "input_col", "na_treatment": "zero", "engineered_feature": False},
    { "col_name": "GrLivArea", "data_type": "int64", "treatment": "input_col", "na_treatment": "zero", "engineered_feature": False},
    { "col_name": "TotalBsmtSF", "data_type": "int64", "treatment": "input_col", "na_treatment": "zero", "engineered_feature": False},
    { "col_name": "EnclosedPorch", "data_type": "int64", "treatment": "input_col", "na_treatment": "zero", "engineered_feature": False },
    { "col_name": "ScreenPorch", "data_type": "int64", "treatment": "input_col", "na_treatment": "zero", "engineered_feature": False },
    
    {   "col_name": "BsmtGLQSF", "data_type": "int64", "treatment": "ignore", "na_treatment": "zero", "engineered_feature": True},
    {   "col_name": "BsmtALQSF", "data_type": "int64", "treatment": "ignore", "na_treatment": "zero", "engineered_feature": True},
    {   "col_name": "BsmtBLQSF", "data_type": "int64", "treatment": "ignore", "na_treatment": "zero", "engineered_feature": True},
    {   "col_name": "BsmtRecSF", "data_type": "int64", "treatment": "ignore", "na_treatment": "zero", "engineered_feature": True},
    {   "col_name": "BsmtLwQSF", "data_type": "int64", "treatment": "ignore", "na_treatment": "zero", "engineered_feature": True},
    { "col_name": "Neighborhood", "data_type": "object", "treatment": "one_hot", "na_treatment": "zero", "engineered_feature": False },
 
    # Ignored (Grouped into engineered features)
    { "col_name": "YrSold", "data_type": "int64", "treatment": "ignore", "na_treatment": "mode", "engineered_feature": False },
    { "col_name": "MoSold", "data_type": "int64", "treatment": "ignore", "na_treatment": "mode", "engineered_feature": False },
    
    # Selected Categorical Features ==> "treatment": "one_hot"
    { "col_name": "MasVnrType", "data_type": "object", "treatment": "one_hot", "na_treatment": "None", "engineered_feature": False },
    { "col_name": "LotShape", "data_type": "object", "treatment": "one_hot", "na_treatment": "zero", "engineered_feature": False },
    { "col_name": "LandContour", "data_type": "object", "treatment": "one_hot", "na_treatment": "zero", "engineered_feature": False },
    { "col_name": "LandSlope", "data_type": "object", "treatment": "one_hot", "na_treatment": "zero", "engineered_feature": False },
    { "col_name": "BldgType", "data_type": "object", "treatment": "one_hot", "na_treatment": "zero", "engineered_feature": False },
    { "col_name": "FireplaceQu", "data_type": "object", "treatment": "encode", "na_treatment": "NoFireplace", "engineered_feature": False },
    { "col_name": "GarageType", "data_type": "object", "treatment": "one_hot", "na_treatment": "None", "engineered_feature": False },
    { "col_name": "KitchenQual", "data_type": "object", "treatment": "one_hot", "na_treatment": "mode", "engineered_feature": False },
    { "col_name": "CentralAir", "data_type": "object", "treatment": "one_hot", "na_treatment": "zero", "engineered_feature": False },
    { "col_name": "Electrical", "data_type": "object", "treatment": "one_hot", "na_treatment": "mode", "engineered_feature": False },
    { "col_name": "LotConfig", "data_type": "object", "treatment": "one_hot", "na_treatment": "zero", "engineered_feature": False },
    { "col_name": "MSSubClass", "data_type": "int64", "treatment": "one_hot", "na_treatment": "None", "engineered_feature": False },  #None? # Al tanto que és un string encara q sembli numèric
    { "col_name": "MSZoning", "data_type": "object", "treatment": "one_hot", "na_treatment": "mode", "engineered_feature": False },
    { "col_name": "HouseStyle", "data_type": "object", "treatment": "one_hot", "na_treatment": "zero", "engineered_feature": False },
    { "col_name": "RoofStyle", "data_type": "object", "treatment": "one_hot", "na_treatment": "zero", "engineered_feature": False },
    { "col_name": "Foundation", "data_type": "object", "treatment": "one_hot", "na_treatment": "zero", "engineered_feature": False },
    { "col_name": "Fence", "data_type": "object", "treatment": "one_hot", "na_treatment": "None", "engineered_feature": False },
    { "col_name": "MiscFeature", "data_type": "object", "treatment": "one_hot", "na_treatment": "None", "engineered_feature": False },
    { "col_name": "SaleType", "data_type": "object", "treatment": "one_hot", "na_treatment": "mode", "engineered_feature": False },
    { "col_name": "SaleCondition", "data_type": "object", "treatment": "one_hot", "na_treatment": "zero", "engineered_feature": False },
    
    # Selected Numeric Features with correl < 0.4 ==> "treatment": "input_col"
    
    { "col_name": "BsmtFullBath", "data_type": "int64", "treatment": "input_col", "na_treatment": "zero", "engineered_feature": False },
    { "col_name": "BsmtHalfBath", "data_type": "int64", "treatment": "input_col", "na_treatment": "zero", "engineered_feature": False },
    { "col_name": "KitchenAbvGr", "data_type": "int64", "treatment": "input_col", "na_treatment": "zero", "engineered_feature": False },
    { "col_name": "LowQualFinSF", "data_type": "int64", "treatment": "input_col", "na_treatment": "zero", "engineered_feature": False },    
    { "col_name": "WoodDeckSF", "data_type": "int64", "treatment": "input_col", "na_treatment": "zero", "engineered_feature": False },
    { "col_name": "OpenPorchSF", "data_type": "int64", "treatment": "input_col", "na_treatment": "zero", "engineered_feature": False },
    { "col_name": "LotArea", "data_type": "int64", "treatment": "input_col", "na_treatment": "zero", "engineered_feature": False },
    { "col_name": "OverallCond", "data_type": "int64", "treatment": "input_col", "na_treatment": "zero", "engineered_feature": False },
    { "col_name": "BsmtUnfSF", "data_type": "int64", "treatment": "input_col", "na_treatment": "zero", "engineered_feature": False },
    { "col_name": "BedroomAbvGr", "data_type": "int64", "treatment": "input_col", "na_treatment": "zero", "engineered_feature": False },
    { "col_name": "LotFrontage", "data_type": "float64", "treatment": "input_col", "na_treatment": "median", "engineered_feature": False },
    { "col_name": "PoolArea", "data_type": "int64", "treatment": "input_col", "na_treatment": "zero", "engineered_feature": False },
    { "col_name": "MiscVal", "data_type": "int64", "treatment": "input_col", "na_treatment": "zero", "engineered_feature": False },
    { "col_name": "3SsnPorch", "data_type": "int64", "treatment": "input_col", "na_treatment": "zero", "engineered_feature": False },
    { "col_name": "GarageYrBlt", "data_type": "float64", "treatment": "input_col", "na_treatment": "median", "engineered_feature": False },
    { "col_name": "HalfBath", "data_type": "int64", "treatment": "input_col", "na_treatment": "zero", "engineered_feature": False },
    
    # Numeric Features with correl >= 0.4 (Very important features) ==> "treatment": "input_col"
    {   "col_name": "OverallQual", "data_type": "int64", "treatment": "input_col", "na_treatment": "median", "engineered_feature": False},
    {   "col_name": "YearBuilt", "data_type": "int64", "treatment": "input_col", "na_treatment": "median", "engineered_feature": False},
    {   "col_name": "YearRemodAdd", "data_type": "int64", "treatment": "input_col", "na_treatment": "zero", "engineered_feature": False}, #YearRemodAdd: Remodel date (same as construction date if no remodeling or additions)
    {   "col_name": "MasVnrArea", "data_type": "int64", "treatment": "input_col", "na_treatment": "zero", "engineered_feature": False},
    {   "col_name": "FullBath", "data_type": "int64", "treatment": "input_col", "na_treatment": "mode", "engineered_feature": False},
    {   "col_name": "TotRmsAbvGrd", "data_type": "int64", "treatment": "input_col", "na_treatment": "zero", "engineered_feature": False},    
    {   "col_name": "Fireplaces", "data_type": "int64", "treatment": "input_col", "na_treatment": "zero", "engineered_feature": False},
    {   "col_name": "GarageCars", "data_type": "int64", "treatment": "input_col", "na_treatment": "zero", "engineered_feature": False},
    
    # Engineered features
    {   "col_name": "haspool", "data_type": "int64", "treatment": "input_col", "na_treatment": "zero", "engineered_feature": True},  
    {   "col_name": "hasgarage", "data_type": "int64", "treatment": "input_col", "na_treatment": "zero", "engineered_feature": True},
    {   "col_name": "hasbsmt", "data_type": "int64", "treatment": "input_col", "na_treatment": "zero", "engineered_feature": True},
    {   "col_name": "hasfireplace", "data_type": "int64", "treatment": "input_col", "na_treatment": "zero", "engineered_feature": True},
    {   "col_name": "YrMoSold", "data_type": "int64", "treatment": "input_col", "na_treatment": "zero", "engineered_feature": True},
    {   "col_name": "TotalHouseSF", "data_type": "int64", "treatment": "input_col", "na_treatment": "zero", "engineered_feature": True},
    {   "col_name": "TotalAreaSF", "data_type": "int64", "treatment": "input_col", "na_treatment": "zero", "engineered_feature": True},
    {   "col_name": "DolPerFeetNeigLiv", "data_type": "int64", "treatment": "input_col", "na_treatment": "zero", "engineered_feature": True}, 
    {   "col_name": "DolPerFeetNeigLot", "data_type": "int64", "treatment": "input_col", "na_treatment": "zero", "engineered_feature": True},

    {   "col_name": "LivingAreaSF", "data_type": "int64", "treatment": "input_col", "na_treatment": "zero", "engineered_feature": True},
    {   "col_name": "StorageAreaSF", "data_type": "int64", "treatment": "input_col", "na_treatment": "zero", "engineered_feature": True}
    
    #    {   "col_name": "Exterior1stOr2nd", "data_type": "int64", "treatment": "input_col", "na_treatment": "zero", "engineered_feature": True},
    #    {   "col_name": "Condition1Or2", "data_type": "int64", "treatment": "input_col", "na_treatment": "zero", "engineered_feature": True}
]

#Drop Outliers.
train = train[train.GrLivArea < 4500]
train = train[train.TotalBsmtSF <= 5000]
train = train.drop(train[(train['1stFlrSF']>4000)].index)

my_datasets = [train, test]

#NA Treatment
for my_ds in my_datasets:
    my_ds['MSSubClass'] = my_ds['MSSubClass'].apply(lambda x: str(x))
    for fea in features:
        if (fea['engineered_feature']==False):
            if (fea['na_treatment']=="zero"):
                my_ds[fea['col_name']] = my_ds[fea['col_name']].fillna(0)

            if (fea['na_treatment']=="None"):
                my_ds[fea['col_name']] = my_ds[fea['col_name']].fillna('None')

            if (fea['na_treatment']=="mode"):
                my_ds[fea['col_name']] = my_ds[fea['col_name']].fillna(train[fea['col_name']].mode()[0])

            if (fea['na_treatment']=="median"):
                my_ds[fea['col_name']] = my_ds[fea['col_name']].fillna(train[fea['col_name']].median())

            #For PoolQC
            if (fea['na_treatment']=="NoPool"):
                my_ds[fea['col_name']] = my_ds[fea['col_name']].fillna("NoPool")

            #For Alley
            if (fea['na_treatment']=="NoAccess"):
                my_ds[fea['col_name']] =my_ds[fea['col_name']].fillna("NoAccess")      
            
            #FireplaceQu
            if (fea['na_treatment']=="NoFireplace"):
                my_ds[fea['col_name']] =my_ds[fea['col_name']].fillna("NoFireplace")             
    
            #GarageFinish
            if (fea['na_treatment']=="NoGarage"):
                my_ds[fea['col_name']] =my_ds[fea['col_name']].fillna("NoGarage") 
    
            #Functional
            if (fea['na_treatment']=="Typ"):
                my_ds[fea['col_name']] =my_ds[fea['col_name']].fillna("Typ") 

    #Feature Engineering
    my_ds['haspool'] = my_ds['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
    my_ds['hasgarage'] = my_ds['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
    my_ds['hasbsmt'] = my_ds['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
    my_ds['hasfireplace'] = my_ds['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
    my_ds['YrMoSold'] = my_ds['YrSold'] * 100 + my_ds['MoSold']
    my_ds['TotalHouseSF'] = my_ds['TotalBsmtSF'] + my_ds['1stFlrSF'] + my_ds['2ndFlrSF'] # Also Known As AllFlrs_SF
    my_ds['TotalAreaSF'] = my_ds['GrLivArea']+ my_ds['TotalBsmtSF']+ my_ds['GarageArea']+ my_ds['EnclosedPorch']+ my_ds['ScreenPorch']

    for col in ['BsmtGLQ','BsmtALQ','BsmtBLQ','BsmtRec','BsmtLwQ']:
        my_ds[col+'SF'] = 0
    
    # fill remaining finish type columns
    for row in my_ds.index:
        fin1 = my_ds.loc[row,'BsmtFinType1']
        if (fin1!='None') and (fin1!='Unf'):
            my_ds.loc[row,'Bsmt'+fin1+'SF'] += my_ds.loc[row,'BsmtFinSF1']

        fin2 = my_ds.loc[row,'BsmtFinType2']
        if (fin2!='None') and (fin2!='Unf'):
            my_ds.loc[row,'Bsmt'+fin2+'SF'] += my_ds.loc[row,'BsmtFinSF2']
        
    # remove initial BsmtFin columns
    #my_ds.drop(['BsmtFinType1','BsmtFinSF1','BsmtFinType2','BsmtFinSF2'], axis=1, inplace=True)

    my_ds['LivingAreaSF'] = my_ds['1stFlrSF'] + my_ds['2ndFlrSF'] + my_ds['BsmtGLQSF'] + my_ds['BsmtALQSF'] + my_ds['BsmtBLQSF']
    my_ds['StorageAreaSF'] = my_ds['LowQualFinSF'] + my_ds['BsmtRecSF'] + my_ds['BsmtLwQSF'] + my_ds['BsmtUnfSF'] + my_ds['GarageArea']

#More feature engineering: Price per feet (1. Living Area 2. Lot)
#1. cost of 1 square feet of living area per house by Neighborhood groups
#Its a kind of mean encoding (or likehood encoding)
train['DolPerFeetLiv'] = train['SalePrice']/train['GrLivArea']
temp_data = pd.concat([train['Neighborhood'], train['DolPerFeetLiv']], axis=1)
cost_per_district = temp_data.groupby('Neighborhood')['DolPerFeetLiv'].mean()
test['DolPerFeetNeigLiv'] = test['Neighborhood'].apply(lambda x: cost_per_district[x])
train['DolPerFeetNeigLiv'] = train['Neighborhood'].apply(lambda x: cost_per_district[x])

#2. cost of 1 square feet of lot area per house by Neighborhood groups
train['DolPerFeetLot'] = train['SalePrice']/train['LotArea']
temp_data = pd.concat([train['Neighborhood'], train['DolPerFeetLot']], axis=1)
cost_per_district = temp_data.groupby('Neighborhood')['DolPerFeetLot'].mean()
test['DolPerFeetNeigLot'] = test['Neighborhood'].apply(lambda x: cost_per_district[x])
train['DolPerFeetNeigLot'] = train['Neighborhood'].apply(lambda x: cost_per_district[x])
for fea in features:
    if (fea["treatment"]=="encode"):
        for my_dataset in my_datasets:            
            my_dataset[fea['col_name']].replace(['Po', 'Fa', 'TA', 'Gd', 'Ex', 'None',
                                                 'No', 'Mn', 'Av', 'Gd',
                                                 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ',
                                                 'Unf', 'RFn', 'Fin',
                                                 'CarPort', 'Basment', 'Detchd', '2Types', 'Basement', 'Attchd', 'BuiltIn',
                                                 'Sal','Sev','Maj2','Maj1','Mod','Min2','Min1','Typ',
                                                 'Ex','Gd','TA','Fa','NoPool',
                                                 'NoFireplace',
                                                 'NoGarage','Unf','RFn','Fin'
                                                ], [1,2,3,4,5,0,
                                                    1,2,3,4,
                                                    1,2,3,4,5,6,
                                                    1,2,3,
                                                    1, 2, 3, 4, 5, 6, 7,
                                                    0, 1, 1, 2, 5, 8, 9, 10,
                                                    5,4,3,2,0,
                                                    0,
                                                    0, 1, 2, 3
                                                   ], inplace=True)

my_df_train= pd.DataFrame()
my_df_test = pd.DataFrame()

for fea in features:
    if (fea["treatment"]=="one_hot"):
        my_df_train=pd.concat([my_df_train,pd.get_dummies(train[fea["col_name"]], prefix=fea["col_name"])], axis=1)
        my_df_test=pd.concat([my_df_test,pd.get_dummies(test[fea["col_name"]], prefix=fea["col_name"])], axis=1)

my_df_train_features = list(my_df_train)
my_df_test_features = list(my_df_test)
diff_test_train = set(list(set(my_df_test_features) - set(my_df_train_features)))
diff_train_test = set(list(set(my_df_train_features) - set(my_df_test_features)))

# Add all the columns that are not common on all 3 dfs
for missing_column in diff_test_train:
    my_df_test=my_df_test.drop([missing_column], axis=1)

for missing_column in diff_train_test:
    my_df_train=my_df_train.drop([missing_column], axis=1)

for fea in features:
    if (fea["treatment"]=="input_col"):
        my_df_train=pd.concat([my_df_train,train[fea["col_name"]]], axis=1)
        my_df_test=pd.concat([my_df_test,test[fea["col_name"]]], axis=1)

my_df_train=pd.concat([my_df_train,train["SalePrice"]], axis=1)
my_datasets = [my_df_train, my_df_test]

for myd in my_datasets:    
    myd = myd.sort_index(axis=1, inplace=True) 
# Skewness treatment
all_data = pd.concat([my_df_train, my_df_test]).reset_index(drop=True)
for fea in features:
    if ((fea["data_type"] != "object") & (fea["treatment"] == "input_col") ):
        if (fea["col_name"]!="SalePrice"):
            all_data[fea["col_name"]],lambdita = boxcox(all_data[fea["col_name"]].values+1)

my_df_test = all_data.iloc[my_df_train.shape[0]:my_df_test.shape[0]+my_df_train.shape[0]]
my_df_train = all_data.iloc[0:my_df_train.shape[0]]
my_df_dev = pd.DataFrame()
my_df_train["SalePrice"],lambdaY = boxcox(my_df_train["SalePrice"].values)

#if (random_sample_test_data==True):
#    train_part = train.sample(frac=train_split_factor, replace=False)
#    dev_part = train[~train.index.isin(train_part.index)]
#else:
if (1==1):
    train_size = my_df_train.shape[0]
    print ("train_size:", train_size)

    train_part_size = int(train_size*train_split_factor)
    dev_part_size = train_size-train_part_size

    print ("train_part_size:", train_part_size)
    print ("dev_part_size:", dev_part_size)    

    my_df_dev = my_df_train.iloc[train_part_size:train_size]
    my_df_train = my_df_train.iloc[0:train_part_size]
# Create grouped engineered features
#    {   "col_name": "Exterior1stOr2nd", "data_type": "int64", "treatment": "input_col", "na_treatment": "zero", "engineered_feature": True},
#    {   "col_name": "Condition1Or2", "data_type": "int64", "treatment": "input_col", "na_treatment": "zero", "engineered_feature": True}

Y_scaled_training = my_df_train["SalePrice"].to_frame()
Y_scaled_dev = my_df_dev["SalePrice"].to_frame()

my_df_train=my_df_train.drop(["SalePrice"], axis=1)
my_df_dev=my_df_dev.drop(["SalePrice"], axis=1)
my_df_test=my_df_test.drop(["SalePrice"], axis=1)

#X_scaler = MinMaxScaler(feature_range=(0, 1))

# Scale both the training inputs and outputs
X_scaled_training = my_df_train #X_scaler.fit_transform(my_df_train)
X_scaled_dev = my_df_dev # X_scaler.transform(my_df_dev)
X_scaled_test = my_df_test #X_scaler.transform(my_df_test)
my_df_train=pd.concat([pd.DataFrame(X_scaled_training),pd.DataFrame(X_scaled_dev)])
my_df_train_Y=pd.concat([Y_scaled_training,Y_scaled_dev])
my_df_train["SalePrice"] = my_df_train_Y["SalePrice"]
my_df_train = my_df_train.sample(my_df_train.shape[0])
number_of_inputs = X_scaled_training.shape[1]
number_of_outputs = 1
layer_nodes = int(number_of_inputs * hidden_units_multiplier)
tf.reset_default_graph()

with tf.variable_scope('input', reuse=True):
    X = tf.placeholder(tf.float32, shape=(None, number_of_inputs))

layer_output=[]
weights=[]
biases=[]

for i in range(num_hidden_layers):
    if (i==0):
        inps = number_of_inputs
        inpu = X
    else:
        inps = layer_nodes
        inpu = layer_output[i-1]

    with tf.variable_scope('layer_{0:1d}'.format(i+1)):
        weights.append(tf.get_variable(name="weights_{0:1d}".format(i+1), shape=[inps, layer_nodes], initializer=tf.initializers.he_normal(seed=None)))#tf.contrib.layers.xavier_initializer())
        biases.append(tf.get_variable(name="biases_{0:1d}".format(i+1), shape=[layer_nodes], initializer=tf.zeros_initializer()))
        layer_output.append(tf.nn.dropout(tf.nn.relu(tf.matmul(inpu, weights[i]) + biases[i]), keep_prob))

with tf.variable_scope('output'):
    weights0 = tf.get_variable(name="weights_0", shape=[layer_nodes, number_of_outputs], initializer=tf.initializers.he_normal(seed=None))#=tf.contrib.layers.xavier_initializer())
    biases0 = tf.get_variable(name="biases_0", shape=[number_of_outputs], initializer=tf.zeros_initializer())
    prediction = tf.nn.relu(tf.matmul(layer_output[num_hidden_layers-1], weights0) + biases0)


with tf.variable_scope('cost_regu'):
    cost_regu =  tf.nn.l2_loss(weights0)
    for i in range(num_hidden_layers):
        cost_regu=tf.add(cost_regu, tf.nn.l2_loss(weights[i]))
    cost_regu = tf.multiply(cost_regu, lambda_regularization_parameter)

with tf.variable_scope('cost'):
    Y = tf.placeholder(tf.float32, shape=(None,1))
    cost_pred =  tf_rmsle(prediction, Y)

    reg_ws =  tf.nn.l2_loss(weights0)
    
    for i in range(num_hidden_layers):
        reg_ws=tf.add(reg_ws, tf.nn.l2_loss(weights[i]))

    cost = tf.add(cost_pred,tf.multiply(reg_ws,lambda_regularization_parameter))


with tf.variable_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate, beta1=beta1, beta2=beta2).minimize(cost)
n = ceil(my_df_train.shape[0] * (1-train_split_factor))
list_df = [my_df_train[i:i+n] for i in range(0,my_df_train.shape[0],n)]
num_rounds = len(list_df)
round_dev_min_clean_costs = []
Y_predicted_sum =pd.DataFrame(pd.np.zeros(( X_scaled_test.shape[0], 1)), columns=['SalePrice'])
Y_predicted_train_list = pd.DataFrame(pd.np.zeros((X_scaled_test.shape[0],3)), columns=['SalePrice', 'Max', 'Min'])

print ("HyperParams: ", hyperparameters)

for i in range(num_rounds):
    print("Round: ", i+1, "of", num_rounds)

    X_scaled_dev = list_df[i]
    Y_scaled_dev = X_scaled_dev["SalePrice"].to_frame()
    X_scaled_dev = X_scaled_dev.drop(["SalePrice"], axis=1)
    X_scaled_training = pd.DataFrame()
    for j in range(num_rounds):
        if (j != i):
            X_scaled_training= pd.concat([X_scaled_training, list_df[j]]) #.reset_index(drop=True)    
    Y_scaled_training = X_scaled_training["SalePrice"].to_frame()
    X_scaled_training = X_scaled_training.drop(["SalePrice"], axis=1)

    X_scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled_training = X_scaler.fit_transform(X_scaled_training)
    X_scaled_dev = X_scaler.transform(X_scaled_dev)
    X_scaled_test = X_scaler.transform(my_df_test)

    attempt_cost = 0.099
    dev_cost_min = 0.099
    dev_cost_clean_min_round = 0.099
            
    k=0
    while ((attempt_cost*1000 > round_cost_pass_threshold)&(k+1<=max_round_attempts)):
        k+=1
        dev_cost_clean_min_attempt = 0.099
        print("Starting round attempt: {0:2.2f}".format((i+1)+(k/100)))
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            train_costs = []
            dev_costs = []
            train_clean_costs = []
            dev_clean_costs = []
            cost_regs = []    
            epochs = []
            cost_reg = 0.99
            rmsle_train_cost = 0.99
            best_epoch = 1

            for epoch in range(training_epochs):
                session.run(optimizer, feed_dict={X: X_scaled_training, Y: Y_scaled_training})

                train_cost =  session.run(cost, feed_dict={X: X_scaled_training, Y: Y_scaled_training})
                dev_cost =  session.run(cost, feed_dict={X: X_scaled_dev, Y: Y_scaled_dev})
                cost_reg =  session.run(cost_regu, feed_dict={X: X_scaled_dev, Y: Y_scaled_dev})
                
                train_costs.append(train_cost)
                dev_costs.append(dev_cost)
                train_clean_costs.append(train_cost-cost_reg)
                dev_clean_costs.append(dev_cost-cost_reg)        
                cost_regs.append(cost_reg)
                epochs.append(epoch)

                if ((dev_cost - cost_reg < dev_cost_clean_min_attempt)&((train_cost-cost_reg)*1000<=max_train_clean_cost_in_best_epoch)):
                    Y_predicted = inv_boxcox(session.run(prediction, feed_dict={X: X_scaled_test}), lambdaY)
                    dev_cost_min = dev_cost
                    dev_cost_clean_min_attempt = dev_cost - cost_reg
                    dev_cost_clean_min_round=min([dev_cost_clean_min_round, dev_cost_clean_min_attempt])
                    best_epoch = epoch
                    Y_predicted_train = session.run(prediction, feed_dict={X: X_scaled_training})
                    if (dev_cost_clean_min_round == dev_cost_clean_min_attempt):
                        Y_predicted_train_list["capseta_{0:1d}".format(i+1)] = Y_predicted
                if ((epoch % display_step == 0) & (epoch > 0)):
                    print("[{0:2.2f}][{1:5d}]".format((i+1)+(k/100),epoch), \
                            " Train: {0:8.3f}".format(1000*train_cost), \
                            " Train Clean: {0:8.3f}".format(1000*(train_cost-cost_reg)), \
                            " Dev: {0:8.3f}".format(1000*dev_cost), \
                            " Dev Clean: {0:8.3f}".format(1000*(dev_cost-cost_reg)), \
                            " Reg: {0:8.3f}".format(1000*cost_reg), \
                            " Dev Min: {0:8.3f}".format(1000*dev_cost_min), \
                            " Dev Clean Min Attempt: {0:8.3f}".format(1000*(dev_cost_clean_min_attempt)), \
                            " Dev Clean Min Round: {0:8.3f}".format(1000*(dev_cost_clean_min_round)), \
                            " [{}]".format(datetime.datetime.now().strftime('%H:%M:%S')))

                if (epoch - best_epoch >= max_delay_between_best_epochs):
                    print("No progress made in {} epochs. Convergence reached. Performing early stopping".format(epoch - best_epoch))
                    break
            attempt_cost =  dev_cost_clean_min_attempt
            print("End attempt {0:2.2f} with cost: {1:4.3f}".format((i+1)+(k/100), attempt_cost*1000))
    if (k==max_round_attempts):
        print("Max number of round attempts reached, exiting round")
    round_dev_min_clean_costs.append(dev_cost_clean_min_round)
#    Y_predicted_sum=Y_predicted_sum.add(Y_predicted)
for i in range(X_scaled_test.shape[0]):
    print("Calculating median for test example i=", i)
    z1=[]
    for j in range(num_rounds):
        z1.append(int(Y_predicted_train_list["capseta_{0:1d}".format(j+1)][i]))
    # Y_predicted_train_list["SalePrice"][i]=st.median(z1)
    Y_predicted_train_list["SalePrice"][i]=st.mean(z1)
    Y_predicted_train_list["Max"][i]=np.max(z1)
    Y_predicted_train_list["Min"][i]=np.min(z1)

Y_predicted_total=Y_predicted_sum.divide(num_rounds)

print("costs: ", round_dev_min_clean_costs)
total_cost_avg = reduce(lambda x, y: x + y, round_dev_min_clean_costs) / len(round_dev_min_clean_costs)
total_cost_min = reduce(lambda x, y: min(x,y), round_dev_min_clean_costs)
total_cost_max = reduce(lambda x, y: max(x,y), round_dev_min_clean_costs)
print("cost avg: ", 1000*total_cost_avg, " cost min: ", 1000*total_cost_min, " cost max: ", 1000*total_cost_max)
plt.plot(epochs,1000*np.squeeze(train_clean_costs),epochs,1000*np.squeeze(dev_clean_costs),epochs,1000*np.squeeze(cost_regs))
plt.ylabel('train clean, dev clean, reg costs (x1000)')
plt.xlabel('iterations (per tens)')
plt.xlim(2*epoch/10,epoch)
plt.ylim(0,np.amin(dev_clean_costs)*1000*2)
plt.title("Learning rate =" + str(learning_rate)) 
plt.legend(['Train Clean Cost', 'Dev Clean Cost', 'Reg Costs'])
plt.show()
# XGBoost

model = XGBClassifier()
#X_train = pd.concat(Y_scaled_training,X_scaled_dev)
# Y_train = Y_scaled_test
# X_test = X_scaled_test
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# accuracy = accuracy_score(X_train, Y_predicted_train)
# print("XGBoost Accuracy: ", accuracy)
# print("RMSLE: ", mean_squared_log_error(y_true, y_pred))
if (1000*(total_cost_avg)<=9999):
#    predicted_output = pd.concat([test["Id"],pd.DataFrame(Y_predicted_total, columns=["SalePrice"])], axis=1)
    predicted_output = pd.concat([test["Id"],pd.DataFrame(Y_predicted_train_list, columns=["SalePrice"])], axis=1)
    predicted_output.to_csv("./Prediction_" + datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S') + ".csv", index=False)

    print("Prediction output to file completed")
np.set_printoptions(precision=10)
predicted_output = pd.concat([test["Id"],pd.DataFrame(Y_predicted_total, columns=["SalePrice"])], axis=1)
#print(predicted_output.to_string())
test_output = pd.concat([test["Id"],pd.DataFrame(Y_predicted_train_list, columns=["SalePrice", "Max", "Min"])], axis=1)
test_output['Desviacio'] = pd.Series()
for i in range(test_output.shape[0]):
    desviacio = 0.5*(test_output["Max"][i]- test_output["Min"][i])/test_output["SalePrice"][i]*100
    test_output["Desviacio"][i] = desviacio
desviacio_mitjana = pd.DataFrame.mean(test_output["Desviacio"])
desviacio_maxima= pd.DataFrame.max(test_output["Desviacio"])
desviacio_minima= pd.DataFrame.min(test_output["Desviacio"])
desviacio_mediana= pd.DataFrame.median(test_output["Desviacio"])
print("Desviació mitjana: +/-{0:1.2f}%".format(desviacio_mitjana))
print("Desviació mediana: +/-{0:1.2f}%".format(desviacio_mediana))
print("Desviació max: +/-{0:1.2f}%".format(desviacio_maxima))
print("Desviació min: +/-{0:1.2f}%".format(desviacio_minima))
print(test_output[test_output.Desviacio >=10].sort_values(by=['Desviacio'],ascending=False))