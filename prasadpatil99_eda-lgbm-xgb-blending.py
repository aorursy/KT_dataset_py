import numpy as np

import pandas as pd

import seaborn as sns

sns.set(style = 'whitegrid')



import matplotlib.pyplot as plt 

%matplotlib inline



import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings("ignore", category=FutureWarning)
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
training_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

testing_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
training_data.head()
testing_data.describe()
null_col = (training_data.isnull().sum()/len(training_data)) * 100

null_col = null_col.sort_values(ascending=False)

null_col
corr = training_data.corr()

corr.sort_values(["SalePrice"], ascending = False, inplace = True)

corr.SalePrice
train_data = training_data.drop(['BsmtFinSF2','BsmtHalfBath','MiscVal','Id','LowQualFinSF',

                                 'YrSold','OverallCond','MSSubClass','EnclosedPorch','GarageType',

                                 'KitchenAbvGr','PoolQC','MiscFeature','Alley','Fence','FireplaceQu','SalePrice'], axis = 1)



test_data = testing_data.drop(['BsmtFinSF2','BsmtHalfBath','MiscVal','Id','LowQualFinSF',

                                 'YrSold','OverallCond','MSSubClass','EnclosedPorch','GarageType',

                                 'KitchenAbvGr','PoolQC','MiscFeature','Alley','Fence','FireplaceQu'], axis = 1)
categorical_features = train_data.select_dtypes(include = ["object"]).columns

categorical_features
numerical_features = train_data.select_dtypes(exclude = ["object"]).columns

numerical_features
sns.distplot(training_data['SalePrice'], color="r", kde=False)

plt.title("Distribution of Sale Price")

plt.ylabel("Number of Occurences")

plt.xlabel("Sale Price");
sns.barplot(x='SaleCondition', y='SalePrice', data=training_data)
plt.scatter(x ='TotalBsmtSF', y = 'SalePrice', data = training_data)

plt.xlabel('Total Basement in Square Feet')
sns.catplot(x ='Street', y = 'SalePrice', data = training_data)
def mapping(data):

    GarageCondM = {'TA':0, 'Fa':1, 'Gd':2, 'Ex':3}

    data['GarageCond'] = data['GarageCond'].map(GarageCondM)



    GarageQualM = {'TA':0, 'Fa':1, 'Gd':2, 'Ex':3}

    data['GarageQual'] = data['GarageQual'].map(GarageQualM)



    GarageFinishM = {'RFn':0, 'Unf':1}

    data['GarageFinish'] = data['GarageFinish'].map(GarageFinishM) 



    BsmtFinType2M = {'Unf':0, 'BLQ':1, 'ALQ':2, 'Rec':3, 'LwQ':4, 'GLQ':5}

    data['BsmtFinType2'] = data['BsmtFinType2'].map(BsmtFinType2M) 



    BsmtExposureM = {'No':0, 'Gd':1, 'Mn':2, 'Av':3}

    data['BsmtExposure'] = data['BsmtExposure'].map(BsmtExposureM) 



    BsmtCondM = {'TA':0, 'Fa':1, 'Gd':2}

    data['BsmtCond'] = data['BsmtCond'].map(BsmtCondM)



    BsmtQualM = {'TA':0, 'Fa':1, 'Gd':2, 'Ex':3}

    data['BsmtQual'] = data['BsmtQual'].map(BsmtQualM)



    BsmtFinType1M = {'GLQ':0, 'ALQ':1, 'Unf':2, 'Rec':3, 'BLQ':4, 'LwQ':5}

    data['BsmtFinType1'] = data['BsmtFinType1'].map(BsmtFinType1M)



    MasVnrTypeM = {'BrkFace':0, 'None':1, 'Stone':2, 'BrkCmn':3, 'BLQ':4, 'LwQ':5}

    data['MasVnrType'] = data['MasVnrType'].map(MasVnrTypeM)



    ElectricalM = {'SBrkr':0, 'FuseA':1, 'FuseF':2, 'BrkCmn':3, 'BLQ':4, 'LwQ':5}

    data['Electrical'] = data['Electrical'].map(ElectricalM)

    

    return data
train_data = mapping(train_data)

test_data = mapping(test_data)
train_data = train_data.fillna(train_data.mean())

train_data = pd.DataFrame(train_data)



test_data = test_data.fillna(test_data.mean())

test_data = pd.DataFrame(test_data)

train_data.head()
test_data.head()
train_data['train'] = 1

test_data['test'] = 0

combined = pd.concat([train_data, test_data])

combined = pd.get_dummies(combined, prefix_sep='_', columns = list(categorical_features))

combined.head()
train_data = combined[combined["train"] == 1]

test_data = combined[combined["test"] == 0]

train_data.drop(["test","train"], axis = 1, inplace = True)

test_data.drop(["train","test"], axis = 1, inplace = True)
train_data.head()
test_data.head()
categorical_features = train_data.select_dtypes(include = ["object"]).columns

categorical_features
categorical_features = test_data.select_dtypes(include = ["object"]).columns

categorical_features
def null(data):

    null_col = (data.isnull().sum()/len(data)) * 100

    null_col = null_col.sort_values(ascending=False)

    return null_col
null(test_data)
null(train_data)
X_train = train_data.copy()

Y_train = training_data["SalePrice"].values

X_test = test_data.copy()

X_train.shape, Y_train.shape, X_test.shape
from sklearn.model_selection import KFold

from sklearn.preprocessing import RobustScaler

from sklearn.pipeline import make_pipeline
import xgboost as xgb

gbm = xgb.XGBRegressor(

                 colsample_bytree=0.1,   #ratio_of_constructing_each_tree

                 gamma=0.0,              #loss_reduction_param

                 learning_rate=0.01,     #for_updating_param

                 max_depth=3,            #maximum_depth_of_tree

                 min_child_weight=0,     #minimum_sum_of_child_weight

                 n_estimators=10000,     #total_no_of_iterations                                                                   

                 reg_alpha=0.0006,       #updating_coefficient_L1_regularization_term

                 reg_lambda=0.6,         #updating_coefficient_L2_regularization_term

                 subsample=0.7,          #sampling_training_data_randomly

                 seed=30,                #random_number_seed

                 silent=1)               #occurence_of_message

gbm_fit = gbm.fit(X_train, Y_train)

gbm_predictions = gbm.predict(X_test)
import lightgbm as lgb

lgb = lgb.LGBMRegressor(objective='regression',     

                        num_leaves=5,                  #num_of_leaves in a tree

                        learning_rate=0.05,            #updating_weights_to_mimize_loss

                        n_estimators=5000,             #num_of_iterations

                        max_bin = 55,                  #num_of_bins_binning_refers_to_continous_unique_value

                        bagging_fraction = 0.8,        #select_random_data_samples

                        bagging_freq = 5,              #perform_bagging_at_k_iteration

                        feature_fraction = 0.2319,     #randomly_select_feature_here_23%

                        feature_fraction_seed=9,       #random_seed_for_feature_fraction

                        bagging_seed=9,                #random_seed_for_bagging

                        min_data_in_leaf =6,           #minimum_no_leaf

                        min_sum_hessian_in_leaf = 11)

lgb = lgb.fit(X_train, Y_train)

lgb_predictions = lgb.predict(X_test)
blend = lgb_predictions*0.5+gbm_predictions*0.5
submission = pd.DataFrame({

        "Id": testing_data["Id"],

        "SalePrice": blend

    })

submission.to_csv('final_submissison.csv', index=False)
submission.head()