TARGET = 'SalePrice'

featTARGET = [TARGET]



possiblefeature001 = 'LotArea'

possiblefeature002 = 'YearBuilt'

possiblefeature003 = '1stFlrSF'

possiblefeature004 = '2ndFlrSF'

possiblefeature005 = 'FullBath'

possiblefeature006 = 'BedroomAbvGr'

possiblefeature007 = 'TotRmsAbvGrd'



features = [possiblefeature001, possiblefeature002, possiblefeature003, 

            possiblefeature004, possiblefeature005, possiblefeature006,

            possiblefeature007]



featuresAndTarget = features + featTARGET



submissionID = 'Id'

featsubmissionID = [submissionID]

featuresAndSubmissionID = featsubmissionID + features



featuresAndTargetAndSubmissionID = featuresAndSubmissionID + featTARGET

    
def reset_features():

    possiblefeature001 = 'LotArea'

    possiblefeature002 = 'YearBuilt'

    possiblefeature003 = '1stFlrSF'

    possiblefeature004 = '2ndFlrSF'

    possiblefeature005 = 'FullBath'

    possiblefeature006 = 'BedroomAbvGr'

    possiblefeature007 = 'TotRmsAbvGrd'

        

    features = [possiblefeature001, possiblefeature002, possiblefeature003, 

            possiblefeature004, possiblefeature005, possiblefeature006,

            possiblefeature007]

    

    TARGET = 'SalePrice'

    featTARGET = [TARGET]

    

    featuresAndTarget = features + featTARGET

    

    submissionID = 'Id'

    featsubmissionID = [submissionID]

    featuresAndSubmissionID = features + featsubmissionID

    

    featuresAndTargetAndSubmissionID = featuresAndSubmissionID + featTARGET

    

    return features, featuresAndTarget, featuresAndSubmissionID, featuresAndTargetAndSubmissionID
def read_train_test_data(num_rows = None, nan_as_category = False):

    # Read data and merge

    df = pd.read_csv('../input/train.csv', nrows= num_rows)

    

    test_df = pd.read_csv('../input/test.csv', nrows= num_rows)

    print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))

    df = df.append(test_df).reset_index()

        

    reset_features()  #call features

    XYtrainTest = df[featuresAndTargetAndSubmissionID]

    

    del test_df

    del df

    gc.collect()

    return XYtrainTest

# read_train_test_data function END
def divide_train_test(df):

    # Divide in training/validation and test data

    XYtrain= df[df[TARGET].notnull()]

    # Create target object and call it y

    y = XYtrain[featTARGET] #home_data.SalePrice

    # Create X

    X = XYtrain[features] #home_data[features]

    

    Xtest  = df[df[TARGET].isnull()] #test data for later predictions

    

    return y,X,Xtest
#Take N rows to debug first

def main(debug = False):

    num_rows = 1000 if debug else None

    df = read_train_test_data(num_rows)   #call read_train_test_data

    features, featuresAndTarget, featuresAndSubmissionID, featuresAndTargetAndSubmissionID = reset_features()

    y,X,Xtest = divide_train_test(df)

    return y,X,Xtest,df

# main function END
# Code enhanced

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from learntools.core import *

import gc

pd.set_option('display.max_rows', 8)

pd.set_option('display.max_columns', 15)



import matplotlib.pyplot as plt

%matplotlib inline



import warnings

warnings.filterwarnings("ignore")



#***********************************

y,X,Xtest,XYtrainTest = main(debug = False)   #call main to set debug rows - True, False for full run

#***********************************



# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)



# Specify Model

model = DecisionTreeRegressor(random_state=1)

# Fit Model

model.fit(train_X, train_y)



# Make validation predictions and calculate mean absolute error

val_predictions = model.predict(val_X)

val_mae = mean_absolute_error(val_predictions, val_y)

print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))



# Using best value for max_leaf_nodes

model2 = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)

model2.fit(train_X, train_y)

val_predictions2 = model2.predict(val_X)

val_mae2 = mean_absolute_error(val_predictions2, val_y)

print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae2))



# Define the model. Set random_state to 1

rf_model = RandomForestRegressor(random_state=1)

rf_model.fit(train_X, train_y)

rf_val_predictions = rf_model.predict(val_X)

rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)



print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))

# To improve accuracy, create a new Random Forest model which you will train on all training data

rf_model_on_full_data = RandomForestRegressor(random_state=1)



# fit rf_model_on_full_data on all data from the training data

rf_model_on_full_data.fit(X, y)

rf_val_predictions = rf_model_on_full_data.predict(val_X)

rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)



print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))

XYtrainTest
XYtrainTest.describe()
# path to file you will use for predictions

#test_data_path = '../input/test.csv'

# read test data file using pandas

#test_data = pd.read_csv(test_data_path)

# create test_X which comes from test_data but includes only the columns you used for prediction.

# The list of columns is stored in a variable called features

test_X = Xtest[features]  #using already saved dataset



# make predictions which we will submit. 

test_preds = rf_model_on_full_data.predict(test_X)



# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



output = pd.DataFrame({'Id': Xtest.Id,

                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)
X
import seaborn as sns

plt.figure(figsize=(10,4))

plt.xlim(y.SalePrice.min(), y.SalePrice.max()*1.1)

_= sns.boxplot(x=y.SalePrice)
def Store_and_Refresh(df,filename):

    df.to_pickle(filename)   #Store the dataset

    print('#' * 88)

    print('%s file is stored for backup and read back by pickle...' %(filename))   

    data = pd.read_pickle(filename) #retrieve from Pickle

    print('#' * 88) 

    del df

    gc.collect()

    return data
filename = 'XY_Train_Test_Clean_DataSet.pkl'

XYtrainTestFromPickle = Store_and_Refresh(XYtrainTest,filename)
import numpy as np

y['outlier'] = np.where((y[TARGET] <= 50000) | (y[TARGET] > 250000),0,1)
from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler

import statsmodels.api as sm
featureToCheck02 = 'LotArea'

Val = X[featureToCheck02]



feature_minmax = MinMaxScaler().fit_transform(Val.values.reshape(-1, 1).astype(np.float64)).flatten()

feature_std = StandardScaler().fit_transform(Val.values.reshape(-1, 1).astype(np.float64)).flatten()

feature_log = np.log(Val)



_= plt.hist(Val,  bins=30)
_= plt.hist(feature_std, bins=30)
_= plt.hist(feature_log, bins=30)
sm.qqplot(Val, loc=Val.mean(), scale=Val.std())
sm.qqplot(feature_log, loc=feature_log.mean(), scale=feature_log.std())
X['LotArea_normed']= feature_log  #Include the normalized feature to full set to analyze further
# Multivariate plots helps us to better understand the relationships between attributes



_ = pd.tools.plotting.scatter_matrix(X,alpha=0.2, figsize=(20, 25)) #bind a variable to the plot so IPython only shows plots and _ is bound to the unwanted output
def FeatureTransform(df,feat):

    df[feat]= np.log(df[feat]) #FeatureTransform function to update new value across train & test set

    return df
# Hyper Parameters

N_FOLDS = 5

MAX_BOOST_ROUNDS = 700

LEARNING_RATE = .0022
params = {}   #XGBRegressor used in function model_lgb_cv

params['max_bin'] = 10

params['learning_rate'] = LEARNING_RATE # shrinkage_rate

params['boosting_type'] = 'gbdt'

params['objective'] = 'regression'

params['metric'] = 'l1'          # or 'mae'

params['sub_feature'] = 0.50      # feature_fraction 

params['bagging_fraction'] = 0.85 # sub_row

params['bagging_freq'] = 40

params['num_leaves'] = 512        # num_leaf

params['min_data'] = 500         # min_data_in_leaf

params['min_hessian'] = 0.05     # min_sum_hessian_in_leaf

params['verbose'] = 0

params['stratified'] = True

params['show_stdv'] = True

params['nfold'] = 5

params['multiclass'] = 'multiclass'

def model_lgb_cv(X,y,Xtest):

    train_x = X.values.astype(np.float32, copy=False) #new

    train_d = lgb.Dataset(X, label=y)   #new

    print(X.shape, y.shape)



    # Cross-validate

    clf = lgb.cv(params, train_d, num_boost_round=MAX_BOOST_ROUNDS, early_stopping_rounds=None)

    

    print('AFTER lgb.cv fit: Current parameters:\n', params)

    best_boost = len(clf['l1-mean'])

    print('\nBest num_boost_round2:', len(clf['l1-mean']))

    print('Best CV2 score:', clf['l1-mean'][-1])

    

    # train model

    model_lgb = lgb.train(params, train_d, num_boost_round = best_boost)

    

    y_pred = model_lgb.predict(Xtest)

    

    return model_lgb
import lightgbm as lgb

print('before calling model_lgb_cv')

#check and remove any engineered features like outliers or normed included by mistake

bestmodel = model_lgb_cv(X, y, Xtest)
#prepare diabetes from sklearn datasets to practice DataFrame

import pandas as pd

from sklearn import datasets, linear_model

from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt



# Load the Diabetes Housing dataset

columns = 'age sex bmi map tc ldl hdl tch ltg glu'.split() # Declare the columns names

diabetes = datasets.load_diabetes() # Call the diabetes dataset from sklearn

df = pd.DataFrame(diabetes.data, columns=columns) # load the dataset as a pandas data frame
#Quick look at KFold cross validation logic with example data

from sklearn.model_selection import KFold # import KFold

Xx = np.array([[1, 2], [3, 4], [1, 2], [3, 4]]) # create an array

yy = np.array([1, 2, 3, 4]) # Create another array

kf = KFold(n_splits=2) # Define the split - into 2 folds 

kf.get_n_splits(Xx) # returns the number of splitting iterations in the cross-validator

print(kf) 

KFold(n_splits=2, random_state=None, shuffle=False)



for train_index, test_index in kf.split(Xx):

    print('TRAIN:', train_index, 'TEST:', test_index)

    Xx_train, Xx_test = Xx[train_index], Xx[test_index]

    yy_train, yy_test = yy[train_index], yy[test_index]