# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# Preprocessing
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.cross_validation import train_test_split

def extract_features(data):
    data = data.fillna(0)
    train_data, validate_data = train_test_split(data, test_size=0.2, random_state=42)

    # Dates
    data['Hour'] = data.Dates.dt.hour
    data['Day'] = data.Dates.dt.day
    data['DayOfWeekNum'] = pd.Categorical.from_array(data.DayOfWeek).codes
    data['DayOfMonth'] = data.Dates.dt.day
    data['DayOfYear'] = data.Dates.dt.dayofyear
    data['WeekOfYear'] = data.Dates.dt.weekofyear
    data['Month'] = data.Dates.dt.month
    data['Year'] = data.Dates.dt.year
    data["Fri"] = np.where(data.DayOfWeek == "Friday",1,0)
    data["Sat"] = np.where(data.DayOfWeek == "Saturday",1,0)
    data["Weekend"] = data["Fri"] + data["Sat"]

    # PdDisrict
    data['PdDistrictCat'] = pd.Categorical.from_array(data.PdDistrict).codes
  
    
    # Lat/Long
    
    data = data[data.X <-121]
    data = data[data.Y<40]
    
    data["X_reduced"] = data.X.apply(lambda x: "{0:.2f}".format(x)).astype(float)
    data["Y_reduced"] = data.Y.apply(lambda x: "{0:.2f}".format(x)).astype(float)
    data["X_reduced_cat"] = pd.Categorical.from_array(data.X_reduced).codes
    data["Y_reduced_cat"] = pd.Categorical.from_array(data.Y_reduced).codes
    
    data["rot_45_X"] = .707*data["Y"] + .707*data["X"]
    data["rot_45_Y"] = .707* data["Y"] - .707* data["X"]

    data["rot_30_X"] = (1.732/2)*data["X"] + (1./2)*data["Y"]
    data["rot_30_Y"] = (1.732/2)* data["Y"] - (1./2)* data["X"]

    data["rot_60_X"] = (1./2)*data["X"] + (1.732/2)*data["Y"]
    data["rot_60_Y"] = (1./2)* data["Y"] - (1.732/2)* data["X"]

    data["radial_r"] = np.sqrt( np.power(data["Y"],2) + np.power(data["X"],2) )

    # Output feature - crime category
    data['CategoryNum'] = pd.Categorical.from_array(data.Category).codes
    
    classes = pd.Categorical.from_array(data.Category).categories
    
    return pd.concat([data.Hour,
                      data.Day,
                      data.DayOfWeekNum,
                      data.DayOfMonth,
                      pd.get_dummies(data.Month),
                      pd.get_dummies(data.Year),
                      data.PdDistrictCat,
                      data.rot_45_X,
                      data.rot_45_Y,
                      data.CategoryNum
                     ], axis=1), classes
# Set parameters for XGBoost
def set_param():
    
    # setup parameters for xgboost
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.4
    param['silent'] = 0
    param['nthread'] = 4
    param['num_class'] = num_class
    param['eval_metric'] = 'mlogloss'

    # Model complexity
    param['max_depth'] = 8 #set to 8
    param['min_child_weight'] = 1
    param['gamma'] = 0 
    param['reg_alfa'] = 0.05

    param['subsample'] = 0.8
    param['colsample_bytree'] = 0.8 #set to 1

    # Imbalanced data
    param['max_delta_step'] = 1
    
    return param
# Load data and extract features
data = pd.read_csv('../input/train.csv', parse_dates=['Dates'])
data, classes = extract_features(data)
# Split into train/validate test
train_data, validate_data = train_test_split(data, test_size=0.2, random_state=42)

train_X = train_data.drop('CategoryNum', 1)
train_Y = train_data.CategoryNum
validate_X = validate_data.drop('CategoryNum', 1)
validate_Y = validate_data.CategoryNum

dtrain = xgb.DMatrix(train_X, label=train_Y)
dtest = xgb.DMatrix(validate_X, label=validate_Y)

num_class = len(data.CategoryNum.unique())

param = set_param()
watchlist = [ (dtrain,'train'), (dtest, 'eval') ]
num_round = 10

# Train XGBoost    
bst = xgb.train(param, dtrain, num_round, watchlist);
yprob = bst.predict(dtest).reshape( validate_Y.shape[0], num_class)
ylabel = np.argmax(yprob, axis=1)