import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
train = pd.DataFrame(
{"pet": ['dog','cat','dog','cat'],
 "age": [7,6,12,2]
},
    index = [1,2,3,4]
)
train
# add a missing value 
train.loc[1,'pet'] = np.NaN
train
train.isnull().sum()
# 1. Instantiate Imputer and One Hot Encoder
my_OH = OneHotEncoder(handle_unknown = 'ignore', sparse = False)
my_imputer = SimpleImputer(strategy = 'most_frequent')
# 2. Impute missing values in Train Set
train_fix = pd.DataFrame(my_imputer.fit_transform(train))
train_fix.columns = train.columns
train_fix
# 2. One Hot Encoding
OH_cat = pd.DataFrame(my_OH.fit_transform(train_fix[['pet']]))
OH_cat

# 3. Concat 
train_fix = pd.concat([OH_cat,train_fix[['age']]], axis = 1)
train_fix
# Test set
test = pd.DataFrame(
{"pet": ['dog',np.NaN,'fish'],
 "age": [2,3,5]
},
    index = [1,2,3]
)
test
# 1. Impute NaN
test_fix = pd.DataFrame(my_imputer.transform(test))
test_fix.columns = test.columns
test_fix
# 2. OH
test_OH_cat = pd.DataFrame(my_OH.fit_transform(test_fix[['pet']]))
test_OH_cat

# 3. Concat OH categorical variables and numerical
test_fix = pd.concat([test_OH_cat,test_fix[['age']]], axis = 1)
test_fix
