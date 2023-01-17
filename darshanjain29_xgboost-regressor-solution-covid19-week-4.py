# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns

# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")
test_df = pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv")

#Data Study
from pandas_profiling import ProfileReport
train_profile = ProfileReport(train_df, title='Pandas Profiling Report', html={'style':{'full_width':True}})
train_profile

train_df.info()
# Replacing all the Province_State that are null by the Country_Region values
train_df.Province_State.fillna(train_df.Country_Region, inplace=True)
test_df.Province_State.fillna(test_df.Country_Region, inplace=True)

# Handling the Date column
# 1. Converting the object type column into datetime type
train_df.Date = train_df.Date.apply(pd.to_datetime)
test_df.Date = test_df.Date.apply(pd.to_datetime)

#Extracting Date and Month from the datetime and converting the feature as int
train_df.Date = train_df.Date.dt.strftime("%m%d").astype(int)
test_df.Date = test_df.Date.dt.strftime("%m%d").astype(int)


#Label Encoding for object to numeric conversion - Option 1
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

train_df.Country_Region = le.fit_transform(train_df.Country_Region)
train_df['Province_State'] = le.fit_transform(train_df['Province_State'])

test_df.Country_Region = le.fit_transform(test_df.Country_Region)
test_df['Province_State'] = le.fit_transform(test_df['Province_State'])

'''
#One Hot Encoding for object to numeric conversion - Option 2

def one_hot(df, cols):
    """
    @param df pandas DataFrame
    @param cols a list of columns to encode 
    @return a DataFrame with one-hot encoding
    """
    i = 0
    for each in cols:
        #print (each)
        dummies = pd.get_dummies(df[each], prefix=each, drop_first= True)
        if i == 0: 
            print (dummies)
            i = i + 1
        df = pd.concat([df, dummies], axis=1)
    return df
    '''
'''
#Handling categorical data

objList = train_df.select_dtypes(include = "object").columns
train_df = one_hot(train_df, objList) 
test_df = one_hot(test_df, objList) 

print (train_df.shape)
'''
'''
# Removing duplicate entries
train_df = train_df.loc[:,~train_df.columns.duplicated()]
test_df = test_df.loc[:,~test_df.columns.duplicated()]
print (test_df.shape)
'''
'''
# Dropping the object type columns
train_df.drop(objList, axis=1, inplace=True)
test_df.drop(objList, axis=1, inplace=True)
print (train_df.shape)
'''
'''
#Final check on the data
train_df.select_dtypes(include = "object").columns
#So, no columns with object data is there. Our data is ready for training
'''
X_train = train_df.drop(["Id", "ConfirmedCases", "Fatalities"], axis = 1) 
 
Y_train_CC = train_df["ConfirmedCases"] 
Y_train_Fat = train_df["Fatalities"] 

X_test = test_df.drop(["ForecastId"], axis = 1) 
from sklearn.model_selection import ShuffleSplit, cross_val_score
skfold = ShuffleSplit(random_state=7)

#2. XGBRegressor
import xgboost as xgb

clf_xgb_CC = xgb.XGBRegressor(n_estimators = 40000)
clf_xgb_Fat = xgb.XGBRegressor(n_estimators = 20000)

xgb_acc = cross_val_score(clf_xgb_CC, X_train, Y_train_CC, cv = skfold)
xgb_acc_fat = cross_val_score(clf_xgb_Fat, X_train, Y_train_Fat, cv = skfold)

print (xgb_acc.mean(), xgb_acc_fat.mean())
#0.9855346531774586 0.9742142138236705 with LE

clf_xgb_CC.fit(X_train, Y_train_CC)
Y_pred_CC = clf_xgb_CC.predict(X_test) 

clf_xgb_Fat.fit(X_train, Y_train_Fat)
Y_pred_Fat = clf_xgb_Fat.predict(X_test) 
print (Y_pred_Fat)
df_out = pd.DataFrame({'ForecastId': [], 'ConfirmedCases': [], 'Fatalities': []})
soln = pd.DataFrame({'ForecastId': test_df.ForecastId, 'ConfirmedCases': Y_pred_CC, 'Fatalities': Y_pred_Fat})
df_out = pd.concat([df_out, soln], axis=0)
df_out.ForecastId = df_out.ForecastId.astype('int')
df_out.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")