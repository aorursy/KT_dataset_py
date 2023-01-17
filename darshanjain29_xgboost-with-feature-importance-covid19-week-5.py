# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
subm = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-5/submission.csv")

print (subm.head(20))
train_df = pd.read_csv("../input/covid19-global-forecasting-week-5/train.csv")

test_df = pd.read_csv("../input/covid19-global-forecasting-week-5/test.csv")
print (train_df.head(3))

print ("*"*80)

print (test_df.head(3))

print ("*"*80)

print (subm.head(5))

print ("*"*80)
test_df.head(5)
# Replacing all the County, Province_State that are null by the Country_Region values

'''

train_df.Province_State.fillna(train_df.Country_Region, inplace=True)

test_df.Province_State.fillna(test_df.Country_Region, inplace=True)

'''

train_df.County.fillna(train_df.Country_Region, inplace=True)

test_df.County.fillna(test_df.Country_Region, inplace=True)



train_df.drop(["Province_State", "Country_Region"], inplace=True, axis = 1)

test_df.drop(["Province_State", "Country_Region"], inplace=True, axis = 1)

train_df.Date = pd.to_datetime(train_df.Date)

test_df.Date = pd.to_datetime(test_df.Date)



train_df.Date = train_df.Date.dt.strftime("%Y%m%d").astype(int)

test_df.Date = test_df.Date.dt.strftime("%Y%m%d").astype(int)
from sklearn import preprocessing

le = preprocessing.LabelEncoder()





train_df.County = le.fit_transform(train_df.County)

test_df.County = le.fit_transform(test_df.County)

'''

train_df.Province_State = le.fit_transform(train_df.Province_State)

test_df.Province_State = le.fit_transform(test_df.Province_State)



train_df.Country_Region = le.fit_transform(train_df.Country_Region)

test_df.Country_Region =le.fit_transform(test_df.Country_Region)

'''

test_df.Target = le.fit_transform(test_df.Target)

train_df.Target = le.fit_transform(train_df.Target)
print (train_df.info())

print ('*'*80)

print (test_df.info())
X_train = train_df.drop(["Id", "TargetValue"], axis = 1)

Y_train = train_df.TargetValue

X_test = test_df.drop(["ForecastId"], axis = 1)
from sklearn.model_selection import ShuffleSplit, cross_val_score

skfold = ShuffleSplit(random_state=7)
import xgboost as xgb



reg = xgb.XGBRegressor()

xgb_score = cross_val_score(reg, X_train, Y_train, cv = skfold)
print (xgb_score.mean())

#0.9161024722436133 0.9161024722436133 - with all features
reg.fit(X_train,Y_train)



Y_pred = reg.predict(X_test)

print (Y_pred)
#Calculating feature importance

from matplotlib import pyplot



importance = reg.feature_importances_

print (importance)



for i,v in enumerate(importance):

    print ('Feature: %d, Score: %.5f ' % (i+1,v))



pyplot.bar([x for x in range(len(importance))], importance)

pyplot.show()



'''

 1   County          0.31338 non-null  int64  

 2   Province_State  0.10511 non-null  int64  

 3   Country_Region  0.09913 non-null  int64  

 4   Population      0.06679 non-null  int64  

 5   Weight          0.22049 non-null  float64

 6   Date            0.19510 non-null  int64  

 7   Target          0.00000 non-null  int64  

 '''
output = pd.DataFrame({'Id': test_df.ForecastId  , 'TargetValue': Y_pred})



a=output.groupby(['Id'])['TargetValue'].quantile(q=0.05).reset_index()

b=output.groupby(['Id'])['TargetValue'].quantile(q=0.5).reset_index()

c=output.groupby(['Id'])['TargetValue'].quantile(q=0.95).reset_index()





a.columns=['Id','q0.05']

b.columns=['Id','q0.5']

c.columns=['Id','q0.95']

a=pd.concat([a,b['q0.5'],c['q0.95']],1)

a['q0.05']=a['q0.05']

a['q0.5']=a['q0.5']

a['q0.95']=a['q0.95']



sub=pd.melt(a, id_vars=['Id'], value_vars=['q0.05','q0.5','q0.95'])

sub['variable']=sub['variable'].str.replace("q","", regex=False)

sub['ForecastId_Quantile']=sub['Id'].astype(str)+'_'+sub['variable']

sub['TargetValue']=sub['value']

sub=sub[['ForecastId_Quantile','TargetValue']]

sub.reset_index(drop=True,inplace=True)

sub.to_csv("submission.csv",index=False)

sub.info()

print ("Submission file generated successfully.")