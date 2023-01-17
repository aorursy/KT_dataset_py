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



# Any results you write to the current directory are saved as output.
# importing data 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))



dc  = pd.read_csv("../input/dc-residential-properties/DC_Properties.csv")



dc.head()
'''

Split the data into 3 parts – training_data, test_data and unknown_data. 

The unknown_data contains all the records where the value of price is unknown/missing 

in the given data set. The split of training and testing data should be in the ratio of 80:20.

'''
unknown_data = dc[pd.isnull(dc['PRICE'])]['PRICE']

print(unknown_data)
# subsetting data with Price is not null values

subset = dc[dc.PRICE.notnull()]

subset.head()
# defining y as target varible and x is subset of all features 

y = subset.PRICE

x = subset.drop('PRICE',axis = 1)



# treating null values

x = x.drop(['X','Y','CMPLX_NUM','FULLADDRESS','LONGITUDE','CITY','STATE','NATIONALGRID','CENSUS_BLOCK','SALEDATE','QUADRANT'],axis=1)



x.GBA = x.GBA.fillna(x.GBA.mean())



x.AYB = x.AYB.fillna(x.AYB.median())



x.STORIES = x.STORIES.fillna(x.STORIES.median())



x.KITCHENS = x.KITCHENS.fillna(x.KITCHENS.median())



x.NUM_UNITS = x.NUM_UNITS.fillna(x.NUM_UNITS.median())



x.YR_RMDL = x.YR_RMDL.fillna(x.YR_RMDL.median())



x.LIVING_GBA = x.LIVING_GBA.fillna(x.LIVING_GBA.mean())



x.STYLE = x.STYLE.fillna(x.STYLE.mode()[0])



x.STRUCT = x.STRUCT.fillna(x.STRUCT.mode()[0])



x.GRADE = x.GRADE.fillna(x.GRADE.mode()[0])



x.CNDTN = x.CNDTN.fillna(x.CNDTN.mode()[0])



x.EXTWALL = x.EXTWALL.fillna(x.EXTWALL.mode()[0])



x.ROOF = x.ROOF.fillna(x.ROOF.mode()[0])



x.INTWALL = x.INTWALL.fillna(x.INTWALL.mode()[0])



x.ASSESSMENT_SUBNBHD  = x.ASSESSMENT_SUBNBHD.fillna(x.ASSESSMENT_SUBNBHD.mode()[0])



x.isnull().sum()
# converting categorical variables into dummies

from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()



cat = len(x.select_dtypes(include=['object']).columns)

num = len(x.select_dtypes(include=['int64','float64']).columns)

print('Total Features: ', cat, 'categorical', '+',

      num, 'numerical', '=', cat+num, 'features') 

cols = x.select_dtypes(include=['object']).columns

for c in cols:

    lbl = LabelEncoder() 

    lbl.fit(list(x[c].values)) 

    x[c] = lbl.transform(list(x[c].values))

x.head() 

 

# create training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

print(X_train.shape, y_train.shape)

print(X_test.shape, y_test.shape)
X_train.dtypes.sample(20)
# One Hot Encoder try running this

'''

import pandas as pd

one_hot_encoded_training_predictors = pd.get_dummies(X_train)

one_hot_encoded_test_predictors = pd.get_dummies(X_test)

final_train, final_test = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors,

                                                                    join='left', 

                                                                    axis=1)

'''
'''

Build a best fit regression model using Ordinary Least Squares method such that 

you get acceptable performance (RMSE, R-squared, etc. ) on train data.

'''
# fitting linear regression model on training dataset

'''

from sklearn import datasets, linear_model

lm = linear_model.LinearRegression()

model = lm.fit(X_train, y_train)

predictions = lm.predict(X_test)

predictions

'''
# OLS Regression 

import statsmodels.api as sm

from sklearn.metrics import r2_score,mean_squared_error



Pricing_model = sm.OLS(y_train,X_train)

result = Pricing_model.fit()

print(result.summary())

print("RMSE: ",np.sqrt(mean_squared_error(result.fittedvalues,y_train)))

print("r2_score: ",r2_score(y_train, result.fittedvalues))

#ridge_regression

from sklearn.linear_model import Lasso,Ridge,RidgeCV

ridgeReg = Ridge(alpha=5, normalize=True)

ridgeReg.fit(X_train,y_train)

pred = ridgeReg.predict(X_test)

mse = np.sqrt(mean_squared_error(pred , y_test ))

print("The Root mean square error of Ridge Regression is ", mse)

print("The R2 value of Ridge Regression is ",r2_score(y_test ,pred))