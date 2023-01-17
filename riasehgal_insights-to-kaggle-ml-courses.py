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
diamond_file_path='/kaggle/input/diamonds/diamonds.csv'

diamond_data=pd.read_csv(diamond_file_path)

diamond_data.columns
y=diamond_data.price

#Target
#Numerical predictors used

diamond_predictors=diamond_data.drop(['price'],axis=1)

X=diamond_predictors.select_dtypes(exclude=['object'])
#Divide data into training and validation subset

from sklearn.model_selection import train_test_split

X_train,X_valid,y_train,y_valid=train_test_split(X,y,train_size=0.8,test_size=0.2,random_state=0)
#Imputation turns out to be the best way to dea with missing data

#for reference,mean absolute error in the same is added using both drop values and imputations

#But first we check if there is any missing data or so

print(diamond_data.isnull().sum())
#Andd... no missing value.So we move on.

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

d_model=RandomForestRegressor(random_state=1)

d_model.fit(X_train,y_train)

d_val_predictions=d_model.predict(X_valid)

d_val_mae=mean_absolute_error(d_val_predictions,y_valid)

print("Validation MAE for Random Forest Model: {}".format(d_val_mae))