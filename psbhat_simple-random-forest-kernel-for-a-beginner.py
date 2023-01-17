# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from sklearn.ensemble import RandomForestRegressor

#Read Train Data
dt_train = pd.read_csv("../input/train.csv")

#Read Test Data
dt_test = pd.read_csv("../input/test.csv")
#Build a Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100,verbose=10,n_jobs=-1).fit(dt_train.iloc[:,2:4993].values,dt_train['target'])

predicted = rf_model.predict(dt_test.iloc[:,1:4992].values)
#Output Formatting
output = pd.DataFrame()
output['ID'] = dt_test['ID']
output['target'] = predicted
output.to_csv("ouput.csv",index=False)