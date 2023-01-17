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
read=pd.read_csv("../input/graduate-admissions/Admission_Predict_Ver1.1.csv")
read.head()
read.columns
len(read)

read.index
train_data=read[0:400]

test_data=read[400:]

test_y=test_data['Chance of Admit ']

test_data.drop(['Chance of Admit '],axis=1)

test_data.head
train_X=train_data

#type(train_X)

train_X.drop(['Chance of Admit '],axis=1)

train_Y=train_data["Chance of Admit "]

train_X.head
from xgboost import XGBRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

model= XGBRegressor(n_estimators=500)

#model1= DecisionTreeRegressor(random_state=1)

#model2=RandomForestRegressor(random_state=1)

model.fit(train_X,train_Y)

y_predictions=model.predict(test_data)
from sklearn.metrics import mean_absolute_error

x=mean_absolute_error(y_predictions,test_y)
import math

print(str(math.sqrt(x)))