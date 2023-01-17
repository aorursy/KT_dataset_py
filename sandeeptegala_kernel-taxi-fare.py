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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

file=open("/kaggle/input/taxi-fare/taxi-fare-train.txt")
df=pd.read_csv(file)
df.head()
df.drop(["trip_time_in_secs"],axis = 1,inplace=True )
df.head()
df['vendor_id'].replace(to_replace=['CMT', 'VTS'], value=[0,1], inplace=True)
df.head()
X=df
regr=linear_model.LinearRegression()
#Feature set X
X=df[['vendor_id','rate_code','passenger_count','trip_distance']]
X.head()
#Feature set y
y=df[['fare_amount']]
y.head()

#Spilting the data set into train and test sets for evaluation

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

regr.fit(X_train,y_train)

print('Coefficients:', regr.coef_)

#Predicted taxi fare
y_pred=regr.predict(X_test)

#Higher R2 score more accurate the model is!
print("R2-Score:%.2f" % r2_score(y_pred,y_test))