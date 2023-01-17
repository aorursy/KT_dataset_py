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
location_train='../input/mobile-price-classification/train.csv'

phone_data_train=pd.read_csv(location_train)

location_test='../input/mobile-price-classification/test.csv'

phone_data_test=pd.read_csv(location_test)
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt



phone_data_train.head()

print(phone_data_train['price_range'].unique())
train_label=phone_data_train['price_range']

train_features=phone_data_train.drop('price_range',axis=1)
X_train,X_test,y_train,y_test=train_test_split(train_features,train_label,test_size=0.2,random_state=0)
X_train.head()
import seaborn as  sns
from sklearn.naive_bayes import GaussianNB

model_NB=GaussianNB()

model_NB.fit(X_train,y_train)

predictions=model_NB.predict(X_test)
from sklearn.metrics import mean_absolute_error

print("MAE: ",mean_absolute_error(predictions,y_test))

different_predictions=list(predictions-y_test)

different_predictions
from collections import Counter

print(Counter(different_predictions))
from sklearn.ensemble import RandomForestRegressor

model_RF=RandomForestRegressor(n_estimators=10)

model_RF.fit(X_train,y_train)

predictions_RF=model_RF.predict(X_test)

predictions_RF
print("MEA :", mean_absolute_error(predictions_RF,y_test))

difference_RF=list(predictions_RF-y_test)

print(Counter(difference_RF))
from sklearn.cluster import KMeans

model_CL=KMeans()

model_CL.fit(X_train,y_train)

predictions_CL=model_CL.predict(X_test)
print("MEA for clustering :",mean_absolute_error(predictions_CL,y_test))

difference_CL=list(predictions_CL-y_test)

print(Counter(difference_CL))
from sklearn.cluster import MeanShift

model_MF=MeanShift()

model_MF.fit(X_train,y_train)

predictions_MF=model_MF.predict(X_test)

print(mean_absolute_error(predictions_MF,y_test))

differences_MF=list(predictions_MF-y_test)

print(Counter(differences_MF))
phone_data_train.head()
%matplotlib inline

sns.barplot(x=phone_data_train['price_range'],y=phone_data_train['clock_speed'])

sns.barplot(x=phone_data_train['price_range'],y=phone_data_train['ram'])

valuable_info=['ram','battery_power',]
sns.barplot(x=phone_data_train['price_range'],y=phone_data_train['touch_screen'])
print(phone_data_test.shape)

print(phone_data_train.shape)
phone_data_test.columns
phone_data_train.columns