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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import math



import statsmodels.api as sm

from statsmodels.tsa.api import VAR #Vector AutoRegression



from statsmodels.tsa.stattools import adfuller #for the Dicky-Fuller Test

from sklearn.metrics import mean_squared_error #for calculating the performance metric
%matplotlib inline
# read all dataset

df_train = pd.read_csv('../input/into-the-future/train.csv')

df_test = pd.read_csv('../input/into-the-future/test.csv')
df_train.head()
df1=df_train[['time','feature_1','feature_2']]

df1.head()
df_train.describe()
df_test.info()
df_test.describe()
df1.set_index('time', inplace=True)

df1.plot(figsize=(12,8))
df1_train=df1[:int(0.85*len(df1))]

df1_test=df1[int(0.85*len(df1)):]
def adfuller_test(data):

    result=adfuller(data,autolag='AIC')

    labels=["ADF Test Statistic","p-value","#Lags used","#Observations"]

    for val,lab in zip(result,labels):

        print(lab+":"+str(val))

    #taking the significance value as 0.05

    if result[1]>0.05 :

        print("Model is Not Stationary") #Null Hypothesis

    else:

        print("Model is Stationary") #Alternate Hypothesis

        



print("FEATURE 1 ")

adfuller_test(df1_train['feature_1'])

print("FEATURE 2 ")

adfuller_test(df1_train['feature_2'])
model = VAR(df1_train)

results = model.fit(maxlags=20, ic='aic')

results.summary()
predicted = results.forecast(results.y, steps=len(df1_test))



labels=['feature_1','feature_2']

predicted=pd.DataFrame(predicted, columns=labels)
print(predicted.shape)

df1_test.shape
for i in labels:

    print('rmse for '+i+' is : '+str(math.sqrt(mean_squared_error(predicted[i],df1_test[i]))))
plt.plot(predicted['feature_2'])

plt.plot(df1_test['feature_2'])

plt.show()
df_test=pd.read_csv('../input/into-the-future/test.csv')

print(df_test.head())

df_test.shape
data_test=df_test[['time','feature_1']]

data_test.set_index('time', inplace=True)

data_test.head()
final_prediction = results.forecast(results.y, steps=len(data_test)+len(df1_test))

final_prediction.shape
final_prediction1 = pd.DataFrame(final_prediction,columns=['feature_1','feature_2'],index=range(len(df1_test), len(df1_test)+len(final_prediction), 1))

print(final_prediction1.shape)



final_prediction2=final_prediction1[len(df1_test):]

final_prediction2.shape

print('rmse for '+i+' is : '+str(math.sqrt(mean_squared_error(final_prediction2['feature_1'],data_test['feature_1']))))



plt.plot(final_prediction2['feature_1'])

plt.plot(data_test['feature_1'])

plt.show()
final_prediction2['id'] = index=range(564, 564+len(final_prediction2), 1)

final_prediction2.set_index('id',inplace=True)

final_sol =final_prediction2.drop(['feature_1'],1)



final_sol
final_sol.to_csv('Submission.csv')