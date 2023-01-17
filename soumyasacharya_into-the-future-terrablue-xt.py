

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import math



import statsmodels.api as sm

from statsmodels.tsa.api import VAR #Vector AutoRegression



from statsmodels.tsa.stattools import adfuller #for the Dicky-Fuller Test

from sklearn.metrics import mean_squared_error #for calculating the performance metric



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



df_train=pd.read_csv("../input/into-the-future/train.csv")

df_train.head()
df1=df_train[['time','feature_1','feature_2']]

df1.head()
df1.set_index('time', inplace=True)

df1.plot(figsize=(12,8))
df1.info()
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

results = model.fit(maxlags=15, ic='aic')

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
final_sol.to_csv('Final_Solution.csv')