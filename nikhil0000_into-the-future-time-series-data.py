#importing nesseray common libraries

import pandas as pd

import matplotlib.pyplot as plt

import numpy as n

from datetime import datetime

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df_train = pd.read_csv(r"../input/into-the-future/train.csv")

df_test = pd.read_csv(r'../input/into-the-future/test.csv')

df_train_originall = df_train.copy()

df_test_originall = df_test.copy()
print('Train set',df_train.columns)

print('Test set',df_test.columns)
df_train.dtypes
df_test.dtypes
df_train['Datetime'] = pd.to_datetime(df_train['time'],format='%Y-%m-%d %H:%M:%S')

df_test['Datetime'] = pd.to_datetime(df_test['time'],format='%Y-%m-%d %H:%M:%S')
df_train = df_train.drop('time',1)

df_test = df_test.drop('time',1)
df_train_original = df_train.copy()

df_test_original = df_test.copy()
for i in (df_train,df_test,df_test_original,df_train_original):

    i['year'] = i.Datetime.dt.year

    i['Month'] = i.Datetime.dt.month

    i['Hour'] = i.Datetime.dt.hour

    i['day'] = i.Datetime.dt.day

# df_train['week_days'] = df_train['Datetime'].dt.dayofweek

for i in (df_train,df_test,df_test_original,df_train_original):

    i['minute'] = i.Datetime.dt.minute

df_train['minute'] = df_train['Datetime'].dt.minute
def applyer(row):

    if row.dayofweek == 5 or row.dayofweek == 6:

        return 1

    else:

        return 0

temp2 = df_train['Datetime'].apply(applyer)

df_train['weekend'] = temp2
df_train.index = df_train['Datetime']

df = df_train.drop('id' , 1) 
ts = df['feature_2']

plt.figure(figsize=(16,8)) 



plt.plot(ts,label='Feature_2')

# plt.plot(df['feature_1'],label='Other_feture')

plt.title('Time Series') 

plt.xlabel("Time") 

plt.ylabel("number_count") 

plt.legend()
import seaborn as sns

sns.jointplot(x='feature_1',y='feature_2',data=df_train)
df_train.groupby('year')['feature_2'].mean().plot.bar()
df_train.groupby('Hour')['feature_2'].mean().plot.bar()
df_train.groupby(['Month','minute'])['feature_2'].mean().plot.bar()
df_train = df_train_originall

df_test = df_test_originall
df_train['time'] = pd.to_datetime(df_train['time'],format='%Y-%m-%d %H:%M:%S')

data = df_train.drop('time',1)

data.index = df_train['time']

data = data.drop('id',1)





#missing value treatment

cols = data.columns

for j in cols:

    for i in range(0,len(data)):

       if data[j][i] == -200:

           data[j][i] = data[j][i-1]
train = data[:int(0.8*(len(data)))]

valid = data[int(0.8*(len(data))):]

ttrain = train.copy()

vvalid = valid.copy()
from statsmodels.tsa.vector_ar.var_model import VAR

import numpy as np



model = VAR(endog=np.asarray(train))

model_fit = model.fit()

prediction = model_fit.forecast(model_fit.y, steps=len(valid))

valid.index = range(0,len(valid))
pred = pd.DataFrame(index=range(0,len(prediction)),columns=[cols])

for j in range(0,2):

    for i in range(0, len(prediction)):

       pred.iloc[i][j] = prediction[i][j]



#check RMSE

import math  

from sklearn.metrics import mean_squared_error

pred.columns = [x[0] for x in pred.columns]

for i in cols:

#     print(pred[i], valid[i])

    print('rmse value for', i, 'is : ', math.sqrt(mean_squared_error(pred[i], valid[i])))
#let's chenge our valid index to original index.

valid.index = vvalid.index

pred.index = vvalid.index
plt.plot(pred['feature_2'])

plt.plot(train['feature_2'])

plt.plot(valid.iloc[:,1:2], '--')

plt.show()
# # let's perform the same preprocessing step on train data as we performed on Training set

df_test['time'] = pd.to_datetime(df_test['time'],format='%Y-%m-%d %H:%M:%S')

pred = df_test.drop('time',1)

pred.index = df_test['time']

pred = pred.drop('id',1)

cols = data.columns
prediction = model_fit.forecast(model_fit.y, steps=len(pred))
if len(prediction) == len(df_test):

    prediction = pd.DataFrame(prediction,columns=['feature_1','feature_2'],index=range(0, len(prediction), 1))

    pred['feature_2'] = list(prediction['feature_2'])

    print("Length mached")

else:

    print("Length Does Not Matched")
plt.plot(pred['feature_2'])

sns.jointplot(x='feature_1',y='feature_2',data=valid)
prediction['id'] = index=range(564, 564+len(prediction), 1)

prediction.index = prediction['id']

prediction =prediction.drop(['id','feature_1'],1)
prediction
prediction.to_csv('Solution.csv')
df_train = pd.read_csv(r"../input/into-the-future/train.csv")

df_test = pd.read_csv(r'../input/into-the-future/test.csv')
df_train['time'] = pd.to_datetime(df_train['time'],format='%Y-%m-%d %H:%M:%S')

data = df_train.drop('time',1)

data.index = df_train['time']

data = data.drop('id',1)





#missing value treatment

cols = data.columns

for j in cols:

    for i in range(0,len(data)):

       if data[j][i] == -200:

           data[j][i] = data[j][i-1]



train = data[:int(0.8*(len(data)))]

valid = data[int(0.8*(len(data))):]
from statsmodels.tsa.statespace.varmax import VARMAX



model = VARMAX(train, order = (1,2))

model_fit = model.fit()
predictions_multi = model_fit.forecast( steps=len(valid))
plt.plot(train['feature_2'],label='Train')

plt.plot(valid['feature_2'],label = 'valid')

plt.plot(predictions_multi.iloc[:,1:2], '--',label= 'predictions')



plt.title('Time Series') 

plt.xlabel("Time") 

plt.ylabel("feature_2") 

plt.legend()

plt.show()
predictions_multi.columns = valid.columns

for i in cols:

    

#     print(pred[i], valid[i])

    print('rmse value for', i, 'is : ', math.sqrt(mean_squared_error(predictions_multi[i], valid[i])))