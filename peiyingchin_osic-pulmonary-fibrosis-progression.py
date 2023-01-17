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
df_train = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')

df_test = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
print('train_dim',df_train.shape)

print('test_dim',df_test.shape)
print(df_train.head())
print(df_test.head())
print('Total patients:',len(df_train.Patient.unique()))
print(df_train.groupby(['Patient','Weeks'])['Weeks'].count().head(20))
print(df_train.groupby(['Weeks'])['Patient'].count().head(20))
import matplotlib.pyplot as plt
df_train.columns
plt.hist(df_train['FVC'],bins=30)

plt.plot()
df_train['Patient'].unique()[1:5]
df_train.columns
import matplotlib.pyplot as plt

for i in df_train['Patient'].unique()[0:5]:

    plt.figure(figsize=(20, 4))

    plt.plot(df_train[df_train['Patient']==i]['Weeks'],df_train[df_train['Patient']==i]['FVC'])

    plt.ylabel('FVC')

    plt.xlabel('weeks')

    a=df_train[df_train['Patient']==i]['Weeks'].max() - df_train[df_train['Patient']==i]['Weeks'].min()

    plt.title(i +' FVC in '+str(a)+' Weeks')

    plt.show()
df_train.groupby(['Patient'])['Weeks'].count().describe()
df_train.columns
plt.figure(figsize=(20, 14))

plt.subplot(2,2,1)

plt.title('Mean Age versus FVC')

plt.plot(df_train.groupby(['Age'])['FVC'].mean())

plt.subplot(2,2,2)

plt.title('Min Age versus FVC')

plt.plot(df_train.groupby(['Age'])['FVC'].min())

plt.subplot(2,2,3)

plt.title('Max Age versus FVC')

plt.plot(df_train.groupby(['Age'])['FVC'].max())
df_train.Sex.value_counts()
plt.plot(df_train.Age)


df_train
plt.scatter(df_train.FVC,df_train.Percent,  color='gray')

plt.show()
sub=pd.DataFrame()

sub['patient']=pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv')['Patient_Week'].str.split('_', n = 1, expand = True)[0]

sub['week']=pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv')['Patient_Week'].str.split('_', n = 1, expand = True)[1]
sub['patient'].unique()
sub['week'].unique()
sub.shape
features= df_train.drop(['Patient','FVC','Percent'],axis=1)

X=pd.get_dummies(features)

y=df_train.FVC

#y=df_train.Percent
from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression

from sklearn import metrics

%matplotlib inline

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=11)

regressor = LinearRegression()  

regressor.fit(X_train, y_train) #training the algorithm

y_pred = regressor.predict(X_test)

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

plt.scatter(y_pred, y_test,  color='gray')

plt.show()

print('Mean Absolute Error:',metrics.mean_absolute_error(y_test, y_pred))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# from sklearn.linear_model import Ridge

# from sklearn.ensemble import AdaBoostRegressor

# from sklearn.ensemble import GradientBoostingRegressor

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)

# ls=[]

# lr=[]

# mae=[]

# mse=[]

# rmse=[]

# estimator=[]

# mss=[]

# ss=[]

# md=[]

# for loss in ['ls', 'lad', 'huber', 'quantile']:

#     for learning_rate in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:

#         for n_estimators in [10,20,30,50,100,120]:

#             for subsample in [0.1,0.2,0.3,0.5,0.7,0.9]:

#                 for min_samples_split in [2,3,5,7,10,12]:

#                     for max_depth in [3,5,7,10,12]:

#                         ls.append(loss)

#                         lr.append(learning_rate)

#                         estimator.append(n_estimators)

#                         ss.append(subsample)

#                         mss.append(min_samples_split)

#                         md.append(max_depth)

#                         clf = GradientBoostingRegressor(random_state=0,loss=loss,learning_rate=learning_rate,n_estimators=n_estimators,

#                                                        subsample=subsample,min_samples_split=min_samples_split,max_depth=max_depth)

#                         clf.fit(X_train, y_train) #training the algorithm

#                         y_pred = clf.predict(X_test)

#                         #df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

#                         #plt.scatter(y_pred, y_test,  color='gray')

#                         #plt.show()

#                         mae.append(metrics.mean_absolute_error(y_test, y_pred))

#                         mse.append(metrics.mean_squared_error(y_test, y_pred))

#                         rmse.append( np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)

clf = GradientBoostingRegressor(random_state=0)

clf.fit(X_train, y_train) #training the algorithm

y_pred = clf.predict(X_test)

plt.scatter(y_pred, y_test,  color='gray')

plt.show()

print('Mean Absolute Error:',metrics.mean_absolute_error(y_test, y_pred))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
test_mod=pd.DataFrame()

pat=[]

for i in df_test.Patient:

    for j in sub['week'].unique(): 

        pat.append(i)

        df_test.loc[df_test.Patient==i, 'Weeks'] = j

        test_mod=test_mod.append(df_test[df_test.Patient == i])

test_mod['Weeks']=test_mod['Weeks'].astype(str).astype(int)

test=pd.get_dummies(test_mod.drop(['Patient','FVC','Percent'],axis=1))

test2=test.drop(['SmokingStatus_Ex-smoker','SmokingStatus_Never smoked','Sex_Male'],axis=1)

test2['Sex_Female']=0

test2['Sex_Male']=test['Sex_Male']

test2['SmokingStatus_Currently smokes']=0

test2['SmokingStatus_Ex-smoker']=test['SmokingStatus_Ex-smoker']

test2['SmokingStatus_Never smoked']=test['SmokingStatus_Never smoked']
test2['pred']=regressor.predict(test2)

test2['Patient']=pat
pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv').head()
test2['Patient_Week']=test2["Patient"].str.cat(test2['Weeks'].astype(str),sep='_') 
submission=pd.DataFrame()

submission['Patient_Week']=test2['Patient_Week']

submission['FVC']=test2['pred']

submission['Confidence']=100

submission.to_csv('submission.csv',index=False)
# result=pd.DataFrame()

# result['loss']=ls

# result['learning_rate']=lr

# result['estimator']=estimator

# result['subsample']=ss

# result['min_samples_split']=mss

# result['max_depth']=md

# result['mae']=mae

# result['mse']=mse

# result['rmse']=rmse
#result[result.mae==result.mae.min()]
# from sklearn.ensemble import RandomForestRegressor

# clf =  RandomForestRegressor(max_depth=2, random_state=0)

# clf.fit(X_train, y_train) #training the algorithm

# y_pred = clf.predict(X_test)

# df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

# plt.scatter(y_pred, y_test,  color='gray')

# plt.show()

# print('Mean Absolute Error:',metrics.mean_absolute_error(y_test, y_pred))  

# print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  

# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# test=pd.get_dummies(df_test.drop(['Patient','FVC','Percent'],axis=1))

# test2=test.drop(['SmokingStatus_Ex-smoker','SmokingStatus_Never smoked','Sex_Male'],axis=1)

# test2['Sex_Female']=0

# test2['Sex_Male']=test['Sex_Male']

# test2['SmokingStatus_Currently smokes']=0

# test2['SmokingStatus_Ex-smoker']=test['SmokingStatus_Ex-smoker']

# test2['SmokingStatus_Never smoked']=test['SmokingStatus_Never smoked']