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
train=pd.read_csv("/kaggle/input/cs419-assgmt1-2019-20/train.csv")
train.head()
pickup=train['pickup_datetime'].values
pickup
dates=[]
for i in pickup:
    dates.append(i.split())
dates
days=[]
for date in dates:
    days.append(date[0])
days
day_string=[]
for i in days:
    day_string.append(i.split('-')[2])
day=[]
for i in day_string:
    day.append(int(i))
day
month_string=[]
for i in days:
    month_string.append(i.split('-')[1])
month=[]
for i in month_string:
    month.append(int(i))
month
hour_string=[]
for i in dates:
    hour_string.append(i[1].split(':')[0])
hour=[]
for i in hour_string:
    hour.append(int(i))
hour
train['hour']=hour
train.head()
train['day']=day
train['month']=month
train.head()
y=train['fare_amount'].values
y
train.drop(['pickup_datetime'],axis=1, inplace=True)
train.drop(['fare_amount'],axis=1, inplace=True)
train.head()
x0=[]
for i in range(60000):
    x0.append(1)
train['x0']=x0
train.head()
cols = list(train.columns.values)
cols
train = train[['x0','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','passenger_count','hour','day','month']]
train.head()

phi=train.values
phi
phi=train.values
phi
alpha=0.000001
n=10000
from numpy.random import randn
weights=np.zeros(len(phi[0]))
# phi=(phi-phi.mean())/np.var(phi)
prediction=np.dot(phi,weights)
prediction
weights
phi
for i in range(n):
    print("iteration "+str(i))
    prediction=np.dot(phi,weights)
    print("prediction = "+str(prediction))
    weights=weights-alpha*np.dot(phi.T,prediction-y)/len(y)
    print("weights = " + str(weights))
    print('\n')
weights
prediction
y
MSE = np.square(np.subtract(y,prediction)).mean() 
np.sqrt(MSE)
dev=pd.read_csv("/kaggle/input/cs419-assgmt1-2019-20/dev.csv")
def get_features(file_path):
	# Given a file path , return feature matrix and target labels 
	
	data=pd.read_csv(file_path)
	pickup=data['pickup_datetime'].values
	dates=[]
	for i in pickup:
		dates.append(i.split())
	days=[]
	for date in dates:
		days.append(date[0])
	day_string=[]
	for i in days:
		day_string.append(i.split('-')[2])
	day=[]
	for i in day_string:
		day.append(int(i))
	month_string=[]
	for i in days:
		month_string.append(i.split('-')[1])
	month=[]
	for i in month_string:
		month.append(int(i))
	hour_string=[]
	for i in dates:
		hour_string.append(i[1].split(':')[0])
	hour=[]
	for i in hour_string:
		hour.append(int(i))
	data['hour']=hour
	data['day']=day
	data['month']=month
	y=data['fare_amount'].values
	data.drop(['pickup_datetime'],axis=1, inplace=True)
	data.drop(['fare_amount'],axis=1, inplace=True)
	x0=[]
	for i in range(len(y)):
		x0.append(1)
	data['x0']=x0
	data = data[['x0','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','passenger_count','hour','day','month']]

	phi=data.values

	return phi, y
phi_dev,y_dev=get_features("/kaggle/input/cs419-assgmt1-2019-20/dev.csv")
phi_dev.shape
y_dev.shape
prediction_dev=np.dot(phi_dev,weights)
MSE_dev = np.square(np.subtract(y_dev,prediction_dev)).mean() 
np.sqrt(MSE_dev)
test=pd.read_csv("/kaggle/input/cs419-assgmt1-2019-20/test.csv")
test.head()
def get_features_test(file_path):
	# Given a file path , return feature matrix and target labels 
	
	data=pd.read_csv(file_path)
	pickup=data['pickup_datetime'].values
	dates=[]
	for i in pickup:
		dates.append(i.split())
	days=[]
	for date in dates:
		days.append(date[0])
	day_string=[]
	for i in days:
		day_string.append(i.split('-')[2])
	day=[]
	for i in day_string:
		day.append(int(i))
	month_string=[]
	for i in days:
		month_string.append(i.split('-')[1])
	month=[]
	for i in month_string:
		month.append(int(i))
	hour_string=[]
	for i in dates:
		hour_string.append(i[1].split(':')[0])
	hour=[]
	for i in hour_string:
		hour.append(int(i))
	data['hour']=hour
	data['day']=day
	data['month']=month
# 	y=data['fare_amount'].values
	data.drop(['pickup_datetime'],axis=1, inplace=True)
# 	data.drop(['fare_amount'],axis=1, inplace=True)
	x0=[]
	for i in range(len(data['hour'])):
		x0.append(1)
	data['x0']=x0
	data = data[['x0','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','passenger_count','hour','day','month']]

	phi=data.values

	return phi
phi_test=get_features_test("/kaggle/input/cs419-assgmt1-2019-20/test.csv")
phi_test.shape
prediction_test=np.dot(phi_test,weights)
prediction_test
# save numpy array as csv file
from numpy import asarray
from numpy import savetxt
# define data
data = asarray(prediction_test)
# save to csv file
savetxt('test_results.csv', data, delimiter=',')
weights_sgd=np.zeros(len(phi[0]))
alpha_sgd=0.01
m=len(y)
for i in range(10):
    for j in range(m):
        print("iteration "+str(100*i + j))
        index=np.random.randint(0,m)
        x_index=phi[index]
        y_index=y[index]
        pred_sgd=np.dot(x_index,weights_sgd)
        print("pred_sgd = "+str(pred_sgd))
        weights_sgd=weights_sgd-alpha_sgd*np.dot(x_index,pred_sgd-y_index)/m
        print("weights_sgd = "+str(weights_sgd))
        print('\n')
weights_sgd
pred_sgd=np.dot(phi,weights_sgd)
pred_sgd
MSE_sgd = np.square(np.subtract(y,pred_sgd)).mean() 
np.sqrt(MSE_sgd)
pred_dev_sgd=np.dot(phi_dev,weights_sgd)
MSE_dev_sgd = np.square(np.subtract(y_dev,pred_dev_sgd)).mean() 
np.sqrt(MSE_dev_sgd)
pred_test_sgd=np.dot(phi_test,weights_sgd)
pred_test_sgd
data = asarray(pred_test_sgd)
savetxt('test_results_sgd.csv', data, delimiter=',')
weights_pnorm=[]
p=2
lam=1
n=1000
for i in range(len(weights)-1):
    weights_pnorm.append(weights[i+1])
for i in range(n):
    print("iteration "+str(i))
    prediction=np.dot(phi,weights_pnorm)
    print("prediction = "+str(prediction))
    weights=weights-alpha*(np.dot(phi.T,prediction-y) + np.full((9,),p*lam*np.sum(np.power(weights_pnorm,p-1))))/len(y)
    print("weights = " + str(weights_pnorm))
    print('\n')
p*lam*np.sum(np.power(weights_pnorm,p-1))/len(y)
