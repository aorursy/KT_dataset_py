import pandas as pd
import numpy as np
train = pd.read_csv('../input/train.csv',encoding='big5',header=0)
test = pd.read_csv('../input/test.csv',encoding='big5',header=None)
print(train.shape)
print(test.shape)
display(train.head())
display(test.head())
print(train.columns)
print(test.columns)
print(train['測項'].unique())
print(test[1].unique())
train_PM25 = train[train['測項'] == 'PM2.5'].reset_index(drop=True)
display(train_PM25.head())
columns = []
for i in range(24):
    columns.append(str(i))
print(columns)
PM25_TimeSeries = train_PM25[['日期']+columns]
PM25_TimeSeries = PM25_TimeSeries.set_index(keys='日期',drop=True)
PM25_TimeSeries = PM25_TimeSeries.stack().reset_index()
PM25_TimeSeries.columns = ['date','hour','PM25']
PM25_TimeSeries['date'] = pd.to_datetime(PM25_TimeSeries['date'])
PM25_TimeSeries['hour'] = pd.to_numeric(PM25_TimeSeries['hour'])
PM25_TimeSeries = PM25_TimeSeries.sort_values(by=['date','hour'])
display(PM25_TimeSeries.head())
x_data = np.zeros((len(PM25_TimeSeries)-9,9))
y_data = np.zeros(len(PM25_TimeSeries)-9)
for i in range(len(PM25_TimeSeries)-9):
    x_data[i,:] = PM25_TimeSeries['PM25'][i:i+9]
    y_data[i] = PM25_TimeSeries['PM25'][i+9]
print(x_data[-3:,:],y_data[-3:])
print(PM25_TimeSeries['PM25'][-13:])
def LinearRegression(x_data,y_data,weight,bias,epochs,alpha):
    num_sample = len(x_data)
    random_index = np.random.permutation(num_sample)
    y_pred = np.zeros_like(y_data)
    loss = np.zeros_like(y_data)
    for epoch in range(epochs):
        for idx in random_index:
            y_pred[idx] = np.dot(x_data[idx],weight) + bias
            loss[idx] = np.square(y_pred[idx]-y_data[idx])/2.
            weight -= alpha * (y_pred[idx]-y_data[idx])*x_data[idx]
            bias -= alpha * (y_pred[idx]-y_data[idx])
        print('Epoch = {}, the mean_squared_error = {}'.format(epoch,np.mean(loss)))
    return weight,bias,y_pred
x_max = np.max(x_data,axis=0)
x_min = np.min(x_data,axis=0)
x_data = (x_data-x_min)/(x_max-x_min)
print(np.max(x_data,axis=0))
print(x_data.shape)
print(y_data.shape)
weight_0 = np.random.randn(x_data.shape[1])
weight,bias,y_pred = LinearRegression(x_data,y_data,weight=weight_0,bias=0.,epochs=20,alpha=0.3)
print(y_data)
print(y_pred)
test_columns = [0]
for i in range(2,11):
    test_columns.append(i)
print(test_columns)
test_PM25 = test[test[1]=='PM2.5'][test_columns].reset_index(drop=True)
display(test_PM25)
test_x_columns = []
for i in range(2,11):
    test_x_columns.append(i)
print(test_x_columns)
x_test = test_PM25[test_x_columns].astype('float').values
x_test = (x_test-x_min)/(x_max-x_min)
y_test_pred = np.dot(x_test,weight) + bias
ans = pd.DataFrame()
ans['id'] = test_PM25[0]
ans['value'] = y_test_pred
display(ans)
ans.to_csv('ans.csv',index=False)
