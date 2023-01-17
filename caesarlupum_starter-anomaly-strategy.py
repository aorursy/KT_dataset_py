import datetime as dt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import math
import matplotlib.pyplot as plt

from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA

from sklearn.metrics import mean_squared_error
# Any results you write to the current directory are saved as output.

for dirname, _, filenames in os.walk('/kaggle/input/nab/realTraffic/realTraffic/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
dirname0 = "/kaggle/input/nab/realTraffic/realTraffic/"

filename0 = "TravelTime_387.csv"

dataframe = pd.read_csv(dirname0+filename0)#, usecols=[1])#, skipfooter=3)

dataframe.head()
def remap(x, in_min, in_max, out_min, out_max):

    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min



dataframe['prop_value'] = remap( dataframe.value.values, np.min (dataframe.value), np.max( dataframe.value), 0.0, 100.0  )

dataframe.head()
!pip install pyod
from pyod.models.ocsvm import OCSVM   #PyOD is a comprehensive and scalable Python toolkit for detecting outlier objects



random_state = np.random.RandomState(42)     # A fixed values is assigned, then no matter how many time you execute your code,values generated would be the same

#Does this mean that later on the code the outliers 5% higher than maximum value of dataset?

outliers_fraction = 0.05

classifiers = {

        'One Classify SVM (SVM)':OCSVM(kernel='rbf', degree=3, gamma='auto', coef0=0.0, tol=0.001, nu=0.5, shrinking=True, cache_size=200, verbose=False, max_iter=-1, contamination=outliers_fraction)

}
X = dataframe['value'].values.reshape(-1,1)
from scipy import stats

for i, (clf_name, clf) in enumerate(classifiers.items()):

    clf.fit(X)

    # predict raw anomaly score

    scores_pred = clf.decision_function(X) * -1        

    # prediction of a datapoint category outlier or inlier

    y_pred = clf.predict(X)

    n_inliers = len(y_pred) - np.count_nonzero(y_pred)

    n_outliers = np.count_nonzero(y_pred == 1)

    

    # copy of dataframe

    dfx = dataframe[['value','prop_value']]

    dfx['outlier'] = y_pred.tolist()

    IX1 =  np.array(dfx['value'][dfx['outlier'] == 0]).reshape(-1,1)

    OX1 =  dfx['value'][dfx['outlier'] == 1].values.reshape(-1,1)         

    print('OUTLIERS : ',n_outliers,'INLIERS : ',n_inliers, clf_name)        

    # threshold value to consider a datapoint inlier or outlier

    threshold = stats.scoreatpercentile(scores_pred,100 * outliers_fraction)

y = dfx['outlier'].values.reshape(-1,1)
tOut = stats.scoreatpercentile(dfx[dfx['outlier'] == 1]['value'], np.abs(threshold))
def severity_validation():

    tOUT10 = tOut+(tOut*0.10)    

    tOUT23 = tOut+(tOut*0.23)

    tOUT45 = tOut+(tOut*0.45)

    dfx['test_severity'] = "None"

    for i, row in dfx.iterrows():

        if row['outlier']==1:

            if row['value'] <=tOUT10:

                dfx['test_severity'][i] = "Low Severity" 

            elif row['value'] <=tOUT23:

                dfx['test_severity'][i] = "Medium Severity" 

            elif row['value'] <=tOUT45:

                dfx['test_severity'][i] = "High Severity" 

            else:

                dfx['test_severity'][i] = "Ultra High Severity" 



severity_validation()
dfx.head()
print(" inline values proportion mean ",dfx[dfx['outlier']==0]['prop_value'].mean())

print(" outlier values proportion mean",dfx[dfx['outlier']==1]['prop_value'].mean())

print(" outlier values max ",dfx[dfx['outlier']==1]['prop_value'].min())



    

dirname = "/kaggle/input/nab/realTraffic/realTraffic/"

filename = "speed_t4013.csv"



dataframe = pd.read_csv(dirname+filename)#, usecols=[1])#, skipfooter=3)

dataframe.head()
# Any results you write to the current directory are saved as output.

for dirname, _, filenames in os.walk('/kaggle/input/nab/realKnownCause/realKnownCause/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))


for dirname, _, filenames in os.walk('/kaggle/input/nab/realAdExchange/realAdExchange/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Any results you write to the current directory are saved as output.

import os

for dirname, _, filenames in os.walk('/kaggle/input/nab/realAWSCloudwatch/realAWSCloudwatch/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df8 = pd.read_csv('/kaggle/input/nab/realAdExchange/realAdExchange/exchange-2_cpc_results.csv')

df1 = pd.read_csv('/kaggle/input/nab/realAWSCloudwatch/realAWSCloudwatch/ec2_cpu_utilization_825cc2.csv')

df2 = pd.read_csv('/kaggle/input/nab/realAWSCloudwatch/realAWSCloudwatch/ec2_cpu_utilization_ac20cd.csv')

df3 = pd.read_csv('/kaggle/input/nab/realAWSCloudwatch/realAWSCloudwatch/ec2_cpu_utilization_fe7f93.csv')

df4 = pd.read_csv('/kaggle/input/nab/realAWSCloudwatch/realAWSCloudwatch/ec2_cpu_utilization_77c1ca.csv')



df5 = pd.read_csv('/kaggle/input/nab/realAWSCloudwatch/realAWSCloudwatch/ec2_network_in_5abac7.csv')



df6 = pd.read_csv('/kaggle/input/nab/realAWSCloudwatch/realAWSCloudwatch/grok_asg_anomaly.csv')



df7 = pd.read_csv('/kaggle/input/nab/realAWSCloudwatch/realAWSCloudwatch/elb_request_count_8c0756.csv')

df6.head()
df4.head()
x = [dt.datetime.strptime(d,"%Y-%m-%d %H:%M:%S").date() for d in df4["timestamp"]]

y = df4["value"]



plt.plot(x,y)

plt.show()
fpath = "../input/nab/realAWSCloudwatch/realAWSCloudwatch//"

fname = "grok_asg_anomaly.csv"



fullPath = fpath + fname



def parser(x):

	return dt.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")

 

data = pd.read_csv(fullPath, header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
arimaM = ARIMA(data, order=(5,1,0))

arimaMfit = arimaM.fit(disp=0)

print(arimaMfit.summary())
# plot residual errors

errors = pd.DataFrame(arimaMfit.resid)

errors.plot()

pyplot.show()

errors.plot(kind='kde')

pyplot.show()

print(errors.describe())
X = data.values

size = int(len(X) * 0.70)

limitCount = 50

train, test = X[0:size], X[size:size+limitCount]

history = [x for x in train]
predictions = list()

for t in range(len(test)):

	model = ARIMA(history, order=(5,1,0))

	model_fit = model.fit(disp=0)

	output = model_fit.forecast()

	yhat = output[0]

	predictions.append(yhat)

	obs = test[t]

	history.append(obs)

	print('pred=%f, exp=%f' % (yhat, obs))

error = mean_squared_error(test, predictions)

print('Mean Squared Error: %.3f' % error)

# plot

pyplot.plot(test)

pyplot.plot(predictions, color='red')

pyplot.show()
from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error
dataframe = pd.read_csv(dirname+filename, usecols=[1], skipfooter=3)

dataset = dataframe.values

dataset = dataset.astype('float32')
# convert an array of values into a dataset matrix

def create_dataset(dataset, look_back=1):

	dataX, dataY = [], []

	for i in range(len(dataset)-look_back-1):

		a = dataset[i:(i+look_back), 0]

		dataX.append(a)

		dataY.append(dataset[i + look_back, 0])

	return np.array(dataX), np.array(dataY)
# fix random seed for reproducibility

np.random.seed(7)

# load the dataset

# dataframe = pd.read_csv(dirname+filename, usecols=[1], skipfooter=3)

# dataset = dataframe.values

# dataset = dataset.astype('float32')

# dataset = data.values

# dataset = data.astype('float32')



# normalize the dataset

scaler = MinMaxScaler(feature_range=(0, 1))

dataset = scaler.fit_transform(dataset)

# split into train and test sets

train_size = int(len(dataset) * 0.67)

test_size = len(dataset) - train_size

train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
# reshape into X=t and Y=t+1

look_back = 3

trainX, trainY = create_dataset(train, look_back)

testX, testY = create_dataset(test, look_back)

trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))

testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
# create and fit the LSTM network

batch_size = 1

model = Sequential()

model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True))

model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

for i in range(5):

	model.fit(trainX, trainY, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)

	model.reset_states()
# make predictions

trainPredict = model.predict(trainX, batch_size=batch_size)

model.reset_states()

testPredict = model.predict(testX, batch_size=batch_size)
# invert predictions

trainPredict = scaler.inverse_transform(trainPredict)

trainY = scaler.inverse_transform([trainY])

testPredict = scaler.inverse_transform(testPredict)

testY = scaler.inverse_transform([testY])
# calculate root mean squared error

trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))

print('Train Score: %.2f RMSE' % (trainScore))

testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))

print('Test Score: %.2f RMSE' % (testScore))
# shift train predictions for plotting

trainPredictPlot = np.empty_like(dataset)

trainPredictPlot[:, :] = np.nan

trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting

testPredictPlot = np.empty_like(dataset)

testPredictPlot[:, :] = np.nan

testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions

plt.plot(scaler.inverse_transform(dataset))

plt.plot(trainPredictPlot)

plt.plot(testPredictPlot)

plt.show()