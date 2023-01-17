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
#Importing Libraries

import numpy as np

import pandas as pd

import glob

import warnings

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

import math as mat

from matplotlib import pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error

from keras.models import Sequential

from keras.layers import Dense, Activation,Dropout

from keras.layers import LSTM

import math
#Loading Data

path='/kaggle/input/qq16p1Data/'

all_files = glob.glob(path+"/*.csv")

list_of_files = []

for filename in all_files:

    df = pd.read_csv(filename,index_col=None, header=0)

    list_of_files.append(df)

df = pd.concat(list_of_files,axis = 0, ignore_index=True)

df
#Dealing with nan values,infinite values,.....

df.replace([np.inf, -np.inf], np.nan, inplace=True)

df.isna().sum().tolist()

df.fillna(0, inplace=True)
#For clusttering

#Firstly taking Revenue as dep variable for applying PCA(same as for Opearing Income in the nest model)



y = df.iloc[:,[260]].values



columns = df.columns.tolist()



year = []

month = []

date = []



for i in df['time']:

    year.append(int(i.split("-")[0]))

    month.append(int(i.split("-")[1]))

    date.append(int(i.split("-")[2]))



year_values = pd.Series(year)

df.insert(loc=310, column='year', value=year_values)



month_values = pd.Series(month)

df.insert(loc=310, column='month', value=month_values)



date_values = pd.Series(date)

df.insert(loc=311, column='date', value=date_values)



columns = df.columns.tolist()



df.drop(df.columns[[192,260,309]],axis=1,inplace=True)            #Removing time,revenue and Operating Income columns (time is added as y/m/d)



X = df.iloc[:,:].values

y
X
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
#FEATURE EXTRACTION USING PCA
#Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
# Applying  PCA 

from sklearn.decomposition import PCA

pca = PCA().fit(X_train)
#Plotting the Cumulative Summation of the Explained Variance(for finding the n_components  i.e feature extraction)

plt.figure()

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('Number of Components')

plt.ylabel('Variance (%)') #for each component

plt.title('Companies Dataset Explained Variance')

plt.show()
#Fitting PCA to the data

pca = PCA(n_components = 100)

X_train = pca.fit_transform(X_train)

X_test = pca.transform(X_test)
#CLUSTTERING
# Using the elbow method to find the optimal number of clusters

from sklearn.cluster import KMeans

wcss = []      #within cluster sum of squares

for i in range(1, 11):

    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)

    kmeans.fit(X_train)

    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()
# Fitting K-Means to the dataset

kmeans = KMeans(n_clusters = 8, init = 'k-means++', random_state = 42)

y_kmeans = kmeans.fit_predict(X_train)

y_kmeans_test = kmeans.predict(X_test)
# Visualising the training clusters

plt.scatter(X_train[y_kmeans == 0, 0], X_train[y_kmeans == 0, 1], s = 20, c = 'red', label = 'Cluster 1')

plt.scatter(X_train[y_kmeans == 1, 0], X_train[y_kmeans == 1, 1], s = 20, c = 'blue', label = 'Cluster 2')

plt.scatter(X_train[y_kmeans == 2, 0], X_train[y_kmeans == 2, 1], s = 20, c = 'green', label = 'Cluster 3')

plt.scatter(X_train[y_kmeans == 3, 0], X_train[y_kmeans == 3, 1], s = 20, c = 'cyan', label = 'Cluster 4')

plt.scatter(X_train[y_kmeans == 4, 0], X_train[y_kmeans == 4, 1], s = 20, c = 'magenta', label = 'Cluster 5')

plt.scatter(X_train[y_kmeans == 5, 0], X_train[y_kmeans == 5, 1], s = 20, c = 'black', label = 'Cluster 6')

plt.scatter(X_train[y_kmeans == 6, 0], X_train[y_kmeans == 6, 1], s = 20, c = 'blueviolet', label = 'Cluster 7')

plt.scatter(X_train[y_kmeans == 7, 0], X_train[y_kmeans == 7, 1], s = 20, c = 'coral', label = 'Cluster 8')



plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 100, c = 'yellow', label = 'Centroids')

plt.title('Clusters of companies')

plt.xlabel('Pca_Features')

plt.ylabel('Revenue')

plt.legend()

plt.show()
#Visualising the test clusters

plt.scatter(X_test[y_kmeans_test == 0, 0], X_test[y_kmeans_test == 0, 1], s = 20, c = 'red', label = 'Cluster 1')

plt.scatter(X_test[y_kmeans_test == 1, 0], X_test[y_kmeans_test == 1, 1], s = 20, c = 'blue', label = 'Cluster 2')

plt.scatter(X_test[y_kmeans_test == 2, 0], X_test[y_kmeans_test == 2, 1], s = 20, c = 'green', label = 'Cluster 3')

plt.scatter(X_test[y_kmeans_test == 3, 0], X_test[y_kmeans_test == 3, 1], s = 20, c = 'cyan', label = 'Cluster 4')

plt.scatter(X_test[y_kmeans_test == 4, 0], X_test[y_kmeans_test == 4, 1], s = 20, c = 'magenta', label = 'Cluster 5')

plt.scatter(X_test[y_kmeans_test == 5, 0], X_test[y_kmeans_test == 5, 1], s = 20, c = 'black', label = 'Cluster 6')

plt.scatter(X_test[y_kmeans_test == 6, 0], X_test[y_kmeans_test == 6, 1], s = 20, c = 'blueviolet', label = 'Cluster 7')

plt.scatter(X_test[y_kmeans_test == 7, 0], X_test[y_kmeans_test == 7, 1], s = 20, c = 'coral', label = 'Cluster 8')





plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 100, c = 'yellow', label = 'Centroids')

plt.title('Clusters of companies')

plt.xlabel('Pca_Features')

plt.ylabel('Revenue')

plt.legend()

plt.show()
#1st model for Revenue

# Fitting LSTM to the data

path='/kaggle/input/qq16p1Data/'



all_files = glob.glob(path+"/*.csv")



list_of_files = []



for filename in all_files:

    df = pd.read_csv(filename,index_col=None, header=0)

    list_of_files.append(df)



df = pd.concat(list_of_files,axis = 0, ignore_index=True)

# take revenue price column

all_y = df["Revenue(Y)"].values

dataset=all_y.reshape(-1, 1)
# normalize the dataset

scaler = MinMaxScaler(feature_range=(0, 1))

dataset = scaler.fit_transform(dataset)
# split into train and test sets, 20% test data, 70% training data

train_size = int(len(dataset) * 0.70)

test_size = len(dataset) - train_size

train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
# reshape into X=t and Y=t+1, timestep 240

look_back = 240

# convert an array of values into a dataset matrix

def create_dataset(dataset, look_back=1):

    dataX, dataY = [], []

    for i in range(len(dataset)-look_back-1):

        a = dataset[i:(i+look_back), 0]

        dataX.append(a)

        dataY.append(dataset[i + look_back, 0])

    return np.array(dataX), np.array(dataY)



trainX, trainY = create_dataset(train, look_back)

testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# create and fit the LSTM network, optimizer=adam, 25 neurons, dropout 0.1

model = Sequential()

model.add(LSTM(25, input_shape=(1, look_back)))

model.add(Dropout(0.1))

model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.fit(trainX, trainY, epochs=400, batch_size=240, verbose=1)   #select epochs appropriately so that we can overcome overfitting or underfitting

# make predictions

trainPredict = model.predict(trainX)

testPredict = model.predict(testX)
# invert predictions

trainPredict = scaler.inverse_transform(trainPredict)

trainY = scaler.inverse_transform([trainY])

testPredict = scaler.inverse_transform(testPredict)

testY = scaler.inverse_transform([testY])
# calculate root mean squared error

import math

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

print('testPrices:')

testPrices=scaler.inverse_transform(dataset[test_size+look_back:])
print('testPredictions:')

print(testPredict)
#2nd model for Operating income

#1st model for Revenue

# Fitting LSTM to the data

path='/kaggle/input/qq16p1Data/'



all_files = glob.glob(path+"/*.csv")



list_of_files = []



for filename in all_files:

    df = pd.read_csv(filename,index_col=None, header=0)

    list_of_files.append(df)



df = pd.concat(list_of_files,axis = 0, ignore_index=True)

df["Operating Income (Loss)"]
# take Operating Income price column

all_y = df["Operating Income (Loss)"].values

dataset=all_y.reshape(-1, 1)
# normalize the dataset

scaler = MinMaxScaler(feature_range=(0, 1))

dataset = scaler.fit_transform(dataset)
# split into train and test sets, 20% test data, 70% training data

train_size = int(len(dataset) * 0.70)

test_size = len(dataset) - train_size

train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
# reshape into X=t and Y=t+1, timestep 240

look_back = 240
# convert an array of values into a dataset matrix

def create_dataset(dataset, look_back=1):

    dataX, dataY = [], []

    for i in range(len(dataset)-look_back-1):

        a = dataset[i:(i+look_back), 0]

        dataX.append(a)

        dataY.append(dataset[i + look_back, 0])

    return np.array(dataX), np.array(dataY)



trainX, trainY = create_dataset(train, look_back)

testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# create and fit the LSTM network, optimizer=adam, 25 neurons, dropout 0.1

model = Sequential()

model.add(LSTM(25, input_shape=(1, look_back)))

model.add(Dropout(0.1))

model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.fit(trainX, trainY, epochs=400, batch_size=240, verbose=1)   #select epochs appropriately so that we can overcome overfitting or underfitting

# make predictions

trainPredict = model.predict(trainX)

testPredict = model.predict(testX)
# invert predictions

trainPredict = scaler.inverse_transform(trainPredict)

trainY = scaler.inverse_transform([trainY])

testPredict = scaler.inverse_transform(testPredict)

testY = scaler.inverse_transform([testY])
# calculate root mean squared error

import math

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

print('testPrices:')

testPrices=scaler.inverse_transform(dataset[test_size+look_back:])
print('testPredictions:')

print(testPredict)