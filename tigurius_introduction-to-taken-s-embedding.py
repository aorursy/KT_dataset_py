# load some standard libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import math #math fun
import matplotlib.pyplot as plt #plotting
from mpl_toolkits.mplot3d import Axes3D #3d plots
from sklearn.neighbors import NearestNeighbors 

#load weather data that will be used in the script
cityTable     = pd.read_csv('../input/city_attributes.csv')
temperatureDF = pd.read_csv('../input/temperature.csv', index_col=0)
temperatureDF.index = pd.to_datetime(temperatureDF.index)
def takensEmbedding (data, delay, dimension):
    "This function returns the Takens embedding of data with delay into dimension, delay*dimension must be < len(data)"
    if delay*dimension > len(data):
        raise NameError('Delay times dimension exceed length of data!')    
    embeddedData = np.array([data[0:len(data)-delay*dimension]])
    for i in range(1, dimension):
        embeddedData = np.append(embeddedData, [data[i*delay:len(data) - delay*(dimension - i)]], axis=0)
    return embeddedData;

#Apply Takens embedding to daily weather data of Montreal
t = pd.date_range(pd.to_datetime('22/6/2015',dayfirst=True),pd.to_datetime('31/8/2015',dayfirst=True),freq='H')
weatherDataMontreal = temperatureDF.loc[t,'Montreal'];
origSignal = weatherDataMontreal;
#we are interested in the daily dynamics, so we have to highpass-filter the signal 
#to remove the monthly and yearly dynamics
#apply rolling mean over one day and plot the signal (low pass filter) 
windowSize = 24
lowPassFilteredSignal = weatherDataMontreal.rolling(windowSize, center=True).mean()
# subtract the low pass filtered singal from the original to get high pass filtered signal
weatherDataMontreal = weatherDataMontreal - lowPassFilteredSignal
#remove all NaNs
weatherDataMontreal = weatherDataMontreal.dropna()
#embedd into two dimensions
embeddedWeather = takensEmbedding(weatherDataMontreal,5,2);
#plot the time-series and the embedded one 
fig, ax = plt.subplots(nrows=3,ncols=1,figsize=(15,14));
ax[0].plot(weatherDataMontreal);
ax[1].plot(embeddedWeather[0,:],embeddedWeather[1,:]);
ax[2].axis('off')
#embed into three dimensions
embeddedWeather3 = takensEmbedding(weatherDataMontreal, 6,3);
#plot the 3D embedding
ax = fig.add_subplot(3, 1, 3, projection='3d')
ax.plot(embeddedWeather3[0,:],embeddedWeather3[1,:],embeddedWeather3[2,:]);
def mutualInformation(data, delay, nBins):
    "This function calculates the mutual information given the delay"
    I = 0;
    xmax = max(data);
    xmin = min(data);
    delayData = data[delay:len(data)];
    shortData = data[0:len(data)-delay];
    sizeBin = abs(xmax - xmin) / nBins;
    #the use of dictionaries makes the process a bit faster
    probInBin = {};
    conditionBin = {};
    conditionDelayBin = {};
    for h in range(0,nBins):
        if h not in probInBin:
            conditionBin.update({h : (shortData >= (xmin + h*sizeBin)) & (shortData < (xmin + (h+1)*sizeBin))})
            probInBin.update({h : len(shortData[conditionBin[h]]) / len(shortData)});
        for k in range(0,nBins):
            if k not in probInBin:
                conditionBin.update({k : (shortData >= (xmin + k*sizeBin)) & (shortData < (xmin + (k+1)*sizeBin))});
                probInBin.update({k : len(shortData[conditionBin[k]]) / len(shortData)});
            if k not in conditionDelayBin:
                conditionDelayBin.update({k : (delayData >= (xmin + k*sizeBin)) & (delayData < (xmin + (k+1)*sizeBin))});
            Phk = len(shortData[conditionBin[h] & conditionDelayBin[k]]) / len(shortData);
            if Phk != 0 and probInBin[h] != 0 and probInBin[k] != 0:
                I -= Phk * math.log( Phk / (probInBin[h] * probInBin[k]));
    return I;
datDelayInformation = []
for i in range(1,21):
    datDelayInformation = np.append(datDelayInformation,[mutualInformation(weatherDataMontreal,i,16)])
plt.plot(range(1,21),datDelayInformation);
plt.xlabel('delay');
plt.ylabel('mutual information');
#embedd into two dimensions
embeddedWeather = takensEmbedding(weatherDataMontreal,5,2);
#plot the time-series and the embedded one 
fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(15,14));
ax[0].plot(embeddedWeather[0,:],embeddedWeather[1,:]);
ax[0].set_xlabel('$x_i$');
ax[0].set_ylabel('$x_{i+5}$');
#now with delay=1
embeddedWeather = takensEmbedding(weatherDataMontreal,1,2);
ax[1].plot(embeddedWeather[0,:],embeddedWeather[1,:]);
ax[1].set_xlabel('$x_i$');
ax[1].set_ylabel('$x_{i+1}$');
def false_nearest_neighours(data,delay,embeddingDimension):
    "Calculates the number of false nearest neighbours of embedding dimension"    
    embeddedData = takensEmbedding(data,delay,embeddingDimension);
    #the first nearest neighbour is the data point itself, so we choose the second one
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(embeddedData.transpose())
    distances, indices = nbrs.kneighbors(embeddedData.transpose())
    #two data points are nearest neighbours if their distance is smaller than the standard deviation
    epsilon = np.std(distances.flatten())
    nFalseNN = 0
    for i in range(0, len(data)-delay*(embeddingDimension+1)):
        if (0 < distances[i,1]) and (distances[i,1] < epsilon) and ( (abs(data[i+embeddingDimension*delay] - data[indices[i,1]+embeddingDimension*delay]) / distances[i,1]) > 10):
            nFalseNN += 1;
    return nFalseNN
nFNN = []
for i in range(1,7):
    nFNN.append(false_nearest_neighours(weatherDataMontreal,1,i) / len(weatherDataMontreal))
plt.plot(range(1,7),nFNN);
plt.xlabel('embedding dimension');
plt.ylabel('Fraction of fNN');
embeddedWeatherFinal =  takensEmbedding(weatherDataMontreal,1,4)
plt.figure(figsize=(15,14))
for i in range(0,4):
    plt.plot(embeddedWeatherFinal[i,:])
fig = plt.figure(figsize=(15,14))
ax = fig.add_subplot(2, 2, 1, projection='3d')
ax.plot(embeddedWeatherFinal[0,:],embeddedWeatherFinal[1,:],embeddedWeatherFinal[2,:]);
ax = fig.add_subplot(2, 2, 2, projection='3d')
ax.plot(embeddedWeatherFinal[0,:],embeddedWeatherFinal[1,:],embeddedWeatherFinal[3,:]);
ax = fig.add_subplot(2, 2, 3, projection='3d')
ax.plot(embeddedWeatherFinal[1,:],embeddedWeatherFinal[2,:],embeddedWeatherFinal[3,:]);
ax = fig.add_subplot(2, 2, 4, projection='3d')
ax.plot(embeddedWeatherFinal[0,:],embeddedWeatherFinal[2,:],embeddedWeatherFinal[3,:]);