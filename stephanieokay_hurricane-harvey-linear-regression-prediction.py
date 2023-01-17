import numpy as np 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from bq_helper import BigQueryHelper

#BigQuery Python client library to query tables in this dataset 
bq_assistant = BigQueryHelper("bigquery-public-data", "noaa_spc")
#query the hail table for the dates when hurricane harvey occurrred
#and where hurricane harvey occurred
#https://www.fema.gov/disaster/4332
#for list of affected counties
hailQuery = """SELECT timestamp, state, county, size, comments, latitude, longitude
           FROM `bigquery-public-data.noaa_spc.hail_reports`
           WHERE state = 'TX' AND 
           timestamp BETWEEN '2017-08-17' AND '2017-09-03' AND 
           county in ('ARANSAS', 'AUSTIN', 'BASTROP', 'BEE', 'BRAZORIA', 'CALDWELL', 'CALHOUN', 'CHAMBERS', 'COLORADO', 'DEWITT', 'FAYETTE', 'FORT BEND', 'GALVESTON', 'GOLIAD', 'GONZALES', 'GRIMES', 'HARDIN', 'HARRIS', 'JACKSON', 'JASPER', 'JEFFERSON', 'KARNES', 'KLEBERG', 'LAVACA', 'LEE', 'LIBERTY', 'MATAGORDA', 'MONTGOMERY', 'NEWTON', 'NUECES', 'ORANGE', 'POLK', 'REFUGIO', 'SABINE', 'SAN JACINTO', 'SAN PATRICION', 'TYLER', 'VICTORIA', 'WALKER', 'WALLER', 'WHARTON')
           ORDER BY size desc
        """
bq_assistant.estimate_query_size(hailQuery)
#perform query, show gives the number of rows in the query
hailDataFrame = bq_assistant.query_to_pandas_safe(hailQuery)
hailDataFrame.shape[0]
#give a glimpse of the rows in the dataframe
hailDataFrame.head()
#query the tornado table for the dates when hurricane harvey occurrred
#and where hurricane harvey occurred
tornadoQuery = """SELECT timestamp, state, county, f_scale, comments, latitude, longitude
           FROM `bigquery-public-data.noaa_spc.tornado_reports`
           WHERE state = 'TX' AND 
           timestamp BETWEEN '2017-08-17' AND '2017-09-03' AND
           county in ('ARANSAS', 'AUSTIN', 'BASTROP', 'BEE', 'BRAZORIA', 'CALDWELL', 'CALHOUN', 'CHAMBERS', 'COLORADO', 'DEWITT', 'FAYETTE', 'FORT BEND', 'GALVESTON', 'GOLIAD', 'GONZALES', 'GRIMES', 'HARDIN', 'HARRIS', 'JACKSON', 'JASPER', 'JEFFERSON', 'KARNES', 'KLEBERG', 'LAVACA', 'LEE', 'LIBERTY', 'MATAGORDA', 'MONTGOMERY', 'NEWTON', 'NUECES', 'ORANGE', 'POLK', 'REFUGIO', 'SABINE', 'SAN JACINTO', 'SAN PATRICION', 'TYLER', 'VICTORIA', 'WALKER', 'WALLER', 'WHARTON')
           ORDER BY timestamp asc
        """
bq_assistant.estimate_query_size(tornadoQuery)
#perform query, show gives the number of rows in the query
tornadoDataFrame = bq_assistant.query_to_pandas_safe(tornadoQuery)
tornadoDataFrame.shape[0]
#give a glimpse of the rows in the dataframe
tornadoDataFrame.head()
#query the wind table for the dates when hurricane harvey occurrred
#and where hurricane harvey occurred
windQuery = """SELECT timestamp, state, county, speed, comments, latitude, longitude
           FROM `bigquery-public-data.noaa_spc.wind_reports`
           WHERE state = 'TX' AND 
           timestamp BETWEEN '2017-08-17' AND '2017-09-03' AND
           county in ('ARANSAS', 'AUSTIN', 'BASTROP', 'BEE', 'BRAZORIA', 'CALDWELL', 'CALHOUN', 'CHAMBERS', 'COLORADO', 'DEWITT', 'FAYETTE', 'FORT BEND', 'GALVESTON', 'GOLIAD', 'GONZALES', 'GRIMES', 'HARDIN', 'HARRIS', 'JACKSON', 'JASPER', 'JEFFERSON', 'KARNES', 'KLEBERG', 'LAVACA', 'LEE', 'LIBERTY', 'MATAGORDA', 'MONTGOMERY', 'NEWTON', 'NUECES', 'ORANGE', 'POLK', 'REFUGIO', 'SABINE', 'SAN JACINTO', 'SAN PATRICION', 'TYLER', 'VICTORIA', 'WALKER', 'WALLER', 'WHARTON')
           ORDER BY timestamp asc
        """
bq_assistant.estimate_query_size(windQuery)
#perform query, show gives the number of rows in the query
windDataFrame = bq_assistant.query_to_pandas_safe(windQuery)
windDataFrame.shape[0]
#give a glimpse of the rows in the dataframe
windDataFrame.head(8)
#plot latitude and longitude for tornado data frame
tornadoDataFrame.plot(x='latitude', y='longitude', style='.', color='DarkGreen');
#creating vectors X and Y from two different columns in the tornado data frame
X = tornadoDataFrame['latitude'].values
Y = tornadoDataFrame['longitude'].values
#how many values are in vector X
numberOfValues = len(X)
#finding the weight of X and the bias
#using the numpy library to calculate the mean of X and Y vectors
meanX = np.mean(X)
meanY = np.mean(Y)
sum1 = 0
sum2 = 0
#loop through all the values in vector X
#in loop calculate the sums in order to calculate the weight value
for i in range(numberOfValues):
    sum1 += (X[i] - meanX)*(Y[i] - meanY)
    sum2 += (X[i] - meanX)*(X[i] - meanX)
#calculate the weight value
weightValue = sum1 / sum2
#calculate bias value now that you know the weight value
biasValue = meanY - (weightValue * meanX)
#print the calculated values
print(weightValue, biasValue)
print(X)
print(Y)
#calculating minimum and maximum values from the X vector
minX = np.min(X)
maxX = np.max(X)
#this function returns an array of evenly spaced numbers over the given interval
#in this case minX, maxX
xValues = np.linspace(minX, maxX)
#returns an array of y values from the bias and weight values
yValues = biasValue + weightValue * xValues
#plotting the regression line
plt.plot(xValues, yValues, color='#e59400', label='Regression Line')
#plotting the extracted X,Y data points
plt.scatter(X, Y, c='green', label='Scatter Plot')
# giving x and y plot values
plt.title('Linear Regression for Tornado Data Frame')
plt.xlabel('latitude')
plt.ylabel('longitude')
#creating a key for the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#showing the plot
plt.show()

# calculating root mean square error
rootMeanSquareError = 0 
for i in range(numberOfValues):
    #calculating predicted y value for each x value in the X vector
    predictedY = biasValue + weightValue * X[i]
    #getting the sum of the error, subtracting y predicted from y value
    #then getting the square of it
    rootMeanSquareError += (Y[i] - predictedY) * (Y[i] - predictedY)
#get the square root of the summation and divide it over the numberofValues
rootMeanSquareError = np.sqrt(rootMeanSquareError/numberOfValues)
print(rootMeanSquareError)
#plot latitude and longitude for wind data frame
windDataFrame.plot(x='latitude', y='longitude', style='.', color='Blue');
#creating vectors X and Y from two different columns in the wind data frame
X = windDataFrame['latitude'].values
Y = windDataFrame['longitude'].values
#how many values are in vector X
numberOfValues = len(X)
#finding the weight of X and the bias
#using the numpy library to calculate the mean of X and Y vectors
meanX = np.mean(X)
meanY = np.mean(Y)
sum1 = 0
sum2 = 0
#loop through all the values in vector X
#in loop calculate the sums in order to calculate the weight value
for i in range(numberOfValues):
    sum1 += (X[i] - meanX)*(Y[i] - meanY)
    sum2 += (X[i] - meanX)*(X[i] - meanX)
#calculate the weight value
weightValue = sum1 / sum2
#calculate bias value now that you know the weight value
biasValue = meanY - (weightValue * meanX)
#print the calculated values
print(weightValue, biasValue)
print(X)
print(Y)
#calculating minimum and maximum values from the X vector
minX = np.min(X)
maxX = np.max(X)
#this function returns an array of evenly spaced numbers over the given interval
#in this case minX, maxX
xValues = np.linspace(minX, maxX)
#returns an array of y values from the bias and weight values
yValues = biasValue + weightValue * xValues
#plotting the regression line
plt.plot(xValues, yValues, color='#e59400', label='Regression Line')
#plotting the extracted X,Y data points
plt.scatter(X, Y, c='green', label='Scatter Plot')
# giving x and y plot values
plt.title('Linear Regression for Wind Data Frame')
plt.xlabel('latitude')
plt.ylabel('longitude')
#creating a key for the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#showing the plot
plt.show()

#creating vectors X and Y from two different columns in both dataframes
X = tornadoDataFrame['latitude'].values
addXWindDataFrame = windDataFrame['latitude'].values
totalX = np.concatenate((X, addXWindDataFrame), axis=0)
Y = tornadoDataFrame['longitude'].values
addYWindDataFrame = windDataFrame['longitude'].values
totalY = np.concatenate((Y, addYWindDataFrame), axis=0)
#how many values are in vector X
numberOfValues = len(totalX)
#finding the weight of X and the bias
#using the numpy library to calculate the mean of X and Y vectors
meanX = np.mean(totalX)
meanY = np.mean(totalY)
sum1 = 0
sum2 = 0
#loop through all the values in vector X
#in loop calculate the sums in order to calculate the weight value
for i in range(numberOfValues):
    sum1 += (totalX[i] - meanX)*(totalY[i] - meanY)
    sum2 += (totalX[i] - meanX)*(totalX[i] - meanX)
#calculate the weight value
weightValue = sum1 / sum2
#calculate bias value now that you know the weight value
biasValue = meanY - (weightValue * meanX)
#print the calculated values
print(weightValue, biasValue)
print(totalX)
print(totalY)
#calculating minimum and maximum values from the X vector
minX = np.min(totalX)
maxX = np.max(totalX)
#this function returns an array of evenly spaced numbers over the given interval
#in this case minX, maxX
xValues = np.linspace(minX, maxX)
#returns an array of y values from the bias and weight values
yValues = biasValue + weightValue * xValues

#plot both the tornado and the wind linear regression
plt.plot(xValues, yValues, color='#e59400', label='Regression Line')
#plt.scatter(totalX, totalY, c='blue', label='combination of tornado and wind data points')
plt.plot(tornadoDataFrame['latitude'], tornadoDataFrame['longitude'], 'go', label='tornado')
plt.plot(windDataFrame['latitude'], windDataFrame['longitude'], 'bo', label='wind')
plt.xlabel('latitude')
plt.ylabel('longitude')
plt.title('Linear Regression Prediction Through Latitude and Longitude for both Tornado and Wind')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()
# calculating root mean square error
rootMeanSquareError = 0 
for i in range(numberOfValues):
    #calculating predicted y value for each x value in the X vector
    predictedY = biasValue + weightValue * totalX[i]
    #getting the sum of the error, subtracting y predicted from y value
    #then getting the square of it
    rootMeanSquareError += (totalY[i] - predictedY) * (totalY[i] - predictedY)
rootMeanSquareError = np.sqrt(rootMeanSquareError/numberOfValues)
print(rootMeanSquareError)