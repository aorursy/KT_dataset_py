#Import libraries
import warnings
import itertools
import numpy as np
#Ignore errors 
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore") 
import pandas as pd
from pandas import Series
from sklearn.metrics import mean_squared_error
from math import sqrt
from pandas import datetime
import statsmodels.api as sm
import folium
import matplotlib.pyplot as plt
import matplotlib.dates 
#style mode 
plt.style.use('seaborn-whitegrid')
%matplotlib inline
#Our dataset is a CSV file with air quality data  registered in Madrid in 2016.Obtained from Kaggle
series = pd.read_csv('../input/csvs_per_year/csvs_per_year/madrid_2016.csv')
# Dimentionality of the dataframe
print(series.shape)
#After exploring the data we have selected a station with a relevant amount of PM10 data.
#It is the station 28079008 in Madrid .

stations = pd.read_csv('../input/stations.csv')
stations=stations[stations['id'] == 28079008]
locations  = stations[['lat', 'lon']]
#locations=locations[locations['id'] == 28079008]
locationlist = locations.values.tolist()

popup = stations[['name']]


map_osm = folium.Map(location=[40.44, -3.69],
                    # tiles='Stamen Toner',
                     zoom_start=11) 

for point in range(0, len(locationlist)):
    folium.Marker(locationlist[point], popup=popup.iloc[point,0]).add_to(map_osm)
    
display(map_osm)


#Nan values are filled with the last correct value registered
station1 = series[series['station'] == 28079008].fillna(method='ffill')
#We will select PM10 
station1 = station1[['date', 'PM10']]
#convert date to datetime format 
station1['date'] = pd.to_datetime(station1['date'])
#We sort values by date
station1=station1.sort_values(by='date')
#The new index will be the date
station1=station1.reset_index(drop=True)
station1.set_index('date',inplace=True)


# Convert PM10 values to float
myDataset = pd.DataFrame(station1['PM10'])
Dataset = myDataset['PM10'].values
Dataset = Dataset.astype('float32')
Dataset=station1['PM10']
X=Dataset
#Adjust frequency
X.index = pd.DatetimeIndex(X.index.values,freq=X.index.inferred_freq)
#We eliminate outliers( if PM10 > 150)
for (i, item) in enumerate(X):
    if item > 150:
        X[i] = np.mean(X)
#list with the dataset
history = [X for X in X]

print('dataset')
print(X)

#Plot PM10 values in 2016
X.plot(figsize=(15, 6),style='k.')
plt.xlabel('Date')
plt.ylabel('PM10')
plt.show()
#Boxplot Statistics in 2016
def parse(x):
    return pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
i=0
graph1=list()
months2=list()
from pandas import DataFrame
from pandas import TimeGrouper
from pandas import concat
for year in [2016]:
    i+=1
    year2=str(year)
    one_year = X[year2]
    groups = one_year.groupby(TimeGrouper('M'))
    months = concat([DataFrame(x[1].values) for x in groups], axis=1)
    months = DataFrame(months)
    months.columns =range(1,13)
    months2.append(months)
    graph1=months.boxplot(figsize=(15, 6))
    plt.title(year2)
    plt.show()
stats=list()
#print(months.describe())
for i in range(len(months.columns)):
    print('Month:',i+1)
    stats.append(months[i+1].describe())
    print(stats[i])


#The month with the greatest contamination is September. October has high levels of PM10 too.
#Define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0, 2)

# Generate all different combinations of seasonal p, d and q  
pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal P, D and Q triplets.We select s parameter (stationality) =12, monthly
seasonal_pdq = [(t[0], t[1], t[2], 12) for t in list(itertools.product(p, d, q))]

#Iteration through combinations of parameters.Automated selection of optimal parameters
#Score selected: AIC(Akaike Information Criterion)
#SARIMAX:Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors model
smallest=None
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(X,
                order=param,
                seasonal_order=param_seasonal,
                enforce_stationarity=False,
                enforce_invertibility=False)
            results = mod.fit(disp=0)
            print('ARIMA{}x{} - AIC:{}'.format(param, param_seasonal, results.aic))
            if smallest is None or results.aic < smallest:
                smallest=results.aic
                valor1=param
                valor2=param_seasonal
                #print(smallest,valor1,valor2)
        except:
            continue
print('Value AIC minimum: ',smallest)
print('Optimal pdq parameters: ',valor1)
print('Optimal PDQS parameteres: ',valor2)      

#Diagnostics Graphics
results.plot_diagnostics(figsize=(15, 12))
plt.show()
#residuals are normally distributed
#autocorrelation (i.e. correlogram) shows that the time series residuals have low correlation with lagged versions of itself.
#So we will assume that our model could be correct.
#Selection of optimal parameters to train the model 
#valor1=(1,1,1)
#valor2=(1,1,1,12)
mod = sm.tsa.statespace.SARIMAX(X,order=valor1,seasonal_order=valor2,enforce_stationarity=False,enforce_invertibility=False)
results = mod.fit(disp=0)
#Prediction of 2016 data
pred_ci = results.predict()
#Elimination of negative values in predictions
for (i, item) in enumerate(pred_ci):
    if item < 0:
        pred_ci[i] = 0
print('Predicted values for 2016',pred_ci)
print('RMSE 2016: %.3f'% sqrt(mean_squared_error(X, pred_ci)))
datefmt =matplotlib.dates.DateFormatter("%Y-%m-%d : %H:%M:%S")
fmt = lambda x,y : "{}, {:.5g}".format(datefmt(x), y)
plt.rcParams['figure.figsize'] = (20, 10)
plt.gca().format_coord = fmt
plt.xlabel('Date')
plt.ylabel('PM10')
plt.plot(pred_ci,'r',label='Predicted PM10')
plt.plot(X,'b',label='Expected PM10')
plt.legend(loc='best')
plt.show()


#We will repeat the same operations on 2017 year that we have done with 2016 year
series2 = pd.read_csv('../input/csvs_per_year/csvs_per_year/madrid_2017.csv')
station2 = series2[series2['station'] == 28079008].fillna(method='ffill')
station2 = station2[['date', 'PM10']]
station2['date'] = pd.to_datetime(station2['date'])
station2=station2.sort_values(by='date')
station2=station2.reset_index(drop=True)
station2.set_index('date',inplace=True)

myDataset2 = pd.DataFrame(station2['PM10'])
Dataset2 = myDataset2['PM10'].values
Dataset2 = Dataset2.astype('float32')
Dataset2=station2['PM10']
X2=Dataset2
X2.index = pd.DatetimeIndex(X2.index.values,freq=X2.index.inferred_freq)
for (i, item) in enumerate(X2):
    if item > 150:
        X2[i] = np.mean(X2)

#Lists for observations and predictions 
obs2=list()
predictions=list()
mape=list()
fechas=list()
num_dias=335

#Multistep Prediction for December.
mod3 = sm.tsa.statespace.SARIMAX(X,order=valor1,seasonal_order=valor2,enforce_stationarity=False,enforce_invertibility=False)
results3 = mod3.fit(disp=0)
pred3 = results3.get_prediction(start=pd.to_datetime('2016-12-01 01:00:00'),end='2017-01-01 00:00:00', dynamic=False)
truth=list()
predicciones3=list()
i2=0
#2016 was a leap year
for i2 in range((366-num_dias)*24):
        truth.append(X[num_dias*24:][i2])
        predicciones3.append(pred3.predicted_mean[i2])
        if predicciones3[i2] < 0:
            predicciones3[i2] = 1
        fechas.append(X.index[(num_dias*24)+i2])
        #only print first and last day
        if i2 <23:
            print('First day of predictions')
            print('>Predicted=%.3f, Expected =%3.f,Date= %s"} ' % (predicciones3[i2],truth[i2],fechas[i2]))
        if i2>719:
            print('Last day of predictions')
            print('>Predicted=%.3f, Expected =%3.f,Date= %s"} ' % (predicciones3[i2],truth[i2],fechas[i2]))
RMSE2016=sqrt(mean_squared_error(truth, predicciones3))            
print('RMSE December 2016: %.3f'% RMSE2016)


#Accuracy Calculation between calculations and predictions with an acceptance 
def myAccuracy(predictions, truth, acceptance = 0):
    total = len(predictions)
    errors = 0
    for i in range(total):
        if abs(truth[i] - predictions[i] > acceptance):
            errors += 1
    return (total-errors)/total
accuracy_array = list()
smallest2=None
greatest2=None
i3=0
# Acceptance range less than 2*RMSE
for acceptance in range(0, 2*int(RMSE2016)):
    i3=i3+1
    accuracy_array.append(myAccuracy(predicciones3, truth, acceptance) * 100)
    if smallest2 is None or accuracy_array[i3-1] < smallest2:
                smallest2=accuracy_array[i3-1]
                valor10=acceptance
    if greatest2 is None or accuracy_array[i3-1] > greatest2:
                greatest2=accuracy_array[i3-1]
                valor20=acceptance
            
accuracy_array = np.array(accuracy_array)
print('Accuracy mean,max and min',np.mean(accuracy_array),np.max(accuracy_array),np.min(accuracy_array))
print('Accuracy max with acceptance ',greatest2,valor20)
print('Accuracy min with acceptance ',smallest2,valor10)
accuracy_value = myAccuracy(predicciones3, truth, sqrt(mean_squared_error(truth, predicciones3)))
print('Accuracy for an acceptance equals RMSE: ', accuracy_value)
fig2 = plt.figure(figsize=(15,6))
plt.plot(accuracy_array)
plt.xlabel('Acceptance  Value')
plt.ylabel('Accuracy')
plt.show()
datefmt =matplotlib.dates.DateFormatter("%Y-%m-%d : %H:%M:%S")
fmt = lambda x,y : "{}, {:.5g}".format(datefmt(x), y)
plt.rcParams['figure.figsize'] = (20, 10)
plt.gca().format_coord = fmt
plt.xlabel('Date')
plt.ylabel('PM10')
plt.plot(fechas,truth,'b',label='Expected PM10')
plt.plot(fechas,predicciones3,'r',label='Predicted PM10')
plt.legend(loc='best')
plt.show()
#Onestep rolling prediction for the first week of 2017
num_dias2=7
fechas2=list()
for i in range(0, num_dias2*24):
    mod2 = sm.tsa.statespace.SARIMAX(history,order=valor1,seasonal_order=valor2,enforce_stationarity=False,enforce_invertibility=False)
    results = mod2.fit(disp=0)
    # prediction for t+1
    ypred = results.forecast()[0]
    if ypred < 0:
            ypred = 1
    predictions.append(ypred)
    # expected value
    obs2.append(X2[i])
    #mape calculation
    mape.append(abs((obs2[i]-ypred)/obs2[i]))
    history.append(obs2[i])
    fechas2.append(X2.index[i])
    print('>Predicted=%.3f, Expected =%3.f,Date= %s"} ' % (ypred, obs2[i],fechas2[i]))
    
rmse = sqrt(mean_squared_error(obs2, predictions))
mape_a=np.mean(mape)
print('MAPE error',mape_a)
print('Onestep Prediction RMSE 2017: %.3f' % rmse)

datefmt =matplotlib.dates.DateFormatter("%Y-%m-%d : %H:%M:%S")
fmt = lambda x,y : "{}, {:.5g}".format(datefmt(x), y)
plt.rcParams['figure.figsize'] = (20, 10)
plt.gca().format_coord = fmt
plt.xlabel('Date')
plt.ylabel('PM10')
plt.plot(fechas2,obs2,'b',label='Expected PM10')
plt.plot(fechas2,predictions,'r',label='Predicted PM10')
plt.legend(loc='best')
plt.show()
