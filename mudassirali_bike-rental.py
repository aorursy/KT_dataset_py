import pandas as pd

import numpy as np
df = pd.read_csv('../input/london-bike-sharing-dataset/london_merged.csv')

df.head()
df.timestamp.value_counts()
for i in range(0,len(df)):

    df.timestamp[i] = df.timestamp[i][0:7]



# I only want the year and the month of the data to do my time series
df.head()
df1 = df[['timestamp' , 'cnt']]



# I also only want the timestamp and the count for my time series
df1
a = []

for i in range(1,10):

  for n in range(15,17):

    a.append({"year/month":f"20{n}-0{i}","count":df1.loc[df1['timestamp'] == f'20{n}-0{i}', 'cnt'].sum()})

for i in range(10,13):

  for n in range(15,17):

    a.append({"year/month":f"20{n}-{i}","count":df1.loc[df1['timestamp'] == f'20{n}-{i}', 'cnt'].sum()})



# this forloop of code will sum all the bikes for the given year + month
bikes = pd.DataFrame.from_dict(a)

bikes['year/month'] = pd.to_datetime(bikes['year/month'])

bikes = bikes.sort_values(by=['year/month'])

bikes.index = sorted(bikes['year/month'])

bikes.drop(['year/month'],axis = 1, inplace = True)



# formatting the aforementioned code into a datatime dataframe
bikes
import matplotlib.pyplot as plt

from matplotlib.dates import DateFormatter

import matplotlib.dates as mdates



fig, ax = plt.subplots(figsize=(20, 10))





ax.plot(np.array(bikes.index.values),

        np.array(bikes), '-o',

        color = 'purple')





ax.set(xlabel="Date",

       ylabel="Count",

       title="Number of bikes")



ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))

ax.xaxis.set_major_locator(mdates.MonthLocator(interval = 3))

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))



plt.show()



#plotting the created dataframe to see any trends and seasonality in the dataframe

# as we can see in the summer time there is a peak in the number of bikes rented

# and as the weather gets colder the number decreases
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

from pylab import rcParams

result = seasonal_decompose(bikes['count'] , model = 'multiplicative')



rcParams['figure.figsize'] = 12,5



result.plot();
train_data = bikes.iloc[:13]

test_data = bikes.iloc[12:]
from statsmodels.tsa.holtwinters import ExponentialSmoothing



fitted_model = ExponentialSmoothing(train_data['count'],

                                   trend = 'mul' , 

                                   seasonal = 'mul' , 

                                   seasonal_periods = 12).fit()
test_predictions = fitted_model.forecast(12)
pd.options.display.float_format = '{:.2f}'.format

test_predictions
train_data['count'].plot(legend = True , label = 'Train');

test_data['count'].plot(legend = True , label = 'Test');

test_predictions.plot(legend = True , label = 'PREDICTION');



plt.show()



# We can see that my prediction is a bit off from the actual testing data set

# Nevertheless it is still a decent prediction for one year
from sklearn.metrics import mean_squared_error , mean_absolute_error



print(test_data.describe())

print('')

print('mean absolute error:', np.round(mean_absolute_error(test_data , test_predictions),2))

print('root mean squared error:', np.round(mean_squared_error(test_data, test_predictions)**(1/2),2))



# The root mean squared error in comparison to the STD is a pretty good value
final_model = ExponentialSmoothing(bikes['count'] , trend = 'mul'

                                  , seasonal = 'mul' , seasonal_periods = 12).fit()
forecast_predictions = final_model.forecast(12)
bikes['count'].plot(legend = True, label = 'Counted Bikes');

forecast_predictions.plot(legend = True , label = 'Yearly Prediction');

plt.show()



# We can see that our prediction into the future somewhat replicates the trend

# of the previous two years
FP = pd.Series(forecast_predictions)

FP