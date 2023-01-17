from time_series import TimeSeries



# Imports for data visualization

import matplotlib.pyplot as plt

from pandas.plotting import register_matplotlib_converters

from matplotlib.dates import DateFormatter

from matplotlib import dates as mpld



register_matplotlib_converters()

import pandas as pd

data = pd.read_csv('../input/monthly-sales/monthly_sales.csv')

ts = TimeSeries(data, train_size=0.8)



print("Sales Data")

print(ts.data.describe())



print("Head and Tail of the time series")

print(ts.data.head(5).iloc[:,1])

print(ts.data.tail(5).iloc[:,1])



# Plot of raw time series data

plt.plot(ts.data.index,ts.data.sales)

plt.gcf().autofmt_xdate()

date_format = mpld.DateFormatter('%Y-%m')

plt.gca().xaxis.set_major_formatter(date_format)

plt.title("Sales Data Analysis (2013-2016)")

plt.xlabel("Time")

plt.ylabel("Sales")

plt.show()
from statsmodels.tsa.seasonal import seasonal_decompose



result_add = seasonal_decompose(ts.data.iloc[:,1],period=12,model='additive')

result_add.plot()

plt.gcf().autofmt_xdate()

date_format = mpld.DateFormatter('%y-%m')

plt.gca().xaxis.set_major_formatter(date_format)



result_mul = seasonal_decompose(ts.data.iloc[:,1],period=12,model='multiplicative')

result_mul.plot()

plt.gcf().autofmt_xdate()

date_format = mpld.DateFormatter('%y-%m')

plt.gca().xaxis.set_major_formatter(date_format)

plt.show()

from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Scaling down the data by a factor of 1000

ts.set_scale(1000)



# Training the model

model = ExponentialSmoothing(ts.train,trend='additive',seasonal='additive',seasonal_periods=12).fit(damping_slope=1)

plt.plot(ts.train.index,ts.train,label="Train")

plt.plot(ts.test.index,ts.test,label="Actual")



# Create a 5 year forecast

plt.plot(model.forecast(60),label="Forecast")



plt.legend(['Train','Actual','Forecast'])

plt.gcf().autofmt_xdate()

date_format = mpld.DateFormatter('%Y-%m')

plt.gca().xaxis.set_major_formatter(date_format)

plt.title("Sales Data Analysis (2013-2016)")

plt.xlabel("Time")

plt.ylabel("Sales (x1000)")

plt.show()
from statsmodels.tsa.holtwinters import ExponentialSmoothing



ts = TimeSeries(data, train_size=0.8)

# Additive model

model_add = ExponentialSmoothing(ts.data.iloc[:,1],trend='additive',seasonal='additive',seasonal_periods=12,damped=True).fit(damping_slope=0.98)

prediction = model_add.predict(start=ts.data.iloc[:,1].index[0],end=ts.data.iloc[:,1].index[-1])



plt.plot(ts.data.iloc[:,1].index,ts.data.iloc[:,1],label="Train")

plt.plot(ts.data.iloc[:,1].index,prediction,label="Model")

plt.plot(model_add.forecast(60))



plt.legend(['Actual','Model','Forecast'])

plt.gcf().autofmt_xdate()

date_format = mpld.DateFormatter('%Y-%m')

plt.gca().xaxis.set_major_formatter(date_format)

plt.title("Sales Data Analysis (2013-2016)")

plt.xlabel("Time")

plt.ylabel("Sales")

plt.show()





# Multiplicative model

model_mul = ExponentialSmoothing(ts.data.iloc[:,1],trend='additive',seasonal='multiplicative',seasonal_periods=12,damped=True).fit()

prediction = model_mul.predict(start=ts.data.iloc[:,1].index[0],end=ts.data.iloc[:,1].index[-1])



plt.plot(ts.data.iloc[:,1].index,ts.data.iloc[:,1],label="Train")

plt.plot(ts.data.iloc[:,1].index,prediction,label="Model")

plt.plot(model_mul.forecast(60))

plt.legend(['Actual','Model','Forecast'])

plt.gcf().autofmt_xdate()

date_format = mpld.DateFormatter('%Y-%m')

plt.gca().xaxis.set_major_formatter(date_format)

plt.title("Sales Data Analysis (2013-2016)")

plt.xlabel("Time")

plt.ylabel("Sales")

plt.show()
print(model_add.summary())

print(model_mul.summary())