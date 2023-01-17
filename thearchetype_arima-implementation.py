import  numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from sklearn.metrics import mean_squared_error

from statsmodels.tsa.arima_model import ARIMA





input_file = pd.read_csv("../input/Online Retail.csv", encoding='ISO-8859-1')



input_file.drop('Description', axis=1, inplace=True)





#print(input_file)

#removing all the data with missing values

input_file.dropna(axis=0, how='any', inplace=True)



#removing disoriented data due to inappropriate csv format (unwanted commas in description)

input_file = input_file[input_file.StockCode.str.isalpha() == False]

input_file = input_file[input_file.StockCode.str.isspace() == False]

"""

Dimensions for the data are:

1.) InvoiceNo (Same invoice no. means same transaction time)

2.) StockCode (primary key kind of a thing)

3.) Quantity (Needs to be summed up for the analysis)

4.) InvoiceDate (primary attribute for time-series forecasting)

5.) UnitPrice (needs to be summed up)

6.) Country (geographic data analysis)

"""



#converting to datetime format from string

input_file['InvoiceDate'] = pd.to_datetime(input_file['InvoiceDate'])#.dt.date

#print input_file

"""

On this point, I need to aggregate all the sales of a specif product.

I'll have lesser number of rows. The new table I want looks like:

--------------------------------------------------------------

| StockCode | SumQuantity | SalesDate | TotalPrice | Country |

--------------------------------------------------------------

This means, for each day, we need to sum up quantity sold, and

total sales for each product.



Update: I kinda had smth like that, not with exact column names tho

"""

#converting to a 2D matrix

#data = input_file.as_matrix()



sales = input_file.set_index('InvoiceDate').groupby('StockCode')['Quantity', 'UnitPrice'].resample('d').sum()







# sales = input_file.groupby('StockCode')['Quantity', 'UnitPrice', 'InvoiceDate'].sum().reset_index()

sales['TotalSales'] = sales.Quantity*sales.UnitPrice



sales = sales.dropna().reset_index().reindex(columns=['InvoiceDate', 'StockCode', 'TotalSales'])



series = sales[['InvoiceDate', 'TotalSales']].groupby('InvoiceDate')['TotalSales'].sum()

#print(series)

# series.plot()

# plt.show("All sales")





#ARIMA STUFF!!! (Test code)

#fitting model

# model = ARIMA(series, order=(5, 1, 0))

# model_fit = model.fit(disp=0)

# print model_fit.summary()

#

# # plot residual errors

# residuals = pd.DataFrame(model_fit.resid)

# residuals.plot()

# plt.show()

# residuals.plot(kind='kde')

# plt.show()

# print(residuals.describe())



#ARIMA STUFF!!! (The real deal)



X = series.values

size = int(len(X) * 0.60)

train, test = X[0:size], X[size:len(X)]

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

	print('predicted=%f, expected=%f' % (yhat, obs))

error = mean_squared_error(test, predictions)

print('Test MSE: %.3f' % error)

# plot

plt.ylabel("Sales")

plt.xlabel("Timeline")

plt.plot(test)

plt.plot(predictions, color='red')

plt.show()









# sales = sales.head(100)

# threeD = plt.figure().gca(projection='3d')

# threeD.scatter(sales.index, sales['UnitPrice'])

# threeD.set_xlabel('Quantity')

# threeD.set_ylabel('UnitPrice')

# threeD.set_zlabel('InvoiceDate')

# plt.show()