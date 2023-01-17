import pandas as pd 

import numpy as np

from matplotlib import pyplot

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
sd_test=pd.read_csv("../input/sale_data_train.csv")

sd_test.head()
sd_test.dtypes
import datetime as dt

sd_test['InvoiceDate'] = pd.to_datetime(sd_test['InvoiceDate'])

sd_test['InvoiceDate']=sd_test['InvoiceDate'].map(dt.datetime.toordinal)

sd_test.head(2)
sd_test.dtypes
sd_test.info()
x=sd_test['InvoiceDate'] #independent variable 

y=sd_test['TotalSales'] #dependent variable 

x_matrix=x.values#without reshaping you will get error

x_matrix.shape

x_matrix=x.values.reshape(-1,1) #Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.
reg=LinearRegression()

reg.fit(x_matrix,y)

reg.score(x_matrix,y)
sd_test.plot()

plt.show()

plt.scatter(x_matrix, y, color = 'red')
import numpy as np

test_X = np.array(719163).reshape(-1, 1) #test our model with a sample data(date must be converted into ordinal form because our model recognize that form)

reg.predict(test_X)
sd_predict=pd.read_csv("../input/sample_submission.csv")

sd_predict['InvoiceDate'] = pd.to_datetime(sd_predict['InvoiceDate'])

sd_predict['InvoiceDate']=sd_predict['InvoiceDate'].map(dt.datetime.toordinal)

x=sd_predict['InvoiceDate']

x_predict=x.values.reshape(-1,1)

submit=pd.read_csv("../input/sample_submission.csv")

submit.head()
submit['TotalSales']=reg.predict(x_predict)

export_csv = submit.to_csv(r'export_dataframe.csv', index = None, header=True) #choose your directory where you want to save that .csv file 