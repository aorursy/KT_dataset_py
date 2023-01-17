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
df1 = pd.read_csv('/kaggle/input/crimes-in-boston/offense_codes.csv', encoding = 'latin-1')
#offense codes table

df1.head()
df2 = pd.read_csv('/kaggle/input/crimes-in-boston/crime.csv', encoding = 'latin-1')
#incidents table

df2.head()
#drop all unnecessary columns



df2.drop(['INCIDENT_NUMBER','OFFENSE_CODE','SHOOTING','YEAR','MONTH','DAY_OF_WEEK','HOUR','UCR_PART','Lat','Long','Location','STREET'], axis = 1, inplace = True)
df2.head()
#drop OFFENSE_DESCRIPTION as its just a descriptive form of OFFENSE_CODE_GROUP

df2.drop(['OFFENSE_DESCRIPTION'], axis = 1, inplace = True)
df2.head()
# you know what...

# lets first do it for the entire city and not district wise

# so drop DISTRICT and REPORTING_AREA as well



df2.drop(['DISTRICT','REPORTING_AREA'], axis = 1, inplace = True)
df2.head()
# OCCURED_ON_DATE is in DATETIME 

# we are concerned only with DATE so...keep only DATE



df2['DATE'] = pd.to_datetime(df2['OCCURRED_ON_DATE']).dt.date
df2.head()
# self explanatory



df2.drop(['OCCURRED_ON_DATE'], axis = 1, inplace = True)
df2.head()
len(df2['OFFENSE_CODE_GROUP'].unique().tolist()) # => 67 unique types of offenses
# Lets see if all are valid ..

# no Nan exists



df2['OFFENSE_CODE_GROUP'].unique().tolist()
#everything is valid...Gr8 !



# use this if Nan exists -> df2.dropna(inplace = True)
#D14_dates = pd.DataFrame()
dates = pd.pivot_table(df2, index=['DATE'], aggfunc='count')
dates.head()
# GET IT ??
len(dates)
# create a list of all dates 

idx = pd.date_range('2015-06-15', '2018-09-03')
len(idx)
# Luckily for us crime in a daily affair in BOSTON



# Just in case it wasn't use this 



dates = dates.reindex(idx, fill_value = 0)



dates
# remove column

dates = dates.rename(columns={'OFFENSE_CODE_GROUP':'OCCURENCES'})  
dates.head()
from matplotlib import pyplot



# lets visualize the data over the entire available time period



dates.plot()

pyplot.show()
from pandas.plotting import autocorrelation_plot



from statsmodels.graphics.tsaplots import plot_acf

from statsmodels.graphics.tsaplots import plot_pacf



plot_acf(dates, lags = 10)
plot_pacf(dates, lags = 10)
from statsmodels.tsa.arima_model import ARIMA



model = ARIMA(dates, order=(1,1,7))
model_fit = model.fit(disp=0)
residuals = pd.DataFrame(model_fit.resid)

residuals.plot()

pyplot.show()

residuals.plot(kind='kde')

pyplot.show()

print(residuals.describe())
from sklearn.metrics import mean_squared_error



D = dates.values

size = int(len(D) * 0.66)

train, test = D[0:size], D[size:len(D)]

history = [x for x in train]

predictions = list()



for t in range(len(test)):

    model = ARIMA(history, order=(1,1,7))

    model_fit = model.fit(disp=0)

    output = model_fit.forecast()

    yhat = output[0]

    predictions.append(yhat)

    obs = test[t]

    history.append(obs)

    print('predicted= %f, expected = %f' %(yhat,obs))

    

error = mean_squared_error(test, predictions)

print(' Test MSE : %3f' %error)





pyplot.plot(test)

pyplot.plot(predictions, color= 'red')

pyplot.show()