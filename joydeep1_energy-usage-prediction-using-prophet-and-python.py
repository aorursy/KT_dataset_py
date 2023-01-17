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
import pystan
# Import required packages



%matplotlib inline



import os

import pandas as pd

import numpy as np

from fbprophet import Prophet
# Read the data and find out it's dimension

endata = pd.read_csv("/kaggle/input/appliances-energy-prediction/KAG_energydata_complete.csv")

endata.shape
# Find glimpse of few lines of the data

endata.head()
# Get only two columns, expected by Prophet

data = endata[['date', 'Appliances']]

data.head()
# Check the variation in the data 

data.set_index('date').plot(figsize=(12,9))
# Check how the new dataframe, 'data' looks like

data
# Convert the column names as expected by prophet

data.columns = ['ds', 'y']

data
# Perform logarithmic trasformation of the Appliances/y data, to smoothen the variation

np.log(data['y'])
# Now check the variations within the data after transformation

data.set_index('ds').plot(figsize=(10,8))
data.tail()
# Call prophet 

model1 = Prophet()
#Fit the model by instantiating a new Prophet object

model1.fit(data)
# Getting a 'future' dataframe for a specified number of days, 1 year in this case



future = model1.make_future_dataframe(periods=365)

future.tail()
# The forecast object here is a new dataframe that includes a column yhat with the forecast, 

# as well as columns for components and uncertainty intervals.

forecast = model1.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
# Plot the forecast by calling the Prophet.plot method and passing in the forecast dataframe.

fig1 = model1.plot(forecast)
# If you want to see the forecast components, you can use the Prophet.plot_components method. By default you’ll see the trend, yearly seasonality, and weekly seasonality of the time series. 

# If you include holidays, you’ll see those here, too.

fig2 = model1.plot_components(forecast)
# An interactive figure of the forecast can be created with plotly. 

# You will need to install plotly separately, as it will not by default be installed with fbprophet.



from fbprophet.plot import plot_plotly

import plotly.offline as py

py.init_notebook_mode()



fig_new = plot_plotly(model1, forecast)  # This returns a plotly Figure

py.iplot(fig_new)