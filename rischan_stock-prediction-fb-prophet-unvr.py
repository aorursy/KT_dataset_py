# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

unvr = pd.read_csv("/kaggle/input/indonesia-popular-stocks/10_popular_indonesian_stock/UNVR.JK.csv") 

unvr.head(5)
unvr.describe()
# Select only the important features i.e. the date and price

unvr = unvr[["Date","Close"]] # select Date and Price

# Rename the features: These names are NEEDED for the model fitting

unvr = unvr.rename(columns = {"Date":"ds","Close":"y"}) #renaming the columns of the dataset

unvr.head(5)
from fbprophet import Prophet

munvr = Prophet(daily_seasonality = True) # the Prophet class (model)

munvr = Prophet(daily_seasonality = True) # the Prophet class (model)

munvr.fit(unvr) # fit the model using all data
future = munvr.make_future_dataframe(periods=365) #we need to specify the number of days in future

prediction = munvr.predict(future)

munvr.plot(prediction)

plt.title("Prediction of the UNVR Stock Price using the Prophet")

plt.xlabel("Date")

plt.ylabel("Close Stock Price")

plt.show()
munvr.plot_components(prediction)

plt.show()