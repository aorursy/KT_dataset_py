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
import pandas as pd
import numpy as np
import scipy
import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
import statsmodels.api as sm
import itertools
import warnings
from pylab import rcParams


%matplotlib inline

pd.options.display.float_format = '{:20,.3f}'.format
pd.options.display.max_columns = None
pd.options.display.max_colwidth = 1000
np.set_printoptions(precision=3)

matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'
df_complaint = pd.read_csv('../input/capstone/311_Service_Requests_from_2010_to_Present_min.csv')
 #Combine HEAT/HOT WATER complaint to HEATING complaint as it is merged in the dataset from 2014##
df_complaint['Complaint Type'] = np.where(df_complaint['Complaint Type']=='HEATING','HEAT/HOT WATER',df_complaint['Complaint Type'])

df_complaint['Complaint Type'].value_counts()
df_complaint.head(5)
### Changing the Object Created Date to a dtype Datetime ####
df_complaint['Created Date'] =  pd.to_datetime(df_complaint['Created Date'])
### Verify that the Data Type has changed##
df_complaint.info()
col = ['Address Type', 'Borough', 'City', 'Closed Date', 'Complaint Type',
        'Incident Address', 'Incident Zip', 'Latitude', 'Location Type', 'Longitude', 'Resolution Description', 
        'Status','Street Name']
df_complaint_ml = df_complaint.drop(col, axis = 1)
df_complaint_ml.columns

###Verify again the data is correct###
df_complaint_ml.head()
#### We dont need the exact time we really just need the date and year###
### Extract this out of the Created Date Col###
df_complaint_ml['Date'] = df_complaint_ml['Created Date'].map(lambda x: x.strftime('%Y-%m'))
df_complaint_ml['Created Year'] = df_complaint_ml['Created Date'].map(lambda x: x.strftime('%Y')).astype(int)

### Check again that there are now new cols in the dataframe## and check the dtype to make sure year is int64
df_complaint_ml.head()
## Drop all columns not needed as you only need the dates####
## The Goal here is to get the Unique Keys and Date###
df_complaint_ml = df_complaint_ml.drop(['Created Date', 'Created Year'], axis =1)
df_complaint_ml = df_complaint_ml.sort_values('Date')
df_complaint_ml.head()
df_complaint_ml.head()
### Rename now to complaints##
df_complaint_ml.rename(columns={'Unique Key': 'Complaints'}, inplace=True)
df_complaint_ml.head()
### Now we count the number of complaints - each one has a unique key###
df_complaint_ml = df_complaint_ml.groupby(['Date']).count()['Complaints']
####Resetting the index of the dataframe to the date##
df_complaint_ml = df_complaint_ml.reset_index('Date')
df_complaint_ml['Date'] = pd.to_datetime(df_complaint_ml['Date'])
df_complaint_ml = df_complaint_ml.set_index('Date')

 

df_complaint_ml.head()
df_complaint_ml = df_complaint_ml.sort_values('Date')
### Just plot so we can see there are peaks and seasons## Not quite so clear
df_complaint_ml.plot(figsize=(15, 6))
plt.show()
rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(df_complaint_ml, model='additive')
fig = decomposition.plot()
plt.show()

#### Evaluating above we can see that there is a seasonality to the complaints but there are also a few days that are outliers##
df_complaint_ml.dtypes
#### On kaggle you can not install Auto ARIMA so run this by importing on local python and find the seasonality to use here###

mod = sm.tsa.statespace.SARIMAX(df_complaint_ml,
                                order=(0, 1, 1),
                                seasonal_order=(0, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()

print(results.summary().tables[1])
results.plot_diagnostics(figsize=(15, 12))
plt.show()
pred = results.get_prediction(start=pd.to_datetime('2018-01'), dynamic=False)
pred_ci = pred.conf_int()
ax = df_complaint_ml['2010':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='Forecast', alpha=.7)

ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)

ax.set_xlabel('Date')
ax.set_ylabel('No of Complaints')
plt.legend()

plt.show()
pred_uc = results.get_forecast(steps=10)
pred_ci = pred_uc.conf_int()
ax = df_complaint_ml.plot(label='observed', figsize=(20, 15))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('No of Complaints')

plt.legend()
plt.show()
