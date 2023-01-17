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

from scipy import stats

import statsmodels.api as sm

import matplotlib.pyplot as plt



%matplotlib inline
df = pd.read_csv('../input/into-the-future/train.csv')
df.head()
df.dtypes
#we need to convert time to datetime 

df['time'] = pd.to_datetime(df['time'])

df.drop('id',axis=1,inplace=True)

df.set_index('time',inplace=True)
df.tail()
print(df.shape)

plt.plot(df['feature_1'])

plt.plot(df['feature_2'])
from fbprophet import Prophet

data = df.reset_index()

data.tail(n=3)
data2 = data[['time','feature_2']].reset_index()

data2.drop('index',axis=1,inplace=True)

data2.columns = ['ds', 'y']
#train test

prediction_size = 60

train_df2 = data2[:-60]

train_df2.tail()
m = Prophet()

m.fit(train_df2)
future = m.make_future_dataframe(periods=435, freq='10S')

future.tail(n=3)
forecast = m.predict(future)

forecast.tail(n=3)
m.plot_components(forecast)
fcast = forecast[504:563]['yhat']

fcast.head()
def score(df, fcast):

    

    df = pd.DataFrame()

    

    df['error'] = data2[504:563]['y'] - fcast

    df['relative_error'] = 100*df['error']/data2[504:563]['y']

    

    

    error_mean = lambda error_name: np.mean(np.abs(df[error_name]))

    

    

    return {'MAPE': error_mean('relative_error'), 'MAE': error_mean('error')}
for err_name, err_value in score(data2, fcast).items():

    print(err_name, err_value)
test = pd.read_csv('../input/into-the-future/test.csv')
d = forecast[564:]['yhat']
final = pd.DataFrame()

final['id'] = test['id']

final['feature_2'] = list(d)
final.head()
final.to_csv("/kaggle/working/solution.csv", index=False)