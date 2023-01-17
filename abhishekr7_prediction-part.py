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



df = pd.read_csv("/kaggle/input/crimes-in-boston/crime.csv", engine="python")
df
df = df[['OFFENSE_CODE_GROUP','OCCURRED_ON_DATE','YEAR','MONTH','DAY_OF_WEEK','Lat','Long']]
df = df.dropna()
df.reset_index(inplace=True)
lis = []



for index, row in df.iterrows():

    li = []

    li.append(row['Lat'])

    li.append(row['Long'])

    lis.append(li)
X = np.array(lis)
from sklearn.cluster import KMeans

import numpy as np

kmeans = KMeans(n_clusters=10, random_state=0).fit(X)
df['Cluster'] = kmeans.labels_.tolist()
df['OCCURRED_ON_DATE'] =pd.to_datetime(df.OCCURRED_ON_DATE)
df = df.sort_values(by='OCCURRED_ON_DATE')

df.reset_index(inplace = True)
lisp = []



for index, row in df.iterrows():

    lisp.append(row['OCCURRED_ON_DATE'].date())

    

df['OCCURRED_ON_DATE'] = lisp
df1 = df[df['Cluster'] == 0]

df2 = df[df['Cluster'] == 2]

df3 = df[df['Cluster'] == 3]

df4 = df[df['Cluster'] == 4]

df5 = df[df['Cluster'] == 5]

df6 = df[df['Cluster'] == 6]

df7 = df[df['Cluster'] == 7]

df8 = df[df['Cluster'] == 8]

df9 = df[df['Cluster'] == 9]
df1 = df1.groupby(['OCCURRED_ON_DATE']).count()[['level_0','index']]

df1.drop(columns=['index'], inplace = True)
df_train = df1[0:941]



df_train = df_train.reset_index()

df_train.rename(columns={'OCCURRED_ON_DATE':'ds', 'level_0':'y'} , inplace = True)
df_train
from fbprophet import Prophet

from fbprophet.plot import plot_plotly

m = Prophet()

m.fit(df_train)
future = m.make_future_dataframe(periods=236)

future.tail()
forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig1 = m.plot(forecast)
predicted = forecast['yhat'][941:1177].tolist()
actual = df1[941:1177]
from sklearn.metrics import mean_squared_error

from math import sqrt



rms = sqrt(mean_squared_error(actual, predicted))
rms
mse = mean_squared_error(actual, predicted)
mse
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(actual, predicted)
mae
from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

from math import sqrt



def get_metrics(y_actual, y_predicted):

    print("***************** Prediction Metrics *******************\n\n")

    print("Root mean squared error (RMSE) => " , sqrt(mean_squared_error(y_actual, y_predicted)))

    print("Mean squared error (MSE) => " , mean_squared_error(y_actual, y_predicted))

    print("Mean absolute error (MAE) => ", mean_absolute_error(y_actual, y_predicted))
get_metrics(actual, predicted)
df[df['Cluster'] == 0]