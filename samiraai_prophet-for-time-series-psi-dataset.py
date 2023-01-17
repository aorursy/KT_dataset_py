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



import datetime



import matplotlib.pyplot as plt



from fbprophet import Prophet

from fbprophet.plot import add_changepoints_to_plot

from fbprophet.diagnostics import cross_validation

from fbprophet.diagnostics import performance_metrics

from sklearn.metrics import mean_squared_error, r2_score ,mean_absolute_error



# read data 

df = pd.read_csv('/kaggle/input/singapore-psi-pm25-20162019/psi_df_2016_2019.csv')

df['timestamp'] = pd.to_datetime(df['timestamp'])

df['timestamp'] = df['timestamp'].dt.tz_localize(None)

#print(df.isnull().sum())



column = ['national' , 'south' , 'north' , 'east', 'central' , 'west']



train_prophet = pd.DataFrame()

test_prophet = pd.DataFrame()

for i in column:

    train_size_prophet = int(len(df) * 0.7)

    train_prophet['ds'] = df['timestamp'].iloc[:train_size_prophet]

    train_prophet['y'] = df[i].iloc[:train_size_prophet]

    test_prophet['ds'] = df['timestamp'].iloc[train_size_prophet:]

    test_prophet['y'] = df[i].iloc[train_size_prophet:]



    m = Prophet()

    m.fit(train_prophet)

    forecast = m.predict(test_prophet)

    #forecast.head()

    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

    fig1 = m.plot(forecast)

    fig2 = m.plot_components(forecast)

    plt.show(fig1)

    plt.show(fig2)

    f, ax = plt.subplots(1)

    f.set_figheight(5)

    f.set_figwidth(15)

    ax.scatter(test_prophet['ds'], test_prophet['y'], color='r')

    fig = m.plot(forecast, ax=ax)

    plt.show(fig)

    mse = mean_squared_error(y_true=test_prophet['y'],y_pred=forecast['yhat'])

    print("MSE :" ,mse)