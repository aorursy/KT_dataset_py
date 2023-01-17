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

from sklearn.ensemble import RandomForestRegressor

import keras







# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Importing data

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score
# Importing and filtering data

tot_cases=pd.read_csv('../input/covid-19-data/total_cases.csv',parse_dates=[0],usecols=['date','India'],date_parser=lambda x: pd.datetime.strptime(x,'%Y-%m-%d'))

tot_deaths=pd.read_csv('../input/covid-19-data/total_deaths.csv',parse_dates=[0],usecols=['date','India'],date_parser=lambda x: pd.datetime.strptime(x,'%Y-%m-%d'))

tot_cases.set_index('date',inplace=True)

tot_deaths.set_index('date',inplace=True)
# Taking data after starting from 12th march

tot_cases=tot_cases.loc['2020-03-12':]

tot_deaths=tot_deaths.loc['2020-03-12':]
def linear_regression_with_poly_features(data,title):

    x=np.array(range(data.shape[0])).reshape((data.shape[0],1))

    y=np.array(data['India'].values).reshape((data.shape[0],1))

    poly_features=PolynomialFeatures(degree=4)

    x_poly=poly_features.fit_transform(x)

    print('x_poly shape: ',x_poly.shape)

    model=LinearRegression()

    #model=Lasso(alpha=0.3)

    x_train=x_poly[:-28]

    x_val=x_poly[-28:,]

    y_train=y[:-28]

    y_val=y[-28:]

    print('Fitting the model')

    model.fit(x_train,y_train)

    index=np.array(data.index).reshape((data.shape[0],1))

    train_index=index[:-28]

    val_index=index[-28:]

    print('train index shape: ',train_index.shape)

    print('prediction shape: ',model.predict(x_train).shape)

    #plt.figure(figsize=[10,8])

    plt.plot(train_index,y_train,label='Actual data')

    plt.plot(train_index,model.predict(x_train),label='Predicted data')

    plt.title('Training results for {}'.format(title))

    plt.legend()

    plt.show()

    #plt.figure(figsize=[10,8])

    plt.plot(val_index,y_val,label='Actual data')

    plt.plot(val_index,model.predict(x_val),label='Predicted data')

    plt.title('Validation results for {}'.format(title))

    plt.legend()

    plt.show()

    print('Metric values for training process: ')

    print('r2_score: {}'.format(r2_score(y_train,model.predict(x_train))))

    print('RMSE: {}'.format(np.sqrt(mean_squared_error(y_train,model.predict(x_train)))))

    print('Metrics for validation data: ')

    print('r2_score: {}'.format(r2_score(y_val,model.predict(x_val))))

    print('RMSE: {}'.format(np.sqrt(mean_squared_error(y_val,model.predict(x_val)))))

linear_regression_with_poly_features(tot_cases,'Total cases')
linear_regression_with_poly_features(tot_deaths,'Total deaths')