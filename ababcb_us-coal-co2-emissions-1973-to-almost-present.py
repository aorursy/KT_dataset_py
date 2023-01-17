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
#Import necessary libraries

import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt



from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import Ridge

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error 

ds = pd.read_csv('/kaggle/input/co2emissions/MER_T12_06.csv')



# Coal Electric Power Sector CO2 Emissions

ds_1 = ds[ds['Column_Order']==1]

# Remove yearly values

ds_1 = ds_1[~ds_1.index.isin(ds_1[12::13].index)].sort_values(['YYYYMM'])

x = pd.to_datetime(ds_1['YYYYMM'], format='%Y%m')

y = ds_1['Value'].astype('float64')



plt.figure(figsize=(25, 10))

plt.xlabel('Date')

plt.ylabel('Million Metric Tons of Carbon Dioxide')



plt.scatter(x,y)
x_scaler = StandardScaler()

y_scaler = StandardScaler()



x_scaler.fit(x.values.reshape(-1, 1))

y_scaler.fit(y.values.reshape(-1, 1))



X = x_scaler.transform(x.values.reshape(-1, 1))

Y = y_scaler.transform(y.values.reshape(-1, 1))







x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)



X_train = x_scaler.transform(np.array(x_train).reshape(-1, 1))

Y_train = y_scaler.transform(np.array(y_train).reshape(-1, 1))

X_test = x_scaler.transform(np.array(x_test).reshape(-1, 1))

Y_test = y_scaler.transform(np.array(y_test).reshape(-1, 1))



print(X_train.shape)

print(Y_train.shape)

print(X_test.shape)

print(Y_test.shape)


plt.figure(figsize=(25, 10))

plt.xlabel('Date')

plt.ylabel('Million Metric Tons of Carbon Dioxide')





plt.scatter(x, y, alpha=0.3)



for deg in range(1,18):

    model = make_pipeline(PolynomialFeatures(deg), Ridge(alpha = 0))

    model.fit(X_train,Y_train)

    

    r_test = mean_squared_error(Y_test, model.predict(X_test))

    print('{:13f}: {}, {}'.format(deg, r_test, model.score(X_test, Y_test)))

    

    plt.plot(x,y_scaler.inverse_transform(model.predict(X)),label = 'deg={}'.format(deg))



plt.legend()




plt.figure(figsize=(25, 10))

plt.xlabel('Date')

plt.ylabel('Million Metric Tons of Carbon Dioxide')



# alpha=transparency

plt.scatter(x, y, alpha=0.3)



# This is the alpha we're actually interested in.

for alpha in np.logspace(-5, 5, 11):

    model = make_pipeline(PolynomialFeatures(6), Ridge(alpha = alpha))

    model.fit(X_train,Y_train)

    

    r_test = mean_squared_error(Y_test, model.predict(X_test))

    print('{:13f}: {}, {}'.format(alpha, r_test, model.score(X_test, Y_test)))

    

    plt.plot(x,y_scaler.inverse_transform(model.predict(X)),label = 'alpha={}'.format(alpha))



plt.legend()
plt.figure(figsize=(25, 10))

plt.xlabel('Date')

plt.ylabel('Million Metric Tons of Carbon Dioxide')



for i in range(1,10):

    ds_2 = ds[ds['Column_Order']==i]

    # Remove yearly values

    ds_2 = ds_2[~ds_2.index.isin(ds_2[12::13].index)].sort_values(['YYYYMM'])

    ds_2 = ds_2[ds_2['Value'] != 'Not Available']

    x2 = pd.to_datetime(ds_2['YYYYMM'], format='%Y%m')

    y2 = ds_2['Value'].astype('float64')

    plt.scatter(x2,y2, s=20, label=ds_2['Description'].iloc[0], alpha=0.30)

    

    x2scaler = StandardScaler()

    y2scaler = StandardScaler()

    

    X2 = x2scaler.fit_transform(x2.values.reshape(-1, 1))

    Y2 = y2scaler.fit_transform(y2.values.reshape(-1, 1))

    

    model = make_pipeline(PolynomialFeatures(6), Ridge(alpha = 0.001))

    model.fit(X2,Y2)

    

    plt.plot(x2,y2scaler.inverse_transform(model.predict(X2)))



plt.legend()




