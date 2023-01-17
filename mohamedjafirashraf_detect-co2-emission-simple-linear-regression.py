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
#import needed libraries

import numpy as np

import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

mpl.style.use('ggplot')
#read the data

df = pd.read_csv('../input/fuel-consumption-co2/FuelConsumptionCo2.csv')

df.head()
#describtion of given data

df.describe()
#information about given data

df.info()
df = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','FUELCONSUMPTION_COMB_MPG','CO2EMISSIONS']]

df.head()
#finding missing values

missing_data = df.isnull()



for column in missing_data.columns:

    print(column)

    print(missing_data[column].value_counts())

    print("")
df['CYLINDERS'].value_counts().plot(kind='bar')
df.corr()
corr_train = df.corr()

plt.subplots(figsize=(12,9))

sns.heatmap(corr_train, vmax=0.9, square=True, center = 0)
#Correlation for each columns

df.plot(kind='scatter', x='ENGINESIZE', y='CO2EMISSIONS')

df.plot(kind='scatter', x='CYLINDERS', y='CO2EMISSIONS')

df.plot(kind='scatter', x='FUELCONSUMPTION_CITY', y='CO2EMISSIONS')

df.plot(kind='scatter', x='FUELCONSUMPTION_HWY', y='CO2EMISSIONS')

df.plot(kind='scatter', x='FUELCONSUMPTION_COMB', y='CO2EMISSIONS')

df.plot(kind='scatter', x='FUELCONSUMPTION_COMB_MPG', y='CO2EMISSIONS')

plt.show()
df.drop('FUELCONSUMPTION_COMB_MPG', axis=1, inplace=True)
df.head()
#start modeling

from sklearn.model_selection import train_test_split

from sklearn import linear_model

from sklearn.metrics import r2_score



#split the data

X = df.iloc[:,0:5]

y = df['CO2EMISSIONS']

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=1)



#define the model

reg = linear_model.LinearRegression()

reg.fit(train_X,train_y)
#co-efficient

print(reg.coef_)

#imtercept

print(reg.intercept_)




#prediction

pred = reg.predict(test_X)



#calculate the accuracy score and r2 score

print("Mean absolute error: %.2f" % np.mean(np.absolute(pred - test_y)))

print("Residual sum of squares (MSE): %.2f" % np.mean((pred - test_y) ** 2))

print("R2-score: %.2f" % r2_score(pred , test_y) )