# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import scipy.stats as stats

from sklearn.tree import DecisionTreeRegressor



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



df = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/train.csv")  

df.head(30)

df.dtypes

df['Date']

df.info()

df.isna().sum()

country_province = df.fillna('N/A').groupby(['Country/Region','Province/State'])['ConfirmedCases', 'Fatalities'].max().sort_values(by='ConfirmedCases', ascending=False)



country_province.head(10)
plt.figure(figsize=(15,15))

sns.heatmap(df.corr(),linewidths=0.1,vmax=1.0, 

            square=True, linecolor='white', annot=True)
df.describe()



Skewness = pd.DataFrame({'Skewness' : [stats.skew(df.ConfirmedCases),stats.skew(df.Fatalities)]},

                        index=['confirmedcases','fatalities'])  # Measure the skeweness of the required columns

Skewness
sns.scatterplot(df.ConfirmedCases,df.Fatalities)

data=df

data.head()
data.drop('Province/State',axis=1,inplace=True)
data.drop('Country/Region',axis=1,inplace=True)
data.drop('Lat',axis=1,inplace=True)
data.drop('Long',axis=1,inplace=True)
data.drop('Date',axis=1,inplace=True)
data.head()
x=data

y=data[['Fatalities']]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=1)
from sklearn.tree import DecisionTreeRegressor  

  

# create a regressor object 

regressor = DecisionTreeRegressor(random_state = 0) 

# fit the regressor with X and Y data 

regressor.fit(x, y) 
y_pred = regressor.predict(X_test)
y_pred
from sklearn.linear_model import LinearRegression

regression_model = LinearRegression()

regression_model.fit(X_train, y_train)
# Let us explore the coefficients for each of the independent attributes



for idx, col_name in enumerate(X_train.columns):

    print("The coefficient for {} is {}".format(col_name, regression_model.coef_[0][idx]))
intercept = regression_model.intercept_[0]



print("The intercept for our model is {}".format(intercept))


regression_model.score(X_test, y_test)