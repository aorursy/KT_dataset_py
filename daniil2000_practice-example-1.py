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

rainf = pd.read_csv('../input/rainfdata/datasets_7018_31508_data_monthly_rainfall.csv', delim_whitespace = True)

new_rainf=rainf['Year,Station,Month,Rainfall,StationIndex'].str.split(',' ,expand=True)

new_rainf.columns=['Year','Station','Month', 'Rainfall', 'StationIndex']

print ('shape:', new_rainf.shape)

new_rainf[["Year", "Month", "Rainfall"]] = new_rainf[["Year", "Month", "Rainfall"]].apply(pd.to_numeric)

new_rainf.dtypes

#new_rainf['Rainfall'].plot()

new_rainf.reset_index().plot.scatter(x = 'Year', y = 'Rainfall')
import seaborn as sns

sns.lmplot("Month", "Rainfall", new_rainf)

import seaborn as sns

sns.lmplot("Year", "Rainfall", new_rainf)
month_means = pd.Series([])

for i in range(12): 

  month_means[i]=new_rainf.Rainfall[new_rainf.Month == i+1].mean()

  print (month_means[i])

  i=i+1

print(month_means.mean())

#print(month_means.var())
for i in range(12):

  new_rainf.Rainfall[new_rainf.Month == i+1]=100*(new_rainf.Rainfall[new_rainf.Month == i+1]- month_means[i])/month_means.mean()

  print(new_rainf.Rainfall[new_rainf.Month == i+1])

  sns.lmplot("Month", "Rainfall", new_rainf)

sns.lmplot("Year", "Rainfall", new_rainf)
from sklearn.linear_model import LinearRegression

est = LinearRegression(fit_intercept = True)
x = new_rainf[['Year']]

y = new_rainf[['Rainfall']]

est.fit(x, y)

print ("Coefficients:", est.coef_)

print ("Intercept:", est.intercept_)

from sklearn import metrics

y_hat = est.predict(x)

print ("MSE:", metrics.mean_squared_error(y_hat , y))

print ("R^2:", metrics.r2_score(y_hat , y))

print ('var:', y.var())