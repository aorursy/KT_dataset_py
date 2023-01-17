# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
%matplotlib inline

import numpy as np

import pandas as pd

df = pd.read_csv('../input/weatherHistory.csv')

df.head()
import seaborn as sns

sns.pairplot(df[["Precip Type","Temperature (C)","Apparent Temperature (C)","Humidity"]],

             hue="Precip Type",

             palette="YlGnBu")
corr = df.drop('Loud Cover', axis=1).corr() 

sns.heatmap(corr,  cmap="YlGnBu", square=True);
from sklearn import linear_model

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score



ls = linear_model.LinearRegression()

data = df.where(df['Precip Type']!='null')

data.dropna(inplace=True)





X = data["Humidity"].values.reshape(-1,1)

y = data["Temperature (C)"].values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33,random_state=0)

ls.fit(X_train, y_train)

y_pred = ls.predict(X_test)

print("MSE Linear regression = ",mean_squared_error(y_test, y_pred))

print("R2 Linear regression = ",r2_score(y_test, y_pred))
from sklearn.ensemble import RandomForestRegressor

reg = RandomForestRegressor()



reg.fit(X_train, y_train)



y_pred_forest = reg.predict(X_test)

print("MSE Randomn Forest Regressor= ",mean_squared_error(y_test, y_pred_forest))

print("R2 Randomn Forest Regressor = ",r2_score(y_test, y_pred_forest))
humidity_example = np.array(0.7)

temperature_output = reg.predict(humidity_example.reshape(1,-1))

print("For such {} humidity, Randomn Forest Regression predict a temperature of {}C".format(humidity_example, temperature_output))