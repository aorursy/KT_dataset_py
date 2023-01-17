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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import Imputer

import seaborn as sns

from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error

impute = Imputer(np.nan,strategy="mean")

df = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

#index= [overallqual,totalbsmtsf,1stflrsf,grlivarea,garagecars,garagearea]

#for training data

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()



correlation = df.corr()

sns.set_style(style="whitegrid")

x = df.iloc[:,[17,38,43,46,61,62]]

y = df.iloc[:,-1].values

x = impute.fit_transform(x)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25)

regressor = RandomForestRegressor(n_estimators=300,random_state=0)

regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

error = r2_score(y_test,y_pred)



#for testing data



test_df = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

id1 = test_df["Id"]

x1 = test_df.iloc[:,[17,38,43,46,61,62]]

x1 = impute.fit_transform(x1)

result = regressor.predict(x1)

output = pd.DataFrame({ 'Id': id1,

                            'SalePrice': result })

output.to_csv("output.csv")
