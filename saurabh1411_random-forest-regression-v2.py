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

import seaborn as sns
df=pd.read_csv("../input/car-speeding-and-warning-signs/amis.csv")
df.head(5)
# visualize the relationship between the features and the response using scatterplots

sns.pairplot(df, x_vars=['speed','period','pair'], y_vars='warning', size=7, aspect=0.7)
x=df.speed.values.reshape(-1,1)

y=df.warning.values.reshape(-1,1)
plt.figure(figsize=(18,6))

plt.scatter(x,y,color="g")



plt.xlabel("speed")

plt.ylabel("warning")

plt.show()
from sklearn.ensemble import RandomForestRegressor

regressor=RandomForestRegressor(n_estimators=100,random_state=0)
regressor.fit(x,y)
regressor.predict([[7.5]])
x_=np.arange(min(x),max(x),0.01).reshape(-1,1)

y_head=regressor.predict(x_)
plt.figure(figsize=(18,6))

plt.scatter(x,y,color="r")

plt.plot(x_,y_head,color="y")

plt.xlabel("Cost")

plt.ylabel("warning")

plt.show()
import statsmodels.formula.api as smf
z=smf.ols(formula='warning~speed',data=df).fit()
z.pvalues
z.rsquared
regressor.summary()