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
#import data

df=pd.read_excel('/kaggle/input/spartaworldbank.xlsx')

df.head()
#dropping industry 

#also dropping year due to complications of time series regression i.e. collinearity and lag

del df['Year']

del df['Industry (including construction), value added (% of GDP)']
df['intercept'] = 1 #Adding a column corresponding to intercept
#Set variables for regression

X=df.drop(["Earth's surface (land and ocean - Fahreinheit)"], axis=1)

y=df["Earth's surface (land and ocean - Fahreinheit)"]

from sklearn import linear_model



lm = linear_model.LinearRegression(fit_intercept=False)

model_sklearn = lm.fit(X, y)

predictions = lm.predict(X)

import seaborn as sns

sns.set_style("darkgrid")

sns.set(rc={'figure.figsize':(11.7,8.27)})

ax=sns.regplot(x=predictions, y= y, data=df,color='red',ci=95)

ax.set(xlabel='Predicted Temperature', ylabel='True Temperature')

ax.set_title('Predicted Temperature vs. True Temperature')
import statsmodels.api as sm

model = sm.OLS(y, X)

results = model.fit()

print(results.summary())