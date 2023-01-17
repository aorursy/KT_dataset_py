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
#Model 6 = v12

#Model 5 = v13 

#Model 4 = v14 etc.
import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

import scipy as scip
df=pd.read_excel('/kaggle/input/newapproach6661/new approach RP.xlsx')

df.head()

df2=pd.read_excel('/kaggle/input/newapproach666/new approach RP.xlsx')

df2.head()

d=df2['Labor force with basic education (% of total working-age population with basic education)']
df=dat1 = pd.concat([df, d], axis=1)

del df['School enrollment, secondary (% net)']

df.head()
df.isnull().sum()
del df['Region']

columns = df.loc[:, df.columns != 'Country Name']

for column in columns:

    df[column] = df[column].astype(float)



df.head()
df = df.dropna(axis=0, subset=['Trade (% of GDP)'])

df = df.dropna(axis=0, subset=['Social Globalisation'])

df = df.dropna(axis=0, subset=['Strength of legal rights index (0=weak to 12=strong)'])











df.isnull().sum()
len(df)
#Check normality, see if a log transformation would help

from scipy import stats

stats.probplot(df['GDP growth (annual %)'], plot=plt)
stats.probplot(np.log(df['GDP growth (annual %)']), plot=plt)
corrmat = df.corr()

f, ax = plt.subplots(figsize=(35, 15))

sns.heatmap(corrmat, square=True, cmap='YlGnBu',annot=True)




df["ln(Foreign direct investment, net inflows (% of GDP))"] = np.log(df["Foreign direct investment, net inflows (% of GDP)"])

del df['Foreign direct investment, net inflows (% of GDP)']

df["ln(Trade (% of GDP))"] = np.log(df["Trade (% of GDP)"])

del df['Trade (% of GDP)']



df["2016 GDP (constant 2010 US$)"] = np.log(df["2016 GDP (constant 2010 US$)"])

del df['2016 GDP (constant 2010 US$)']









df.head()
df=df.drop(['Economic Globalisation','High-technology exports (% of manufactured exports)','Logistics performance index: Quality of trade and transport-related infrastructure (1=low to 5=high)'], axis=1)

df=df.drop(['Tariff rate, applied, weighted mean, all products (%)','Inflation, consumer prices (annual %)','Labor force with basic education (% of total working-age population with basic education)'], axis=1)



corrmat = df.corr()

f, ax = plt.subplots(figsize=(35, 15))

sns.heatmap(corrmat, square=True, cmap='YlGnBu',annot=True)
df = df.dropna(axis=0, subset=['ln(Foreign direct investment, net inflows (% of GDP))'])

from sklearn import linear_model

df = df.dropna(axis=0, subset=['GDP growth (annual %)'])

df['intercept'] = 1 #Adding a column corresponding to intercept

X=df.drop(['GDP growth (annual %)', 'Country Name'], axis=1)

y=df['GDP growth (annual %)']

X.head()
lm = linear_model.LinearRegression(fit_intercept=False)

model_sklearn = lm.fit(X, y)

predictions = lm.predict(X)

sns.set_style("darkgrid")

sns.set(rc={'figure.figsize':(11.7,8.27)})

ax=sns.regplot(x=predictions, y= y, data=df,color='purple',ci=95)

ax.set(xlabel='Predicted ln(annual GDP growth)', ylabel='True ln(annual GDP growth)')

ax.set_title('Predicted ln(annual GDP growth) vs. True ln(annual GDP growth)')
import statsmodels.api as sm

model = sm.OLS(y, X)

results = model.fit()

print(results.summary())
sns.residplot(x=predictions, y=y)
from statsmodels.stats.outliers_influence import variance_inflation_factor



vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print(vif)
from sklearn.metrics import mean_squared_error

mean_squared_error(y, predictions)
stats.probplot(predictions-y, plot=plt)
stats.probplot(y, plot=plt)
print (df.describe())