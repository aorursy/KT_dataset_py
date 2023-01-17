# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import preprocessing #(For data transformation)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


os.getcwd()
mvs=pd.read_csv('/kaggle/input/mktmix.csv',sep=',',header=0, na_values=["Missing", " "])
mvs.head()
mvs.shape
mvs.dtypes
mvs.describe()
#plt.hist(mvs.Base_Price, bins = 5)

plt.hist(mvs.Base_Price)

plt.show()

mvs.boxplot(column='Base_Price')
mvs['Base_Price'].quantile(np.arange(0,1,0.1))
min_value= mvs['Base_Price'].quantile(0.01)

min_value
mvs.loc[(mvs.Base_Price<min_value)]
mvs.loc[(mvs.Base_Price<min_value), "Base_Price"]= min_value
mvs.boxplot(column='Base_Price')
# Base Price

mvs.plot(x="NewVolSales",y="Base_Price",kind="scatter")
pd.options.display.float_format = '{:.2f}'.format

mvs.corr()

# Correlation between NewVolSales and Base Price is high and Radio is least correlated
import seaborn as sns

corr = mvs.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.index.values)

new = mvs.corr()

new.to_csv('corr.csv', sep=',')
def find_bkt(x):

    #high= data.quartile(0.75)

    #low= data.quartile(0.25)

    if x>15.75:

        return "High"

    elif x>15:

        return "Medium"

    else:

        return "Low"

    
mvs["bkt_price"]= mvs.Base_Price.map(find_bkt)
mvs.bkt_price
mvs['bkt_price'].value_counts()

#Finding frequency of high medium and low values
mvs=pd.get_dummies(mvs)

#mvs = pd.get_dummies(mvs, columns=['Website_Campaign'])

#mvs = pd.get_dummies(mvs, columns=['NewspaperInserts'])
mvs.head()
import statsmodels.formula.api as smf

reg=smf.ols("NewVolSales~Base_Price+InStore+TV+Discount+Stout",data=mvs)

results=reg.fit()

print(results.summary())
#Predictions

predictions=results.predict(mvs)
actual=mvs.NewVolSales
## Actual vs Predicted plot

plt.plot(actual,"b")

plt.plot(predictions,"red")

plt.figure(figsize=(50,30))
residuals=results.resid

type(residuals)
plt.scatter(actual,residuals)

#Scatter plot of residuals must be random
import sklearn.metrics as metrics

mae = metrics.mean_absolute_error(actual,predictions)

mae
np.mean(abs((predictions-actual)/actual))