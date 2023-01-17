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
#importing dataset using pandas



df = pd.read_csv('/kaggle/input/Advertising.csv')

df.head()
df = df.drop('Unnamed: 0', axis =1)
df.describe()
import matplotlib.pyplot as plt

import seaborn as sns
sns.pairplot(df)
df.corr()
sns.heatmap(df.corr(), annot = True)
label = df['sales']

features = df.drop('sales', axis =1)
df.shape
for x in features.columns:

    sns.scatterplot(features[x], label)

    plt.title("Sales vs " + x)

    plt.xlabel(x)

    plt.ylabel("sales")

    plt.show()

    
for x in features.columns:

    sns.distplot(features[x], bins ='auto')

    plt.title("Distribution plot of " + x)

    plt.xlabel(x)

    plt.ylabel("Frequency")

    plt.show()
from scipy import stats



fig = plt.figure()

ax1 = fig.add_subplot(211)

stats.probplot(df['newspaper'], dist= stats.norm, plot=ax1)

plt.show()
#Transforming the newspaper as the data is skewed towards right

fig = plt.figure()

ax2 = fig.add_subplot(212)

y, z= stats.boxcox(df['newspaper'])

stats.probplot(y, dist= stats.norm, plot=ax2)

plt.show()
#fitting the OLS model

import statsmodels.formula.api as sm



model1 = sm.ols(formula = "sales ~ TV+ radio + newspaper", data = df).fit()

model1.summary()
# A high p-value for newspaper signifies that it is not a significant parameter and it should be removed from the model.



model2 = sm.ols(formula= "sales~ TV + radio", data = df).fit()

model2.summary()
#testing a model without radio 



model3 = sm.ols(formula= "sales ~ TV", data = df).fit()

model3.summary()
#Evaluting few more significant parameter



print('Parameters: ', model2.params)

print('R2: ', model2.rsquared)

print('Adj R2: ', model2.rsquared_adj)

print('Standard errors: ', model2.bse)
#predicting sales values 



y_pred = model2.predict()

df_pred = pd.DataFrame({'Actual' : label, 'Predictions' :y_pred

                       })

df_pred.head()
#plotting a curve of best fit line by model2

import statsmodels.api as sm

fig = plt.figure(figsize=(12,8))

fig = sm.graphics.plot_regress_exog(model2, "TV", fig=fig)
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split



lm = LinearRegression()



train_x,test_x, train_y,test_y = train_test_split(features, label, test_size = 0.2, random_state = 42)

print(train_x.shape)

print(test_x.shape)

print(train_y.shape)

print(test_y.shape)
import sklearn.preprocessing as skp

nwsppr = np.array(train_x[['newspaper']])

normalizer = skp.PowerTransformer(method = 'box-cox', standardize = False)

df.newspaper = pd.DataFrame(normalizer.fit_transform(nwsppr))

train_x.head()
lm.fit(train_x,train_y)
print(lm.score(train_x,train_y))

print(lm.coef_)

print(lm.intercept_)
lm.score(test_x,test_y)
features_2 = features.drop('newspaper', axis = 1)

train_x,test_x, train_y,test_y = train_test_split(features_2, label, test_size = 0.2, random_state = 42)

print(train_x.shape)

print(test_x.shape)

print(train_y.shape)

print(test_y.shape)
lm.fit(train_x,train_y)
print(lm.score(train_x,train_y))

print(lm.coef_)

print(lm.intercept_)
lm.score(test_x,test_y)