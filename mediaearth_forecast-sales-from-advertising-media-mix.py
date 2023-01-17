#This program attempts to visualize media planning spend data by media channel against sales data

# After visualizing variable pairs, we try to fit a linear regression model for the dataset

# After fitting the LR model, we predict sales based on past media spend patterns

# we estimate the accuracy of the prediction

# we have to use a few external python libraries to complete the prediction

# for visualisation, both pandas and seaborn were used

# This dataset has been constructed from scratch as in the real world, it is difficult to assemble channel level sales data



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

# This kernel covers starter concepts of Linear Regression Machine Learning

# Inspiration for this kernel is from Pierian Data
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
mediamix = pd.read_csv("../input/mediamix_sales.csv")

mediamix.tail()
mediamix.describe()
sns.set_palette("GnBu_d")

sns.set_style('whitegrid')

sns.jointplot(x='tv_cricket',y='sales',data=mediamix)

# trying to visualise Sales when Cricket ads were shown on TV channels
sns.jointplot(x='tv_RON',y='sales',data=mediamix)
sns.jointplot(x='tv_sponsorships',y='sales',data=mediamix)
sns.jointplot(x='tv_sponsorships',y='sales',kind='hex', data=mediamix)

# change visualisation to observe relationships between dependent and independent variables
sns.pairplot(mediamix)

#explore facetgrid in seaborn documenteantation for customising pairplot visualisations
#correlation matrix

corr_media = mediamix.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corr_media, vmax=.8, square=True);
sns.lmplot(x='radio', y='sales', data=mediamix)
y=mediamix['sales']

X=mediamix[['tv_RON', 'tv_sponsorships', 'tv_cricket','radio', 'NPP','Magazines','OOH', 'Social', 'Display_Rest', 

            'Search', 'Programmatic']]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.linear_model import LinearRegression
mm_model = LinearRegression()
mm_model.fit(X_train,y_train)
# Each variable in the dataset weights the predictions diferently

# these weights are referred to as coefficients

print('Coefficients: \n', mm_model.coef_)
sales_forecast = mm_model.predict(X_test)
plt.scatter(y_test,sales_forecast)

plt.xlabel('Y test')

plt.ylabel('Predicted Y')
from sklearn import metrics

print ('MAE :', metrics.mean_absolute_error(y_test, sales_forecast))

print ('MSE :', metrics.mean_squared_error(y_test, sales_forecast))

print ('RMSE :', np.sqrt(metrics.mean_squared_error(y_test, sales_forecast)))
sns.distplot(sales_forecast, bins=50)