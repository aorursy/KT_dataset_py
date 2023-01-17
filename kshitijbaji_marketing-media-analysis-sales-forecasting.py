import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import numpy as np

from sklearn import metrics

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

%matplotlib inline
mediamix = pd.read_csv("../input/mediamix_sales.csv")

mediamix.tail()
mediamix.describe()
sns.jointplot(x='tv_sponsorships',y='sales',kind='hex', data=mediamix)  #visualisation to observe relationships between dependent and independent variables
sns.pairplot(mediamix)
#correlation matrix

corr_media = mediamix.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corr_media, vmax=.8, square=True);
#visualise Sales when from search when

sns.set_palette("GnBu_d")

sns.set_style('whitegrid')

sns.jointplot(x='radio',y='sales',data=mediamix)  #ads are run on radio

sns.jointplot(x='Search',y='sales',data=mediamix)  #ads are run on search engines

sns.jointplot(x='tv_sponsorships',y='sales',data=mediamix)   #sponsored ads are run on tv

sns.jointplot(x='tv_cricket',y='sales',data=mediamix)  #sponsored ads are run on tv during cricket telecast

sns.jointplot(x='Display_Rest',y='sales',data=mediamix)   #ads are just on stationary display
#Creating a Linear Model Plot for sales from radio ads

sns.lmplot(x='radio', y='sales', data=mediamix)

sns.lmplot(x='Search', y='sales', data=mediamix)

sns.lmplot(x='tv_sponsorships', y='sales', data=mediamix)

sns.lmplot(x='tv_cricket', y='sales', data=mediamix)

sns.lmplot(x='Display_Rest', y='sales', data=mediamix)
#Split dataset and train

y=mediamix['sales']

X=mediamix[['tv_sponsorships', 'tv_cricket', 'radio', 'Social', 'Display_Rest', 'tv_RON', 'Programmatic', 'Magazines', 'Search']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
mm_model = LinearRegression()

mm_model.fit(X_train,y_train)
print('Coefficients: \n', mm_model.coef_)
sales_forecast = mm_model.predict(X_test)

plt.scatter(y_test,sales_forecast)

plt.xlabel('Y test')

plt.ylabel('Predicted Y')
print ('MAE :', metrics.mean_absolute_error(y_test, sales_forecast))

print ('MSE :', metrics.mean_squared_error(y_test, sales_forecast))

print ('RMSE :', np.sqrt(metrics.mean_squared_error(y_test, sales_forecast)))