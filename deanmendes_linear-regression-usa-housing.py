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
import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score
df = pd.read_csv('/kaggle/input/USA_Housing.csv')

df.head(2)
# drop 'Address' since it is of no interest to regression analysis

df.drop(columns=['Address'], inplace=True)



# change column/features names for easier manipulation and access

df.rename(columns={'Avg. Area Income':'Income', 

                   'Avg. Area House Age':'House Age', 

                   'Avg. Area Number of Rooms': 'Avg Room_area', 

                   'Avg. Area Number of Bedrooms':'Avg Bedroom_area'}, inplace=True)

df.head(2)
df.info()
# see correlation of features to the target variable 'Price'

df.corr()['Price'].sort_values()
# assessing to see if normalization is necessary

fig, ax1 = plt.subplots(figsize=(6,5))

ax1.set_title('Before Min-Max')

sns.kdeplot(df['Income'], ax=ax1)

sns.kdeplot(df['Area Population'], ax=ax1)

plt.show()
fig, ax2 = plt.subplots(figsize=(6,5))

ax2.set_title('Before Min-Max')

sns.kdeplot(df['House Age'], ax=ax2)

sns.kdeplot(df['Avg Room_area'], ax=ax2)

plt.show()
# normalize x_data using the min max method

X = df[['Income', 'House Age', 'Avg Room_area', 'Avg Bedroom_area', 'Area Population']]

Y  = df['Price']



for x in X:

    df[x] = (df[x] - min(df[x])) / (max(df[x]) - min(df[x]))

df.head(3)
# visualise the effect of normalization

fig, ax3 = plt.subplots(figsize=(6,5))

ax3.set_title('After Min-Max')

sns.kdeplot(df['Income'], ax=ax3)

sns.kdeplot(df['Area Population'], ax=ax3)

sns.kdeplot(df['House Age'], ax=ax3)

sns.kdeplot(df['Avg Room_area'], ax=ax3)

plt.show()
# using seaborn residual plot too identify relationship

sns.residplot(df['Income'], df['Price'])

plt.title('Income and Price')

plt.show()
sns.residplot(df['House Age'], df['Price'])

plt.title('House Age and Price')

plt.show()
sns.residplot(df['Avg Room_area'], df['Price'])

plt.title('Avg Room_area and Price')

plt.show()
sns.residplot(df['Area Population'], df['Price'])

plt.title('Area Population and Price')

plt.show()
# training and testing data

x_data = df[['Income', 'House Age', 'Avg Room_area', 'Area Population']].values

y_data  = df['Price'].values



x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=0)



print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# linear model object

lm = LinearRegression()



# fit the model

lm.fit(x_train, y_train)



# estimate 

yhat = lm.predict(x_test)
# dataframe of testing y values and predicted values

df_new = pd.DataFrame({'Actual': y_test, 'Predicted': yhat})

df_new.head()
# use metrics to determine the fit of the model

# Mean Squared Error 

MSE = mean_squared_error(y_test, yhat)

print('Mean Squared Error: ', MSE)



# root mean squared error, useful it shows the error in terms of y units

RMSE = np.sqrt(MSE)

print('RMSE: ', RMSE)



# r-squared 

R2 = r2_score(y_test, yhat)

print('R2 score: ', R2)

R2_perc = round(R2 * 100, 2)