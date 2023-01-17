# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/gld_price_data.csv')
#Lets have a quick look of dataset

df.info()
#Clearly we see there is no null value in the dataset

#Lets study the Statistical Inferance of the dataset

df.describe()
#Now see the correlation matrix and heatmap

import matplotlib.pyplot as plt

import seaborn as sns

corr = df.corr()

plt.figure(figsize = (6,5))

sns.heatmap(corr,

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values,

            annot=True,fmt='.2f',linewidths=0.30)

plt.title('Correlation of df Features', y = 1.05, size=15)
#Lets look the correlation score

print (corr['GLD'].sort_values(ascending=False), '\n')
#Lets Check our target variable

sns.distplot(df['GLD'], color = 'blue')

print('Skewness: %f', df['GLD'].skew())

print("Kurtosis: %f" % df['GLD'].kurt())
#Now we check the relation with GLD variable

sns.jointplot(x =df['SLV'], y = df['GLD'], color = 'deeppink')
#Now we check the relation with GLD variable

sns.jointplot(x =df['SPX'], y = df['GLD'], color = 'purple')
#Now Lets create a ml model

# Now lets take our matrix of feature and target

x_trail = df[['SPX','USO','SLV','EUR/USD']]

x = x_trail.iloc[:, :].values

y = df.iloc[:, 2].values
#Spliting the dataset into training and test set

from sklearn.model_selection import train_test_split

test_ratio = 0.05

len_train = int(len(x) * (1-test_ratio))

x_train, x_test, y_train, y_test = x[:len_train], x[len_train:], y[:len_train], y[len_train:]
#Now fitting the Random forest regression to the traning set

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)

regressor.fit(x_train, y_train)
#Now predicting the test set result

y_pred = regressor.predict(x_test)
#Now Check the error for regression

from sklearn import metrics

print('MAE :'," ", metrics.mean_absolute_error(y_test,y_pred))

print('MSE :'," ", metrics.mean_squared_error(y_test,y_pred))

print('RMAE :'," ", np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
#Now Lets Check the Training and Test set Accuracy

accuracy_train = regressor.score(x_train, y_train)

accuracy_test = regressor.score(x_test, y_test)

print(accuracy_train)

print(accuracy_test)
#Visualising the Accuracy of Predicted result

plt.plot(y_test, color = 'blue', label = 'Acutal')

plt.plot(y_pred, color = 'deeppink', label = 'Predicted')

plt.grid(0.3)

plt.title('Acutal vs Predicted')

plt.xlabel('Number of Oberservation')

plt.ylabel('GLD')

plt.legend()

plt.show()