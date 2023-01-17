# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#importing the data sets



train_data = pd.read_csv(r'/kaggle/input/into-the-future/train.csv')

test_data = pd.read_csv(r'/kaggle/input/into-the-future/test.csv')
id1 = test_data['id']

time1= test_data['time']

data = {'id': id1, 'time':time1 , 'feature_1':test_data['feature_1']}

final_df = pd.DataFrame(data)
#exploring the data sets

train_data.head()
train_data.info()
train_data.describe()
#data preprocessing

#dropping the id column

train_data.drop(['id'],axis=1,inplace=True)
train_data.head()
#now we can just remove '2019-03-19 ' from the time attribute because it appears in every entry

for i in range (0,564):

    train_data['time'][i]=train_data['time'][i].replace('2019-03-19 ','')
train_data.head()
#now visualizing the general trends in data 

import matplotlib.pyplot as plt

x = train_data['time']   #time attribute

y = train_data['feature_1']   #feature_1

z = train_data['feature_2']   #feature_2
#plotting time vs feature_1

plt.plot(x,y)

plt.show()
#plotting time vs feature_2

plt.plot(x,z)

plt.show()
#plot to visualize the dependencies of feature_1 and feature_2

plt.plot(y,z)

plt.show()
#we can see that the first value in feature_2 is erroneous

#hence, we can drop the first row

train_data = train_data[train_data['time']!='00:00:00']
train_data.head()
#let's visualize the plot for feature_2 again

x = train_data['time']

z = train_data['feature_2']

plt.plot(x,z)

plt.show()
#evaluating the correlation between the attributes 

corr = train_data.corr()

corr
# hence, we can see that both the features show high negative correlation

# using automatic time series decompostion
from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(train_data['feature_2'], model='multiplicative',period = 1)

result.plot()

plt.show()
result1 = seasonal_decompose(train_data['feature_2'], model='addtive',period = 1)

result1.plot()

plt.show()
# from above we can see, that the data shows no seasonal residual features and the trend is the same as 

# the observed data

# the curve goes up

# hence we can convert time to seconds
df = train_data.copy()
for i in range (1,564):

    df['time'][i]=df['time'][i].split(':')
df['time'][0:10]
for i in range (1,564):

    df['time'][i]=int(df['time'][i][2]) + int(df['time'][i][1])*60 + int(df['time'][i][0])*3600
df['time'][0:10]
from sklearn.model_selection import train_test_split

X = df[['time','feature_1']]

Y = df['feature_2']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
# training a linear model

from sklearn.linear_model import LinearRegression

reg = LinearRegression().fit(X_train, y_train)

yhat = reg.predict(X_test)

yhat
from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score

import math

score = mean_squared_error(y_test, yhat)

print(math.sqrt(score))

print(r2_score(y_test, yhat))
from sklearn.preprocessing import PolynomialFeatures

poly_features = PolynomialFeatures(degree=2)

X_train_poly = poly_features.fit_transform(X_train)

poly_model = LinearRegression()

poly_model.fit(X_train_poly, y_train)

# predicting on training data-set

y_train_predicted = poly_model.predict(X_train_poly)

# predicting on test data-set

y_test_predict = poly_model.predict(poly_features.fit_transform(X_test))

y_test_predict
# evaluating the model on training dataset

rmse_train = np.sqrt(mean_squared_error(y_test, y_test_predict))
rmse_train
from sklearn.tree import DecisionTreeRegressor  

  

# create a regressor object 

regressor = DecisionTreeRegressor(random_state = 0)  

  

# fit the regressor with X and Y data 

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

y_pred
score = mean_squared_error(y_test, y_pred)

print(math.sqrt(score))

print(r2_score(y_test, y_pred))
test_data.info()
for i in range(0,374):

    test_data['time'][i]=test_data['time'][i].replace('2019-03-19 ','')

test_data['time'][374]=test_data['time'][374].replace('2019-03-19 ','')
for i in range (0,375):

    test_data['time'][i]=test_data['time'][i].split(':')
for i in range (0,375):

    test_data['time'][i]=int(test_data['time'][i][2]) + int(test_data['time'][i][1])*60 + int(test_data['time'][i][0])*3600
test_data.drop(['id'],axis=1,inplace=True)
test_data.head()
final_y = poly_model.predict(poly_features.fit_transform(test_data))

final_y
final_df['feature_2']=final_y

final_df.drop(['time','feature_1'],axis=1,inplace=True)
final_df.head()
mycsvfile = final_df
mycsvfile.to_csv("/kaggle/working/mycsvfile.csv",index=False)