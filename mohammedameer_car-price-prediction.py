import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import linear_model

from sklearn.linear_model import LinearRegression



# reading the dataset

cars = pd.read_csv("/kaggle/input/car-data/CarPrice_Assignment.csv")

# summary of the dataset: 205 rows, 26 columns, no null values

print(cars.info())
# head

cars.head()
# target variable: price of car

sns.distplot(cars['price'])

plt.show()
# all numeric (float and int) variables in the dataset

cars_numeric = cars.select_dtypes(include=['float64', 'int'])

cars_numeric.head()
# dropping symboling and car_ID 

cars_numeric = cars_numeric.drop(['symboling', 'car_ID'], axis=1)

cars_numeric.head()
# correlation matrix

cor = cars_numeric.corr()

cor
# plotting correlations on a heatmap



# figure size

plt.figure(figsize=(16,8))



# heatmap

sns.heatmap(cor, cmap="YlGnBu", annot=True)

plt.show()
cars_numeric
x = cars_numeric.iloc[:,0:13].values

y = cars_numeric.iloc[:,13].values

y
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0)

x_train.shape
x_test.shape
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)

x_test = scaler.transform(x_test)
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(x_train,y_train)

y_pred = model.predict(x_test)

y_pred
y_test
c = model.intercept_

c
m = model.coef_

m
model.predict([x_train[10]])
y_train[10]
plt.scatter(y_pred,y_test)
df1 = pd.DataFrame({'Actual':y_test,'Predicted':y_pred})

df1
import matplotlib.pyplot as plt

df2 = df1

df2.plot(figsize=(20,8),kind='bar')

plt.show()
plt.figure(figsize=(15,5))



plt.subplot(1,2,1)

sns.scatterplot(x="Actual", y="Predicted", data=df1)



plt.subplot(1,2,2)

sns.regplot(x="Actual", y="Predicted", data=df1)
# metrics

from sklearn.metrics import r2_score,mean_squared_error

# Mean Squared Error 

print(mean_squared_error(y_test,y_pred))

print(r2_score(y_true=y_test, y_pred=y_pred))
import numpy as np

MSE = np.square(np.subtract(y_test,y_pred)).mean() 

MSE
# Error terms

c = [i for i in range(len(y_pred))]

fig = plt.figure()

plt.plot(c,y_test-y_pred, color="blue", linewidth=2.5, linestyle="-")

fig.suptitle('Error Terms', fontsize=20)              # Plot heading 

plt.xlabel('Index', fontsize=18)                      # X-label

plt.ylabel('ytest-ypred', fontsize=16)                # Y-label

plt.show()
cars
# converting symboling to categorical

cars['symboling'] = cars['symboling'].astype('object')
# CarName: first few entries

cars['CarName'][:30]
# Extracting carname



# Method 1: str.split() by space

cars['CarName'] = cars['CarName'].apply(lambda x: x.split(" ")[0])

cars['CarName'][:30]
# look at all values 

cars['CarName'].value_counts()


# volkswagen

cars.loc[(cars['CarName'] == "vw") | 

         (cars['CarName'] == "vokswagen")

         , 'CarName'] = 'volkswagen'



# porsche

cars.loc[cars['CarName'] == "porcshce", 'CarName'] = 'porsche'



# toyota

cars.loc[cars['CarName'] == "toyouta", 'CarName'] = 'toyota'



# nissan

cars.loc[cars['CarName'] == "Nissan", 'CarName'] = 'nissan'



# mazda

cars.loc[cars['CarName'] == "maxda", 'CarName'] = 'mazda'
y = cars['price']

x = cars.drop(['price'], axis = 1)
# creating dummy variables for categorical variables



# subset all categorical variables

cars_categorical = x.select_dtypes(include=['object'])

cars_categorical.head()
# convert into dummies

cars_dummies = pd.get_dummies(cars_categorical)

cars_dummies.head()
x = cars_numeric.iloc[:,0:13]

y = cars_numeric.iloc[:,13].values
# concat dummy variables with X

x = pd.concat([x, cars_dummies], axis=1)
x = x.iloc[:,:].values

x
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0)

x_train.shape


scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)

#.fit_transform first fits the original data and then transforms it

x_test = scaler.transform(x_test)
y_train
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(x_train,y_train)

y_pred = model.predict(x_test)

y_pred
y_test
# metrics

from sklearn.metrics import r2_score,mean_squared_error

# Mean Squared Error 

print(mean_squared_error(y_test,y_pred))

print(r2_score(y_true=y_test, y_pred=y_pred))