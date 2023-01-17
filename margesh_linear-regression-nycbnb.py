%matplotlib inline

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

df = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

df.head()
df.isnull().sum()
df.describe()
not_renting_out = np.array(df['availability_365']==0).sum()

print(not_renting_out)

print(not_renting_out/len(df['price'])*100)
print(df.shape)

df = df[df['availability_365']!=0]

print(df.shape) #New DF shape
df.dtypes
print("Room types - ", len(df['room_type'].value_counts()))

print("Neighbourhood groups - ", len(df['neighbourhood_group'].value_counts()))

print("Neighbourhoods - ", len(df['neighbourhood'].value_counts()))
f, axs = plt.subplots(1,2, figsize=(12,5))

#In decscending order by frequency/counts - 

sns.countplot(df['room_type'], order=df['room_type'].value_counts().index,ax=axs[0])

sns.countplot(df['neighbourhood_group'], order=df['neighbourhood_group'].value_counts().index, ax=axs[1])
df['room_type'] = df['room_type'].astype('category').cat.codes

df['neighbourhood_group'] = df['neighbourhood_group'].astype('category').cat.codes

f, axs = plt.subplots(1,2, figsize=(12,5))

sns.countplot(df['room_type'], ax=axs[0], order=df['room_type'].value_counts().index)

sns.countplot(df['neighbourhood_group'],order=df['neighbourhood_group'].value_counts().index, ax=axs[1])
df.head()
df.drop(['id', 'name', 'host_id', 'host_name', 'neighbourhood', 'last_review', 'reviews_per_month'], axis=1, inplace=True)
df.head()
plt.figure(figsize=(10,10))

sns.heatmap(df.corr().round(3), annot=True)
from sklearn.model_selection import train_test_split

from sklearn import preprocessing



Y = df['price']

X = df[['neighbourhood_group', 'longitude', 'room_type', 'availability_365', 'calculated_host_listings_count']]

X = preprocessing.normalize(X)

X = np.hstack((np.ones( (len(df['price']) ,1)), X))



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

print(X_train.shape)

print(X_test.shape)

print(Y_train.shape)

print(Y_test.shape)
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error,r2_score



lin_model = LinearRegression().fit(X_train, Y_train)
y_train_predict = lin_model.predict(X_train)

rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))

r2 = r2_score(Y_train, y_train_predict)



print("The model performance for training set")

print("--------------------------------------")

print('RMSE is {}'.format(rmse))

print('R2 score is {}'.format(r2*100))

print("\n")



# model evaluation for testing set

y_test_predict = lin_model.predict(X_test)

rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))

r2 = r2_score(Y_test, y_test_predict)



print("The model performance for testing set")

print("--------------------------------------")

print('RMSE is {}'.format(rmse))

print('R2 score is {}'.format(r2*100))
error_frame = pd.DataFrame({'Actual': np.array(Y_test).flatten(), 'Predicted': y_test_predict.flatten()})

error_frame.head(10)
df1 = error_frame[:50]

df1.plot(kind='bar',figsize=(24,20))

plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')

plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

plt.show()
Q1 = df.quantile(0.25)

Q3 = df.quantile(0.75)

IQR = Q3 - Q1
IQR_df = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]

IQR_df.shape
IQR_df.describe()
Y = IQR_df['price']

X = IQR_df[['neighbourhood_group', 'longitude', 'room_type', 'availability_365', 'calculated_host_listings_count']]

X = preprocessing.normalize(X)

X = np.hstack((np.ones( (len(IQR_df['price']) ,1)), X))





X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

print(X_train.shape)

print(X_test.shape)

print(Y_train.shape)

print(Y_test.shape)
lin_model = LinearRegression().fit(X_train, Y_train)
y_train_predict = lin_model.predict(X_train)

rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))

r2 = r2_score(Y_train, y_train_predict)



print("The model performance for training set")

print("--------------------------------------")

print('RMSE is {}'.format(rmse))

print('R2 score is {}'.format(r2*100))

print("\n")



# model evaluation for testing set

y_test_predict = lin_model.predict(X_test)

rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))

r2 = r2_score(Y_test, y_test_predict)



print("The model performance for testing set")

print("--------------------------------------")

print('RMSE is {}'.format(rmse))

print('R2 score is {}'.format(r2*100))
error_frame = pd.DataFrame({'Actual': np.array(Y_test).flatten(), 'Predicted': y_test_predict.flatten()})

error_frame.head(10)
df1 = error_frame[:50]

df1.plot(kind='bar',figsize=(24,20))

plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')

plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

plt.show()