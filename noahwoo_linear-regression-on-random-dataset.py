# Import packages
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt

# Show the tree of dir
# from os import listdir
# x = listdir('../input')
# print(x)
# Read datas
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
# Some Discription statistics 

## List first or last rows of data
print (train_data.head(n=10))
print (test_data.tail())

## Summariases the central Tendary
print (train_data.describe())
print (test_data.describe())
## Information of a dataFrame
print (train_data.info())
print (test_data.info())
## Shape of dataframe
print (train_data.shape)
print (test_data.shape)
## Dropping of the missing data
train_data = train_data.dropna()
test_data= test_data.dropna()
## Print the shape of data after dropping
print (train_data.shape)
print (test_data.shape)
## Visualising Train_data and Test_data
sns.jointplot(x='x', y='y', data=train_data)
sns.jointplot(x='x', y='y', data=test_data)
## Creation of Linear Model Object
lm = LinearRegression()

train_xx = pd.DataFrame(train_data.x)
train_yx = pd.DataFrame(train_data.y)
## why do not use train_data.x direct

# x = pd.DataFrame(train_data.iloc[:,0].values)
# y = pd.DataFrame(train_data.x)
# diff = np.where(x != y)
# print(x.equals(y))
# for i in range(pd.DataFrame(train_data.x).shape[0]):
#     if pd.DataFrame(train_data.x).iloc[i].values != pd.DataFrame(train_data.iloc[:,0].values).iloc[i].values:
#         print(i)
# plt.plot(pd.DataFrame(train_data.x),pd.DataFrame(train_data.iloc[:,0].values),'.')
## have no ideas why print(x.equals(y))  output False

train_x = pd.DataFrame(train_data.iloc[:,0].values)
train_y = pd.DataFrame(train_data.iloc[:,1].values)


test_x = pd.DataFrame(test_data.iloc[:,0].values)
test_y = pd.DataFrame(test_data.iloc[:,1].values)
plt.plot(train_data.x, train_data.y,'.')
plt.figure()
#plt.scatter(train_x, train_y)
## Trainning the Model by training dataset
lm.fit(train_x, train_y)

## Prints the Accuracy of Model
accurary = round(lm.score(train_x, train_y)*100, 2)
print('Accurary:', accurary)

## Prints the Coefficients
print('Coefficients:',lm.coef_)

## Visualising the Trainning Dataset
plt.figure(figsize=(12,6))
plt.scatter(train_x, train_y)
#plt.plot(train_x, lm.predict(train_x), color='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Training Data')
## Visualising the Test Dataset
plt.figure(figsize=(12,6))
plt.scatter(test_x, test_y)
plt.plot(test_x, lm.predict(test_x), color='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Test Data')
#Real Test Values Versus Predicted Test Values
predictions=lm.predict(test_x)
plt.scatter(test_y,predictions)
plt.xlabel('Y Values')
plt.ylabel('Predicted Values')
plt.title('R_values VS P_values')


sns.distplot(test_y-predictions)