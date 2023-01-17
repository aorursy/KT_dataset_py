

import pandas as pd

import matplotlib.pyplot as plt



#Read data from csv file

train = pd.read_csv('Train_A102.csv')

test = pd.read_csv('Test_A102.csv')



#dimension of datasets

print(train.shape)

print(test.shape)



#Summary of datasets

print(train.describe())

print(test.describe())



#mean,mode,median

print(train.mode())

print(train.mean())

print("\n")

print(train.median())



print(test.mode())

print(test.mean())

print("\n")

print(test.median())





#number of missing values in each dataset

print(train.isnull().sum())

print(test.isnull().sum())



#dropping rows with missing values

train.dropna(how='any')

test.dropna(how='any')





# replacing missing values with mean,mode and median

train_mean = train['Item_Weight'].fillna(value=train.mean().Item_Weight)

train_mode = train['Item_Weight'].fillna(value=train.mode().Item_Weight)

train_median = train['Item_Weight'].fillna(value=train.median().Item_Weight)





#scatter plot between different variables

plt.scatter(train.Item_Visibility,train.Item_Outlet_Sales)

plt.show()



plt.scatter(train.Item_MRP,train.Item_Outlet_Sales)

plt.show()



plt.scatter(train.Item_Type,train.Item_Outlet_Sales)

plt.show()



plt.scatter(train.Outlet_Type,train.Item_Outlet_Sales)

plt.show()



#bar plot between different variables

plt.bar(train.Item_Visibility,train.Item_Outlet_Sales)

plt.show()



plt.bar(train.Item_MRP,train.Item_Outlet_Sales)

plt.show()



plt.bar(train.Item_Type,train.Item_Outlet_Sales)

plt.show()



plt.bar(train.Outlet_Type,train.Item_Outlet_Sales)

plt.show()

#Super Market Type 3 has highest sales





#deleting outliers

train.boxplot(column='Item_Visibility')

plt.show()



train[train.Item_Visibility < 0.19].boxplot(column='Item_Visibility')

plt.show()



train.boxplot(column='Item_Outlet_Sales')

plt.show()



train[train.Item_Outlet_Sales < 6200].boxplot(column='Item_Outlet_Sales')

plt.show()



#Creating Categories based on MRP



def func(x):

    if 0<x<= 70:

        return 'Low'

    elif 70 < x <= 130:

        return 'Medium'

    return 'High'

train['Price_Category'] = train['Item_MRP'].apply(func).astype('category')



print(train.Price_Category)