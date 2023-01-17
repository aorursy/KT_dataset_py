# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



#importing libraries

import pandas as pd

import sklearn as sk



import numpy as np

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression



import seaborn as sns

from IPython.display import display

import matplotlib.pyplot as plt  # Matlab-style plotting



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#load the data into a Pandas dataframe

data = pd.read_excel('../input/vegetable-and-fruits-price-in-india/Vegetable and Fruits Prices  in India.xlsx')



print("Training Data")

display(data)



#look for duplicate data, invalid data (e.g. salaries <=0), or corrupt data and remove it

data.duplicated().sum()



def isDataMissing(DataToCheck):

#missing data

    total = DataToCheck.isnull().sum().sort_values(ascending=False)

    percent = (DataToCheck.isnull().sum()/DataToCheck.isnull().count()).sort_values(ascending=False)

    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    display(missing_data.head(20))


print("Check for Missing Data in Training Set")

isDataMissing(data)
#Since Datesk is id field remove it

data.drop(columns =['datesk'],inplace=True)

print("Remove rows where item name is blank")

#Remove rows where item name is blank 

data = data[~data['Item Name'].isnull()]

display(data)

#Extract year from date column

data['Date'] = pd.to_datetime(data['Date'])

data['year'] = data['Date'].dt.year

data.drop(columns='Date',inplace=True)


MeanPrices = data.groupby(['Item Name','year']).mean()

print("Mean Prices for all the items on yearwise basis")

display(MeanPrices)

#Save records where price is null or 0 for the items in null prices dataframe

print("Null Prices Dataset containing null and 0 values for price")

nullPrices = data[(data['price'].isnull()) | (data['price']== 0)]

display(nullPrices)
#Remove rows where price is null or price is 0 as we will assign mean values for them later



data = data[~((data['price'].isnull()) | (data['price']== 0))]

print("Removing null and 0 prices from the training dataset for feature engineering")

display(data)
print("Assigning Mean Price to training set data where price is 0 or Nan")

nullPrices = pd.merge(nullPrices,MeanPrices,left_on=['Item Name','year'],right_on=['Item Name','year'])

display(nullPrices.drop(columns=['price_x']))



df = pd.DataFrame({"Item Name":nullPrices['Item Name'], 

                    "year":nullPrices['year'],"price":nullPrices['price_y']}) 

train_data = data.append(df)

display("Combine dataset after feature engineering")

display(train_data)
#price cannot be zero so filter out data where price is around zero

train_data = train_data[(train_data['price']> 1)]

backup = train_data

print("Backup")

display(backup)
train_data = pd.get_dummies(train_data).reset_index(drop=True)

sns.distplot(train_data['price']);



print("Skewness: " + str(train_data['price'].skew()))



#Histogram is not normally distributed

#sns.distplot(train_data['price']);

#applying log transformation

train_data['price'] = np.log(train_data['price'])

sns.distplot(train_data['price']);
train_output = train_data['price']

#Remove target output column Price from training dataset

train_data.drop(columns='price',inplace=True)

print("Training Dataset")

display(train_data)



X_train, X_test, y_train, y_test = train_test_split(train_data, train_output, test_size = 0.1, random_state = 0)

regressor = LinearRegression()

regressor.fit(X_train, y_train)



y_pred = regressor.predict(X_test)

mean_squared_error(y_pred,y_test)
#Preparing Test Data for 2019 Price predictions

test_data = pd.DataFrame(columns = [ 'year','Item Name'])

#Since there are 302 unique items predicting prices for these items in 2019 year

for item in backup['Item Name'].unique():

    test_data = test_data.append({'Item Name': str(item), 'year':2019},ignore_index=True)



display(test_data)
test_data_one_hot = pd.get_dummies(test_data).reset_index(drop=True)



test_data_one_hot.rename(columns ={'year_2019':'year'},inplace=True)

test_data_one_hot['year'] = 2019



print("Testing Data")

display(test_data_one_hot)



regressor.fit(train_data, train_output)



predictions = regressor.predict(test_data_one_hot)

#print(predictions)



preds = pd.DataFrame()

#convert the logarithmic values to normal form

preds['price'] =np.exp(predictions)

preds['Item Name'] = test_data['Item Name']

preds['year'] = test_data['year']

print("Average price predictions for 2019 year ")

display(preds)

#Saving predictions in output.csv file

preds.to_csv('output.csv')