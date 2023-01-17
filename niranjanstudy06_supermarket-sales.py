import pandas as pd 

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
sales_data = pd.read_csv('../input/supermarket-sales/supermarket_sales - Sheet1.csv')

plt.figure(figsize = (15,6))

sales_data = sales_data.drop(['Invoice ID'], axis = 1)

sales_data.head()
sales_product = pd.get_dummies(sales_data['Product line']) 

plt.figure(figsize = (50,10))

sns.jointplot(x = sales_data['Unit price'] , y = sales_data['Rating'] , kind = 'kde') 
sns.swarmplot(x = sales_data['Branch'], y = sales_data['Rating']).set_title('Branch wise Ratings')
sns.set(style='ticks')

plt.figure(figsize = (15,6))

Total_males_females = sns.relplot(x = 'Unit price', y='Quantity' ,kind = 'line', hue = 'Gender' , style = 'Gender',col = 'Product line', row = 'Branch' , sizes = (20,100), data = sales_data )
pd.DataFrame(sales_data.Payment.value_counts())
sales_data.rename(columns={'Product line':'Product_line'} , inplace=True)
plt.figure(figsize = (10,7));sales_data.Payment.value_counts().plot(kind = "pie" );plt.title("Payment Method")

plt.figure(figsize = (10,7));sales_data.Product_line.value_counts().plot(kind = "bar" );plt.title("Product line")
sns.lmplot(x = 'Rating' , y = 'Total' , data = sales_data , hue = 'Gender' , height = 7 , aspect = 1.5)
x = list(sales_data.columns)

x
y = sales_data['Rating']

features = ['Product_line', 'Unit price','Quantity','Tax 5%','Customer type','City']

X = sales_data[features]

train_X,val_X,train_y,val_y = train_test_split(X,y,random_state = 1)

train_X= pd.get_dummies(train_X)

train_y= pd.get_dummies(train_y)

val_X= pd.get_dummies(val_X)

val_y= pd.get_dummies(val_y)

train_X.head()
leaf_model = RandomForestRegressor(random_state = 1)

leaf_model.fit(train_X,train_y)

predict_value = leaf_model.predict(val_X)

predict_value

mean_error = mean_absolute_error(val_y,predict_value)

print('The Mean Absolute Error is : {}'.format(mean_error))