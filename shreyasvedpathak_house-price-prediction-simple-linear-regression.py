import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
house_sales = pd.read_csv('/kaggle/input/House Price.csv')
house_sales.head()
len(house_sales)
x = house_sales[['sqft_living']]

y = house_sales[['price']]
plt.figure(figsize=(16,10))

sns.scatterplot(x=x['sqft_living'], y= y['price'], marker='x')

plt.title("Price vs Square Feet of Living")

plt.xlabel("Square Feet of Living")

plt.ylabel("Price")

plt.show()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train,y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)



print("Number of samples in Training set: ", len(X_train), "(80%)\nNumber of samples in Testing set: ", len(X_test), "(20%)")
from sklearn.linear_model import LinearRegression

sq_model = LinearRegression()



sq_model.fit(X_train,y_train)
y_pred = pd.DataFrame(sq_model.predict(X_test), columns= ['Predictions'])



from sklearn import metrics

print("Mean Absolute Error:",metrics.mean_absolute_error(y_test, y_pred).round(decimals =2))
plt.figure(figsize=(16,10))

sns.scatterplot(x=x['sqft_living'], y= y['price'],marker='x')

sns.regplot(X_test['sqft_living'], y = y_pred['Predictions'], color= 'red')

plt.title("Price vs Square Feet of Living")

plt.xlabel("Square Feet of Living")

plt.ylabel("Price")

plt.show()
area = int(input("Enter Area in sqft: "))

pred_val = sq_model.predict([[area]])                                                             #Important: Used to take input so that it can predict.

print("\nPredicted Price:", pred_val[0][0].round(decimals = 2))



#The following lines of codes basically finds the house with same area or area close to the given value and takes mean of all houses with that area

#so that we can compare how correctly we predicted the price.



result_index = house_sales['sqft_living'].sub(area).abs().idxmin()

nearest_house = house_sales.iloc[result_index]['sqft_living']

house = house_sales[house_sales['sqft_living']==nearest_house]

mean_price = house["price"].mean()

print("Mean Price for houses with", house_sales.iloc[result_index]['sqft_living'], "sqft area:", mean_price)