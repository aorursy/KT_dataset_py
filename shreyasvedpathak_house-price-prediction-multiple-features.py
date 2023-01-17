import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
house_sales = pd.read_csv('../input/house-price/House Price.csv')
house_sales.head()
len(house_sales)
x = house_sales[['sqft_living','bedrooms','bathrooms', 'condition']]

y = house_sales[['price']]
plt.figure(figsize=(16,6))

sns.scatterplot(x=x['sqft_living'], y= y['price'])

plt.title("Price vs Square Feet of Living")

plt.xlabel("Square Feet of Living")

plt.ylabel("Price")

plt.show()
plt.figure(figsize=(16,6))

sns.barplot(x = x['bedrooms'],y =y['price'])

plt.title("Price vs Bedrooms")

plt.xlabel("Number of Bedrooms")

plt.ylabel("Price")

plt.show()
plt.figure(figsize=(16,6))

sns.barplot(x = x['bathrooms'],y =y['price'])

plt.title("Price vs Bathrooms")

plt.xlabel("Number of Bathrooms")

plt.ylabel("Price")

plt.show()
plt.figure(figsize=(16,6))

sns.barplot(x = x['condition'],y =y['price'])

plt.title("Price vs Condition")

plt.xlabel("Condition")

plt.ylabel("Price")

plt.show()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)



print("Number of samples in Training set: ", len(X_train), "(80%)\nNumber of samples in Testing set: ", len(X_test), "(20%)")
from sklearn.linear_model import LinearRegression

multi_model = LinearRegression(normalize=True)
multi_model.fit(X_train,y_train)
y_pred = multi_model.predict(X_test)
area = int(input("Enter Area in sqft: "))

bedr = int(input("Enter Number of Bedrooms: "))

bathr = float(input("Enter Number of Bathrooms: "))

cond = int(input("Enter Condition (on a scale of 1-5): "))



pred_val = multi_model.predict([[area,bedr,bathr,cond]])

print("\nPredicted price: ", pred_val[0][0].round(decimals = 2))