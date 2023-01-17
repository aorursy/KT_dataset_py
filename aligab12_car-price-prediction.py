import pandas as pd 

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
car_data = pd.read_csv("../input/car-prices/car_prices.csv")
car_data.head()
car_data.rename({"Unnamed: 0.1":"a"}, axis="columns", inplace=True) # Renamed and dropped the 2 unnamed columns.

car_data.drop(["a"], axis=1, inplace=True)



car_data.rename({"Unnamed: 0":"b"}, axis="columns", inplace=True) 

car_data.drop(["b"], axis=1, inplace=True)
car_data.head(2) # Confirm if the two unnamed columns were droped.
car_data.isnull().sum() # Check for null values.
# stroke has 4 null values, therefore clean by getting the mean:



car_data['stroke'].fillna(car_data['stroke'].mean(), inplace=True)
# Recheck if null values have been solved 



car_data.isnull().sum()
car_data.describe() # Statistical summary of the data
# Correlation between the variables using Heatmap.





plt.figure(figsize=(25,15))

plt.title("(fig-1) Correlation of all Variables ")

sns.heatmap(car_data.corr(), annot = True) 

plt.show()   
### Summarize/group categorical data in a column.





drive_wheels_counts=car_data["drive-wheels"].value_counts().to_frame() 
plt.figure(figsize=(10,6))

plt.title("(fig-2) Relationship between Drive-Wheel Categories and Price")

sns.barplot(x="drive-wheels", y= 'price',data=car_data)

plt.show
plt.figure(figsize=(25,15))

plt.title("(fig-3) Relationship between Makes and Price")

sns.boxplot(x= "make", y= 'price',data=car_data)

plt.show()
# 1. The prediction target in this case is PRICE



y = car_data.price
# 2. The catalysts for prediction are the FEATURES columns



car_features = ["engine-size", 'horsepower', 'city-mpg', 'highway-mpg',"bore", "stroke", "peak-rpm", "normalized-losses", 'symboling', "wheel-base", "length",'height', 'width', "curb-weight"]



X = car_data[car_features]
X.describe()
from sklearn.tree import DecisionTreeRegressor



# Define model



car_model = DecisionTreeRegressor(random_state=1)

# 2. Fit Model



car_model.fit(X, y)



# 3. Prediction





print("Making predictions for the first 5 cars:")

print(X.head(5))

print("The predictions are")

print(car_model.predict(X.head(5)))
#scoring the model



car_model.score(X,y)