# Importing Libraries

import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

from sklearn.linear_model import LinearRegression 

from sklearn.preprocessing import PolynomialFeatures 
# Read the dataset - bottle.csv 

# Here, I've made Btl_cnt column as an Index 



data = pd.read_csv("../input/bottle.csv", index_col = "Btl_Cnt")

data.head(3)
# Change Index name by Serial_no, # Inplace true makes it permanent further



data.index.set_names(["Serial_no"], inplace = True)
# Extract two columns(Salnity & T_degC) from dataframe for prediction 



dataset = data[["Salnty","T_degC"]]

dataset.head(1)
# change the name of the columns 



dataset.columns = ["Sal", "Temp"]
dataset.head(1)
#dropdown null values everywhere in dataset

dataset = dataset.dropna(axis=0, how="any")
# take sample size of 500 to speed up the analysis

Trained_data = dataset[:][:500]

len(Trained_data)
#checkout of NaN existance in Sal column of Trained_data

Trained_data["Sal"].isna().value_counts()
#checkout of NaN existance in Temp column of Trained_data

Trained_data["Temp"].isna().value_counts()
#Dropdown duplicates values in Trained_data

Trained_data = Trained_data.drop_duplicates(subset = ["Sal", "Temp"])

len(Trained_data)
import seaborn as sns

sns.set(font_scale=1.6)

plt.figure(figsize=(13, 9))

plt.scatter(Trained_data["Sal"], Trained_data["Temp"],s=65)

plt.xlabel('Sal',fontsize=25)

plt.ylabel('Temp',fontsize=25)

plt.title('Trained_data  - Sal vs Temp',fontsize=25)

plt.show()
# Divide Trained_data into two variables X & y

X = Trained_data.iloc[:, 0:1].values  # all rows of Sal column

y = Trained_data.iloc[:, -1].values  # all rows of Temp column
lin = LinearRegression()

lin.fit(X,y)
#Predict value of Temp with random variable

Prediction_Temp_lin = lin.predict([[33]])

Prediction_Temp_lin
import seaborn as sns

sns.set(font_scale=1.6)

plt.figure(figsize=(13, 9))

plt.scatter(X,y,s=65)

plt.plot(X,lin.predict(X), color='red', linewidth='6')

plt.xlabel('Sal',fontsize=25)

plt.ylabel('Temp',fontsize=25)

plt.title('Comparision Temp and Predicted Temp with Linear Regression',fontsize=25)

plt.show()
# Consider degree=3 

poly = PolynomialFeatures(degree = 3) 

X_poly = poly.fit_transform(X) 

poly.fit(X_poly, y) 

lin2 = LinearRegression() 

lin2.fit(X_poly, y)
#Predict value of Temp randomly

Prediction_Temp_Poly = lin2.predict(poly.fit_transform([[33]])) 

Prediction_Temp_Poly
sns.set(font_scale=1.6)

plt.figure(figsize=(13, 9))

x_grid = np.arange(min(X), max(X), 0.1)

x_grid = x_grid.reshape(-1,1)

plt.scatter(X,y,s=65)

plt.plot(x_grid,lin2.predict(poly.fit_transform(x_grid)) , color='red', linewidth = '6')

plt.xlabel('Sal',fontsize=25)

plt.ylabel('Temp',fontsize=25)

plt.title('Comparision Temp and Predicted Temp with Linear Regression',fontsize=25)

plt.show()
Test_data = data[["Salnty","T_degC"]]

Test_data.head(2)
Test_data["Salnty"].isna().value_counts()
Test_data.dropna(subset = ["Salnty"], inplace = True)
Test_data["Salnty"].isna().value_counts()
Test_data["T_degC"].isna().value_counts()
NaN_Temp = Test_data[Test_data["T_degC"].isna()]

NaN_Temp
def NaN_Temp_Prediction(row):

    Salnty = row[0]

    return lin2.predict(poly.fit_transform([[Salnty]]))
NaN_Temp.iloc[:3262,-1] = NaN_Temp.iloc[:3262,].apply(NaN_Temp_Prediction, axis= 1)
# Here, we can see that all the values have been replaced by predicted values in T_degC column

NaN_Temp

# Values in the T_degC column in the form of an array. Just remove the brackets using .str.strip() method

NaN_Temp["T_degC"] = NaN_Temp["T_degC"].str.get(0)
# Here, is our cleaned dataframe

NaN_Temp