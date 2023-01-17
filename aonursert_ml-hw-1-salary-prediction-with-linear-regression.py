import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df = pd.read_csv("../input/sf-salaries/Salaries.csv")
df.head()
# Data Cleaning
df = df[["Year", "TotalPay"]] # I want only 2 rows
df.head()
df["Experience"] = df["Year"].apply(lambda x: 2020 - x) # I don't want Year column but Experience Year column
df.head()
df.drop("Year", inplace=True, axis=1) # I don't need Year column anymore
df.head()
X = df[["Experience"]] # The feature to train on
y = df["TotalPay"] # Target Variable
# I split train and test datas
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# I use Linear Regression
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
# I train my dataset
lm.fit(X_train, y_train)
# I predict obviously
predictions = lm.predict(X_test)
# I want to see if predictions is successful or not
sns.scatterplot(y_test, predictions)
sns.distplot((y_test - predictions)) # distribution plot of residuals
# It's kinda good.
lm.predict([[9]]) # I predict salary of a person who has 5 years experience
df.groupby("Experience").mean()
# At the end it's not bad