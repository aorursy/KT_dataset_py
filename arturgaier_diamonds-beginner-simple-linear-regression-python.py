# the model I will use is Linear Regression (Method : minimum sum of the squared deviations), with R^2 as Model-Fitness. 

# Linear Regression is usefull for predicting Prices, Weahter ect.
# import your relevant libraries

import pandas as pd

import numpy as np
df= pd.read_csv("../input/diamonds/diamonds.csv", index_col= 0) #erase the index so shuffling the data later will be possible

df.head()

# lets take a quick look on my dataframe
df.info()

# never hurts to have a quick overview of the datatypes which we are handling
# as we can see, there are categorical feartures in this dataset, lets find out what the contain:

df["cut"].unique()
df["clarity"].unique()
df["color"].unique()
# fortunately, the data desciption on kaggle gives us an insight of what these values acutally mean

# cut quality of the cut (Fair, Good, Very Good, Premium, Ideal)

# clarity a measurement of how clear the diamond is (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))

# color diamond colour, from J (worst) to D (best)
#in this machine learning tutorial we want to make linear regressions which need arbitrary values->

#so we make a dicionary!

cut_dict = {"Fair":1, "Good":2, "Very Good":3, "Premium":4, "Ideal":5}

color_dict = {'E':6, 'I':2, 'J':1, 'H':3, 'F':5, 'G':4, 'D':7}

clarity_dict = {"I3":1, "I2": 2, "I1":3, "SI2":4, "SI1":5, "VS2":6, "VS1":7, "VVS2":8, "VVS1":9, "IF":10, "FL":11}



#than we transfer our dictionaries into our dataframe

df["cut"] = df["cut"].map(cut_dict)

df["color"] = df["color"].map(color_dict)

df["clarity"] = df["clarity"].map(clarity_dict)

#ofcourse you must keep in mind that this will not be 100% correct as I have no real idea that for example..

#.. "Good" is twice the value of "Fair"
#did it work?

df.head()
# the columns x,y and z are also described on kaggle

df.rename(columns = {"depth":"DepthPc","x":"lenght", "y":"widht", "z":"depth"}, inplace = True)

df.head()
df.corr(method = "pearson")["price"].sort_values(ascending = False)

# lets look out for highly correlated features to our target "price"
#we see that we have a strong linear relationship between the target and the following 4 features
import matplotlib.pyplot as plt

import seaborn as sns

plt.figure(figsize = (12,8))

sns.heatmap(df.corr(method = "pearson"), cmap="Greens")

# a heatmap is a brilliant figure to visulize multicollinearity, lets keep carat, lenght, width and depth in mind
# in this first run we won´t eliminate these highly correlated features 

# I make a full run with all features

y = df["price"]

x = df.drop(columns = {"price"})



import sklearn 

df = sklearn.utils.shuffle(df)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(x_train, y_train)

print(model.score(x_test, y_test))
# Model-Fitness of 0.9069 is pretty hight, but let´s not forget the high correlation bewteen our four features from above

# for this, we eliminate the features "lenght", widht", "depth"

y = df["price"]

x = df.drop(columns = {"price", "lenght", "widht", "depth"})

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

model2 = LinearRegression()

model2.fit(x_train, y_train)

print(model2.score(x_test, y_test))



# Note, that R-squared will always increase as you add more features to the model, even if they are unrelated to the target

# Selecting the model with the highest R-squared is not a reliable approach for choosing the best linear model
# the accuracy above is pretty good! So why did I eliminate these features?

# with the dimensions of each diamond by the features "lenght","widht" and "depth"..

#..and under the asumption that the density of diamonds is stabil..

# I asume that "lenght","widht" and "depth" has little more information as if we alrady use "carat" 
# if we want a more precise model we can try an eliminate outliers in our target variable (another aproach might be normalizing with functions which are resilient to those)

import matplotlib.pyplot as plt



plt.figure(figsize = (12,8))

ax = sns.boxplot(y = train["SalePrice"], data = train)

plt.show()
# approach to eleminate outliers:

upper_quartile = np.percentile(df["price"], 75)

lower_quartile = np.percentile(df["price"], 25)

iqr = upper_quartile - lower_quartile

upper_whisker = (upper_quartile + 1.5*iqr)

lower_whisker = (lower_quartile - 1.5*iqr)

df = df[(lower_whisker < df["price"]) & (df["price"] < upper_whisker)]

df.info()



y = df["price"]

x = df.drop(columns = {"price", "lenght", "widht", "depth"})

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle=False)

model3 = LinearRegression()

model3.fit(x_train, y_train)

print(model3.score(x_test, y_test))
# a slightly lesser coefficient of determination is a bit suprising

# Maybe I destroyed valuable information with erasing outliers

price_predict = model3.predict(x_test)
# lets compare our predictions for price with the actual value

# the overall result is quite good BUT..

# .. a closer look reveals that our model sometimes predicted a negative price which is of course not reasonable

result = pd.DataFrame({"prediction": price_predict.flatten(), "actual_price": y_test})

result
# thanks for reading, I hope you enjoyed it

# if you have annotations or suggestions on improving my work I would gladly hear your opinion!