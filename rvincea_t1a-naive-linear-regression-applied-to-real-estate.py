# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# machine learning
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


# Any results you write to the current directory are saved as output.
# Read in the data from csv. Note that most of the library methods have many optional arguments but lets start simple and concentrate on high level features.
data = pd.read_csv('../input/kc_house_data.csv')
#Data is a pandas DataFrame which is much like an Excel table:
#It has column headers
print(data.columns)
#It has row headers
print(data.index)
#It has data
print(data.values)
#Evaluate this code and see the results below.
# data.columns shows us an array of the column names, so now we know what data is contained in the csv
# data.index shows us the index of the rows. Don't worry about RangeIndex for now, suffice to say that the rows are described by the numbers 0 to 21612, this
#   similar to Excel except that Excel starts at 1. Also note that Python does not include the stop index, so this is 0-21612 for a total of 21613 rows
# Finally data.values shows the data. Notice the ... which shows that most of the data is suppressed for visibility. Don't worry about the raw data, there are 
#  better ways to view it (next)
#Let's view the first 5 records of the data frame
data.head()

#Pretty cool, huh? But this is just too much data for a tutorial. 
#Let's look only at a subset: price, sqft living, sqft log, yr built and # of bedrooms
#In Python, a list is defined by comma separated items inside brackets and I can create a new DataFrame with only the columns I want as follows:
columns_to_keep =['price','bedrooms','sqft_living','sqft_lot', 'yr_built'] 
data = data[columns_to_keep]

#Really, don't worry about why it is done in this way. We want to fast-track to seeing the data and plotting the data.
#Let's look at the first 5 records
data.head()
#Notice I have chosen columns that are naively associated with price. We expect the selling price of a home to correlate with # of bedrooms, with square footage
# and with the year of the house
#Let's get some information on the entire set of data, rather than just 5 rows
data.describe()
#From the below, you might see the following
# 1) There are 21613 rows of data
# 2) The most expensive home cost 7.7 million dollars!
# 3) The home with the most bedroooms has 33 bedrooms 
# 4) The average home in this data set has 2080 sq ft.
# 5) 75% of the homes had 4 bedrooms or less and were built before 1998

#Let's get back to machine learning track: We want to predict price and we hope that the data is enough to do so.
#So let's look at our naive assumption that price is corrrelated with square footage of the home
plt.scatter(data['sqft_living'], data['price'],  color='black')
plt.xlabel('Square Footage of Home')
plt.ylabel('Price')
plt.show()
#This looks promising. Certainly price is not the only feature associated with price but there is a strong correlation.
#We can see that the most expensive hoouse was not the one with the largest square footage (the house at 7.7 million has 12 thousand square footage)
#Lets look at number of bedrooms next
plt.scatter(data['bedrooms'], data['price'],  color='red')
plt.xlabel('# of Bedrooms')
plt.ylabel('Price')
plt.show()
#Interesting, note that the house with 33 bedrooms seems to be an outlier, some type of extreme home.

#Let's look at this plot again but on a log scale
plt.scatter(data['bedrooms'], data['price'],  color='red')
plt.xlabel('# of Bedrooms')
plt.ylabel('Price')
plt.yscale('log')
plt.show()
#Now the correlation between price and # bedrooms is more clear.
#Enough. We could look at all of the various plots and see what it all means but let's just try to linear regression and see what happens.
linreg = LinearRegression()
#price is our label, the aspect which we want to predict
price = data['price']
#the rest are features and we can't include price as an estimate so we have to drop it
features = data #.drop('price', axis='columns')
features.head()
#Do it
linreg.fit(X=features, y=price) #Do a fit against the chosen features with price as the value to estimate
print('coeff:',linreg.coef_) #print out the linear coefficients
print('m0', linreg.intercept_) #print out the offset
#Get the predicted values so that we can compare against the correct values and get an estimate of errors
predicted_price = linreg.predict(features)
mseScore = r2_score(price, predicted_price)
print('error score:', mseScore)
#We will discuss later, but an ideal error score is 1.0 which is very suspicious that we should do so well on our first try right?
#The coefficients mean that the linear prediction is m0 + feature[1]*coeff[1]+feature[2]+coeff[2] etc
#DID YOU NOTCIE THAT COEFF[1] is 1.000 and that all of the rest of the parameters are almost 0???
#Let's plot price versus the prediction 
plt.scatter(price, predicted_price,  color='red')
plt.xlabel('Real Price')
plt.ylabel('Predicted Price')
plt.show()
#Ugh, why are there 5 features? Because we didn't drop price. In other words, we gave the answer to the linear regression and it correctly used it and 
#ignored the rest. Let's repeat the above steps after dropping the price

features = data.drop('price', axis='columns') #drop the price column
linreg.fit(X=features, y=price) #Do a fit against the chosen features with price as the value to estimate
print('coeff:',linreg.coef_) #print out the linear coefficients
print('m0', linreg.intercept_) #print out the offset
#Get the predicted values so that we can compare against the correct values and get an estimate of errors
predicted_price = linreg.predict(features)
mseScore = r2_score(price, predicted_price)
print('error score:', mseScore)
#Now we have only 4 coefficients, one for each feature: # bedrooms, square footage, lot size and year built.
#The coefficients don't make too much sense, we did expect the second parameter (square footage) to be positive (correlates with price)
#and we expected the year built to anti-correlate. But the other two are the wrong sign and what is with the huge offset? 4.5 million?
#But the error score isn't bad. The problem is that we (I) don't know enough about how this works. I can't even tell if the regression did a good job or not!
#That is why we will do it all gain, simpler and slower, in the next notebook. BUt for now, let's plot the predicted versus real price.
#BTW, we are not supposed to use the same sample for training and evaluation but we are starting slow. We'll get to that.
#Here is the plot:
plt.scatter(price, predicted_price,  color='red')
plt.xlabel('Real Price')
plt.ylabel('Predicted Price')
plt.show()
#This actually looks better than I thought. But the error score seems way off. Let's look ourselves at the differences.
diff = (price-predicted_price)/price
diff = diff.abs()
diff.head()


#The fractional difference between price and predicted price is like 30-40% for the first 5 records. Ugh.
#Let's average that
print('The mean fractional error is: ',diff.mean())

#Next time, we will look at the above more carefully and see if we can do better