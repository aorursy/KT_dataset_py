#import libraries 
import matplotlib.pyplot as plt 
import pandas as pd 
import pylab as pl
import numpy as np 
import os
%matplotlib inline
#Download data
!wget -O FuelConsumption.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv
#Read the data in 
df = pd.read_csv("FuelConsumption.csv")

df.head()
#Data exploration -- summarize
df.describe()
#Select feature to explore 
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)
#Plot features 
viz = cdf[['CYLINDERS', 'ENGINESIZE','CO2EMISSIONS', 'FUELCONSUMPTION_COMB']]
viz.hist()
plt.show()
#Plot each of these features vs the emission to see linear relation 
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("Fuel Consumption Comb")
plt.ylabel("Emissions")
plt.show()
#Plot Cylinder vs Emission 
plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color="Red")
plt.xlabel("Cylinders")
plt.ylabel("CO2 Emissions")
plt.show()


#Create a mask to select random rows 
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]
#Train data distribution  
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color="Blue")
plt.xlabel('Engine Size')
plt.ylabel("Emissions")
plt.show
#Modeling using scikit-learn
from sklearn import linear_model

regr= linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x,train_y)

#the coeff 
print('Coefficients:', regr.coef_)
print('Intercept:', regr.intercept_)
#Plot outputs 
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color="Blue")
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine Size")
plt.ylabel("Emissions")
#Evaluation using MSE 
from sklearn.metrics import r2_score 

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_hat = regr.predict(test_x)

print("Mean Abs. error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean(test_y_hat - test_y))
print("R2-Score: %.2f" % r2_score(test_y_hat, test_y))
