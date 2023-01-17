import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
#Download the vehicle model dataset
# https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv


data=pd.read_csv('../input/fuelconsumption/FuelConsumption_file.csv')
data.head()
# Let’s select some features to explore more :
data = data[['ENGINESIZE','CO2EMISSIONS']]


# ENGINESIZE vs CO2EMISSIONS:
plt.scatter(data['ENGINESIZE'],data['CO2EMISSIONS'])
plt.xlabel('Engine')
plt.ylabel('CO2')
plt.show()
print(len(data))
int((len(data)*0.8))

# Generating training and testing data from our data:
# We are using 80% data for training.
train = data[:(int((len(data)*0.8)))]
test = data[(int((len(data)*0.8))):]


# Modeling:
# Using sklearn package to model data :
regr = linear_model.LinearRegression()
train_x = np.array(train[['ENGINESIZE']])
train_y = np.array(train[['CO2EMISSIONS']])
regr.fit(train_x,train_y)


# The coefficients:
print ('coefficients : ',regr.coef_) #Slope
print ('Intercept : ',regr.intercept_) #Intercept
# Plotting the regression line:
plt.scatter(train['ENGINESIZE'], train['CO2EMISSIONS'])
plt.plot(train_x, regr.coef_*train_x + regr.intercept_, '-r')
plt.plot(train_x, regr.coef_*train_x + regr.intercept_, '-r')
plt.xlabel('Engine size')
plt.ylabel('Emission')
plt.show()
test_x = np.array(test[['ENGINESIZE']])
test_y = np.array(test[['CO2EMISSIONS']])


# Plotting the regression line:
plt.scatter(test['ENGINESIZE'], test['CO2EMISSIONS'])
plt.plot(test_x, regr.coef_*test_x + regr.intercept_, '-r')
plt.plot(test_x, regr.coef_*test_x + regr.intercept_, '-r')
plt.xlabel('Engine size')
plt.ylabel('Emission')
plt.show()


#Predicting 

# Predicting values:
# Function for predicting future values :
def get_regression_predictions(input_features,intercept,slope):
 predicted_values = input_features*slope + intercept
 return predicted_values
# Predicting emission for future car:



my_engine_size = 3.5
estimatd_emission = get_regression_predictions(my_engine_size,regr.intercept_[0],regr.coef_[0][0])
print ('Estimated Emission :',estimatd_emission)
# Checking various accuracy:
from sklearn.metrics import r2_score
test_y = regr.predict(test_x)

print('Mean absolute error: %.2f' % np.mean(np.absolute(test_y,test_y)))
print('Mean sum of squares (MSE): %.2f' % np.mean((test_y-test_y) ** 2))
print('R2-score: %.2f' % r2_score(test_y , test_y) )
#Download the vehicle model dataset
# https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv


data=pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv')
data.head()
train=data[:int(len(data)*0.8)]
print(train.head())

test=data[int(len(data)*0.8):]
print(test.head())
X_train=train[['MODELYEAR','ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','FUELCONSUMPTION_COMB_MPG']]


y_train=train[['CO2EMISSIONS']]

X_test=test[['MODELYEAR','ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','FUELCONSUMPTION_COMB_MPG']]


y_test=test[['CO2EMISSIONS']]

# Modeling:
# Using sklearn package to model data :
regr = linear_model.LinearRegression()
train_x = np.array(X_train)
train_y = np.array(y_train)
regr.fit(train_x,train_y)
test_x = np.array(X_test)
test_y = np.array(y_test)

predict=regr.predict(test_x)
# Check accuracy:
from sklearn.metrics import r2_score
R = r2_score(test_y , predict)
print ('R² :',R)