import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sb

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn import metrics

%matplotlib inline



weatherSummary = pd.read_csv("../input/weatherww2/Summary of Weather.csv")

weatherStationLocations = pd.read_csv("../input/weatherww2/Weather Station Locations.csv")



weatherSummary.shape  # It will give number of Rows and Columns 



weatherSummary.describe()



weatherSummary.plot(x='MinTemp',y='MaxTemp',style='o')

plt.title('MinTemp Vs MaxTemp')

plt.xlabel('MinTemp')

plt.ylabel('MaxTemp')

plt.show()



plt.figure(figsize=(15,10))

plt.tight_layout()

sb.distplot(weatherSummary['MaxTemp'])



x = weatherSummary['MinTemp'].values.reshape(-1,1)

y = weatherSummary['MaxTemp'].values.reshape(-1,1)



x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=0)



# print(len(x_train), len(y_train), len(x_test), len(y_test))



linearRegressor = LinearRegression()

linearRegressor.fit(x_train,y_train)



print(linearRegressor.intercept_)



print(linearRegressor.coef_)



print(linearRegressor.score(x,y))





y_predict = linearRegressor.predict(x_test)



val = pd.DataFrame({'Actual':y_test.flatten(),'predicted':y_predict.flatten()})



val1 = val.head(25)

val1.plot(kind='bar',figsize=(16,10))

plt.grid(which='major',linestyle='-',linewidth='0.5',color='green')

plt.grid(which='minor',linestyle=':',linewidth='0.5',color='black')

plt.show()



plt.scatter(x_test, y_test,  color='gray')

plt.plot(x_test, y_predict, color='red', linewidth=2)

plt.show()





print('Mean Absolute Error: ', metrics.mean_absolute_error(y_test, y_predict))  

print('Mean Squared Error: ', metrics.mean_squared_error(y_test, y_predict))  

print('Root Mean Squared Error: ', np.sqrt(metrics.mean_squared_error(y_test, y_predict)))


