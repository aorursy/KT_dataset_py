import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
dataset = pd.read_csv('/kaggle/input/daily-temperature-of-major-cities/city_temperature.csv')

dataset.head(5)
average_temperatures = dataset[dataset['Region'] == 'Europe'].groupby(dataset.Year).mean()

#removing not meaningfull rows

average_temperatures = average_temperatures[(average_temperatures.index != 200) & (average_temperatures.index != 201) & (average_temperatures.index != 2020)]

print(average_temperatures)
import matplotlib.pyplot as plt

plt.xlabel('Years')

plt.ylabel('Temperatures (F)')

plt.errorbar(average_temperatures.index, average_temperatures.AvgTemperature, color='black', fmt='.')

plt.show()

#We can see how during the years the temperature has been growing steadily
from sklearn import linear_model



x = list(map(lambda val: [val], average_temperatures.index))

y = list(average_temperatures.AvgTemperature)



reg = linear_model.Ridge(alpha=1)

reg.fit(x, y)

prediction = reg.predict(x)
plt.xlabel('Years')

plt.ylabel('Temperatures (F)')



plt.errorbar(average_temperatures.index, average_temperatures.AvgTemperature, fmt='.', color='black')

plt.plot(x, prediction, color='blue', linewidth='1')

plt.show()
reg2 = linear_model.Ridge(alpha=50)

reg2.fit(x, y)

prediction2 = reg2.predict(x)



reg3 = linear_model.Ridge(alpha=500)

reg3.fit(x, y)

prediction3 = reg3.predict(x)
plt.xlabel('Years')

plt.ylabel('Temperatures (F)')



plt.errorbar(average_temperatures.index, average_temperatures.AvgTemperature, fmt='.', color='black')

plt.plot(x, prediction, color='blue', linewidth='1')

plt.plot(x, prediction2, color='green', linewidth='1')

plt.plot(x, prediction3, color='purple', linewidth='1')

plt.show()
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import PolynomialFeatures

x = list(map(lambda val: [val], average_temperatures.index))

y = list(average_temperatures.AvgTemperature)



# alpha 1

polymodel = make_pipeline(PolynomialFeatures(2), linear_model.Ridge(alpha=0))

polymodel.fit(x, y)

predpolymodel = polymodel.predict(x)

#alpha 2

polymodel2 = make_pipeline(PolynomialFeatures(2), linear_model.Ridge(alpha=0.01))

polymodel2.fit(x, y)

predpolymodel2 = polymodel2.predict(x)

#alpha 3

polymodel3 = make_pipeline(PolynomialFeatures(2), linear_model.Ridge(alpha=0.001))

polymodel3.fit(x, y)

predpolymodel3 = polymodel3.predict(x)
plt.xlabel('Years')

plt.ylabel('Temperatures (F)')



plt.errorbar(average_temperatures.index, average_temperatures.AvgTemperature, fmt='.', color='black')

p1 = plt.plot(x, predpolymodel, color='orange')

p2 = plt.plot(x, predpolymodel2, color='pink')

p3 = plt.plot(x, predpolymodel3, color='brown')



plt.legend(['alpha = 0', 'alpha = 0.01', 'alpha = 0.001'], numpoints=1)

plt.show()