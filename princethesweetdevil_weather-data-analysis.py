import pandas as pd

import matplotlib.pyplot as pl

import numpy as np

import seaborn as sns
data = pd.read_csv("/kaggle/input/weather-many-cities/weather.csv")
data
data.info()
data.describe
print(list(data))
cities = data.groupby('city')
def retrieve_array(values):

    val = []

    for x in cities.city.unique().values:

        val.append(x[0])

    return val



def F2C(x):

    return (x - 32) * 5/9 #Fahrenheit = (Celsius * 9/5) + 32.

pl.pie(cities.city.count(),labels=retrieve_array(cities.city.unique()[:]),shadow=True,radius=3,labeldistance=0.6)

pl.show()
print("Average Highest Tempreture (Based On City)",F2C(cities.high_temp.sum()/((365*2)+1)))

print()

print("Average Average Tempreture (Based On City)",F2C(cities.avg_temp.sum()/((365*2)+1)))

print()

print("Average Lowest Tempreture (Based On City)",F2C(cities.low_temp.sum()/((365*2)+1)))
Y2016 = data[data.year == 2016]

Y2017 = data[data.year == 2017]
Days2016 = Y2016[Y2016.city == 'Mumbai'].groupby(['city','month']).date.count()

Mumbai2016 = Y2016[Y2016.city == 'Mumbai'].groupby(['city','month'])



sns.barplot([x for x in range(1,13)],F2C(Mumbai2016.high_temp.sum()/Days2016))

pl.title('Highest Tempreture Monthwise Mumbai-2016')

pl.yticks([x for x in range(0,38,2)])

pl.show()



sns.barplot([x for x in range(1,13)],F2C(Mumbai2016.avg_temp.sum()/Days2016))

pl.title('Highest Tempreture Monthwise Mumbai-2016')

pl.yticks([x for x in range(0,38,2)])

pl.show()



sns.barplot([x for x in range(1,13)],F2C(Mumbai2016.low_temp.sum()/Days2016))

pl.title('Highest Tempreture Monthwise Mumbai-2016')

pl.yticks([x for x in range(0,38,2)])

pl.show()

Days2017 = Y2017[Y2017.city == 'Mumbai'].groupby(['city','month']).date.count()

Mumbai2017 = Y2017[Y2017.city == 'Mumbai'].groupby(['city','month'])



sns.barplot([x for x in range(1,13)],F2C(Mumbai2017.high_temp.sum()/Days2016))

pl.title('Highest Tempreture Mumbai-2016')

pl.yticks([x for x in range(0,38,2)])

pl.show()



sns.barplot([x for x in range(1,13)],F2C(Mumbai2017.avg_temp.sum()/Days2016))

pl.title('Highest Tempreture Mumbai-2016')

pl.yticks([x for x in range(0,38,2)])

pl.show()



sns.barplot([x for x in range(1,13)],F2C(Mumbai2017.low_temp.sum()/Days2016))

pl.title('Highest Tempreture Mumbai-2016')

pl.yticks([x for x in range(0,38,2)])

pl.show()
