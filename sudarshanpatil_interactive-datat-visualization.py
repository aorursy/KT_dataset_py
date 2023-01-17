import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.defaultaction = "ignore"



import datetime

import math

import pandas as pd

import random 

import radar 

from faker import Faker

fake = Faker()



def generateData(n):

  listdata = []

  start = datetime.datetime(2019, 8, 1)

  end = datetime.datetime(2019, 8, 30)

  delta = end - start

  for _ in range(n):

    date = radar.random_datetime(start='2019-08-1', stop='2019-08-30').strftime("%Y-%m-%d")

    price = round(random.uniform(900, 1000), 4)

    listdata.append([date, price])

  df = pd.DataFrame(listdata, columns = ['Date', 'Price'])

  df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

  df = df.groupby(by='Date').mean()



  return df
a = generateData(100)
a.head()
import matplotlib.pyplot as plt

plt.xlabel ("Dataetime")

plt.ylabel("Data")

plt.plot(a)

plt.show()
import calendar

mon = [i for i in range(1,13)]

sold = [round(random.uniform(100,200)) for i in mon]

plt.xlabel("Months ")

plt.ylabel("The No. of Units sold")

plt.plot(mon, sold)
# This is the actual bar graph

plt.bar(mon, sold)

plt.style.use("seaborn")

plt.xlabel("months of year")

plt.ylabel("the sell of ZOFLOFT")

plt.show()
figure, axis = plt.subplots()

plt.xticks(mon, calendar.month_name[1:13], rotation = 20)

plt.xlabel("months of year")

plt.ylabel("the sell of ZOFLOFT")

plot = axis.bar(mon, sold)



# For displaying the height of the columns

for rectangle in plot:

    height = rectangle.get_height()

    axis.text(rectangle.get_x() + rectangle.get_width() /2., 1.002 * height, '%d' % int(height), ha='center', va = 'bottom')

    

plt.show()
months = list(range(1, 13))

sold_quantity = [round(random.uniform(100, 200)) for x in range(1, 13)]



figure, axis = plt.subplots()



plt.yticks(months, calendar.month_name[1:13], rotation=20)



plot = axis.barh(months, sold_quantity)



for rectangle in plot:

  width = rectangle.get_width()

  axis.text(width + 2.5, rectangle.get_y() + 0.38, '%d' % int(width), ha='center', va = 'bottom')



plt.show()
headers_cols = ['age','min_recommended', 'max_recommended', 'may_be_appropriate_min', 'may_be_appropriate_max', 'min_not_recommended', 'max_not_recommended'] 



sleepDf = pd.read_csv('https://raw.githubusercontent.com/PacktPublishing/hands-on-exploratory-data-analysis-with-python/master/Chapter%202/sleep_vs_age.csv', index_col = "Unnamed: 0" )

sleepDf.head(10)
sleepDf["age_year"] = sleepDf.age // 12
plt.plot (sleepDf.age_year, sleepDf.min_recommended, 'g--')

plt.plot (sleepDf.age_year, sleepDf.max_recommended, 'r--')

plt.xlabel("The age in years")

plt.ylabel("no. of hrs the sleep is recommended")

plt.show()
from sklearn.datasets import load_iris

data = load_iris()

frame = pd.DataFrame(data.data, columns=data.feature_names)

frame["species"] = data.target

frame.head()
plt.rcParams['figure.figsize'] = (8, 6)

plt.rcParams['figure.dpi'] = 125

sns.set()

plt.xlabel("sepal length (cm)")

plt.ylabel("sepal width (cm)")

plt.scatter(frame["sepal length (cm)"], frame["sepal width (cm)"], c= frame.species) 

plt.show()
plt.rcParams['figure.figsize'] = (8, 6)

plt.rcParams['figure.dpi'] = 125

sns.set()

plt.xlabel("petal length (cm)")

plt.ylabel("petal width (cm)")

plt.scatter(frame["petal length (cm)"], frame["petal width (cm)"], c= frame.species) 

plt.show()
df = frame

plt.scatter(frame["petal length (cm)"], frame["petal width (cm)"],

            s=50*frame["petal length (cm)"]*frame["petal width (cm)"], 

            c=df.species,

            alpha=0.5

            )

plt.title("The Bubble plot of the petal lenghts to the petals width")

plt.show()
# Years under consideration

years = ["2010", "2011", "2012", "2013", "2014"]



# Available watt

watts = ['4.5W', '6.0W', '7.0W','8.5W','9.5W','13.5W','15W']

unitsSold = [

             [65, 141, 88, 111, 104, 71, 99],

             [85, 142, 89, 112, 103, 73, 98],

             [75, 143, 90, 113, 89, 75, 93],

             [65, 144, 91, 114, 90, 77, 92],

             [55, 145, 92, 115, 88, 79, 93],

            ]



# Define the range and scale for the y axis

values = np.arange(0, 600, 100)
led_bulbs = pd.DataFrame(unitsSold, columns = watts)
led_bulbs.index = years
led_bulbs.head()
plt.plot(led_bulbs.index, led_bulbs["4.5W"], "g--", marker = "o")



plt.title("The sale of bulbs over the year")

plt.xlabel("Years")

plt.ylabel("No. of units sold")



plt.show()
yearsOfExperience = np.array([10, 16, 14, 5, 10, 11, 16, 14, 3, 14, 13, 19, 2, 5, 7, 3, 20,

       11, 11, 14, 2, 20, 15, 11, 1, 15, 15, 15, 2, 9, 18, 1, 17, 18,

       13, 9, 20, 13, 17, 13, 15, 17, 10, 2, 11, 8, 5, 19, 2, 4, 9,

       17, 16, 13, 18, 5, 7, 18, 15, 20, 2, 7, 0, 4, 14, 1, 14, 18,

        8, 11, 12, 2, 9, 7, 11, 2, 6, 15, 2, 14, 13, 4, 6, 15, 3,

        6, 10, 2, 11, 0, 18, 0, 13, 16, 18, 5, 14, 7, 14, 18])

plt.hist(yearsOfExperience)

plt.xlabel("Years of experience with Python Programming")

plt.ylabel("Frequency")

plt.title("Distribution of Python programming experience in the vocational training session")

# This line represents the mean of the distributions

plt.axvline(x=yearsOfExperience.mean(), linewidth=3, color = 'g') 

plt.show()