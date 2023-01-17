# Before we start, we need to import the necessary libraries.

import numpy as np

import pandas as pd

import matplotlib

import matplotlib.pyplot as plt
# Dummy Dataset

data = np.array([3,1,4,1,5,9,2,6,5])
plt.plot(data)  # plot data

plt.show()  # show data
# Dummy Dataset

x = np.array([1,2,3])

y = np.array([1,4,9])
plt.plot(x, y)  # (x axis, y axis)

plt.show()
x = np.array([1,2,3])

y = np.array([1,4,9])



plt.title("test") # Graph Title

plt.xlabel("x") # X axis label

plt.ylabel("y") # Y axis label



plt.plot(x, y)

plt.show()
x = np.array([1,2,3])

y = np.array([1,4,9])



# Fontsize

plt.title("test", fontsize = 20)

plt.xlabel("x", fontsize = 20)

plt.ylabel("y", fontsize = 20)



plt.plot(x, y)

plt.show()
x = np.array([1,2,3])

y = np.array([1,4,9])



plt.title("test")

plt.xlabel("x")

plt.ylabel("y")

plt.plot(x, y)



# Legend

plt.legend(["Blue Team"])

plt.show()
x = np.array([1,2,3])

y = np.array([1,4,9])



plt.title("test")

plt.xlabel("x")

plt.ylabel("y")

plt.plot(x, y)



# legend(loc)

plt.legend(["Blue Team"], loc=1)

plt.show()
x = np.array([1,2,3])

y = np.array([1,4,9])



plt.title("test")

plt.xlabel("x")

plt.ylabel("y")



# color、linewidth、linestyle

plt.plot(x, y, color='red', linewidth=2, 

         linestyle='--')



plt.legend(["Blue Team"], loc=1)

plt.show()
x = np.linspace(-1, 1, 50)



y1 = x + 1

y2 = -x*0.5



plt.plot(x, y1)

plt.plot(x, y2)



# Be careful of the legend order

plt.legend(["y = x + 1", "y = -0.5x"])

plt.show()
# Change the Scales of X axis

x = np.linspace(-5, 5, 50)  



# The scale of Y axis will be changed correspondingly

y1 = x + 1

y2 = -x*0.5



plt.plot(x, y1)

plt.plot(x, y2)

plt.legend(["y = x + 1", "y = -0.5x"])

plt.show()
x = np.linspace(-5, 5, 50)

y1 = x + 1

y2 = -x*0.5

plt.plot(x, y1)

plt.plot(x, y2)



plt.xlim((2,3))  # limit X axis 

plt.ylim((4,8)) # limit Y axis

plt.legend(["y = x + 1", "y = -0.5x"])

plt.show()
# Start you answer here
x = np.linspace(-1, 1, 50)

y = 2*x + 1



plt.plot(x, y)

plt.legend(["y = 2x + 1"])

plt.show()
# Start you answer here
x = np.linspace(-1, 1, 50)

y = x**2



plt.plot(x, y)

plt.legend(["y = x^2"])

plt.show()
# Start you answer here
x = np.arange(0, 10, 0.01)

y = np.sin(x)



plt.plot(x, y)

plt.show()
# use this package for "e"

import math

e = math.e



# Start you answer here
x = np.arange(-10, 10, 0.1)

y = 1 / (1 + e**-x)



plt.plot(x, y)

plt.show()
# Dummy Dataset

Score = np.array([20, 35, 30, 35, 27])



plt.bar([0,1,2,3,4], Score)

plt.show()
height = np.array([10, 45, 30, 15, 50])



plt.bar(x=[0,1,2,3,4], height=height, 

        color="green", width=0.5)

plt.show()
height1 = np.array([10, 45, 30, 15, 50])

height2 = np.array([35, 15, 20, 15, 20])



plt.bar(x=[0,1,2,3,4], height=height1, color="green", width=0.5)



# set bottom = height1, means start plotting height2 base on height1.

plt.bar(x=[0,1,2,3,4], height=height2, bottom=height1, color="orange", width=0.5) 



plt.legend(['height1','height2'])

plt.show()
price = np.array([10, 45, 30, 15, 50])



# error bar needs standard deviation data for each bar.

std = np.array([2, 5, 8, 3, 1])



# yerr

plt.bar(x=[0,1,2,3,4], height=price, yerr = std,

        color="green", width=0.5)



plt.show()
price = np.array([10, 45, 30, 15, 50])

std = np.array([2, 5, 8, 3, 1])



plt.xlabel("Stock")

plt.ylabel("Price")

plt.title("Stock Price")

plt.bar(x=[0,1,2,3,4], height=price, yerr = std,

        color="green", width=0.5)



plt.show()
price = np.array([10, 45, 30, 15, 50])

std = np.array([2, 5, 8, 3, 1])



# Instead of [0,1,2,3,4], put the categories you need for each bar.

x_axis = ["APPLE", "AMAZON", "YAHOO", "GOOGLE", "FACEBOOK"]



plt.xlabel("Stock")

plt.ylabel("Price")

plt.title("Stock Price")

plt.bar(x=x_axis, height=price, yerr = std,

        color="green", width=0.5)



plt.show()
price = np.array([10, 45, 30, 15, 50])

std = np.array([2, 5, 8, 3, 1])

x_axis = ["APPLE", "AMAZON", "YAHOO", "GOOGLE", "FACEBOOK"]



plt.xlabel("Stock")

plt.ylabel("Price")

plt.title("Stock Price")

plt.bar(x=x_axis, height=price, yerr = std,

        color="green", width=0.5)



plt.grid(True)

plt.show()
price = np.array([10, 45, 30, 15, 50])

std = np.array([2, 5, 8, 3, 1])

x_axis = ["APPLE", "AMAZON", "YAHOO", "GOOGLE", "FACEBOOK"]



plt.xlabel("Stock")

plt.ylabel("Price")

plt.title("Stock Price")

plt.bar(x=x_axis, height=price, color="green", width=0.5)



# add annotations

for x, y in zip(x_axis, price):

    plt.text(x, y, y, ha='center', va='bottom')



plt.grid(True)

plt.show()
bitcoin = pd.read_csv("/kaggle/input/Bitcoin.csv")

bitcoin.head()
# Start you answer here
# set graph size

fig=plt.figure(figsize=(12,8))



plt.grid(True)

plt.plot(bitcoin.Date, bitcoin.Price)

plt.xticks(rotation=30)



for x, y in zip(bitcoin.Date, bitcoin.Price):

    plt.text(x, y, y, ha='center', va='bottom')
# Start you answer here
fig=plt.figure(figsize=(15,8))



plt.xticks(rotation=30)

plt.grid(True)

plt.bar(bitcoin.Date, bitcoin.Price)



for x, y in zip(bitcoin.Date, bitcoin.Price):

    plt.text(x, y, y, ha='center', va='bottom')
# import dataset that will be used.

world = pd.read_csv("/kaggle/input/world2015.csv")

world.head()
fig=plt.figure(figsize=(15,8))



plt.scatter(world.GDP_per_capita, world.Life_expectancy)
fig=plt.figure(figsize=(15,8))



# calculate the size based on population for each country

size = world.Population / 1e6 # 1e6 = 1,000,000



plt.scatter(world.GDP_per_capita, world.Life_expectancy, s=size, alpha = 0.5)



plt.show()
fig=plt.figure(figsize=(15,8))



# first create map dictionary

map_dict = {      

    'Asia':'red',

    'Europe':'green',

    'Africa':'blue',

    'North America':'yellow',

    'South America':'yellow',

    'Oceania':'black'

}



# Then, use 'map'function to change the color by continent.

colors = world.Continent.map(map_dict)



size = world.Population / 1e6

plt.scatter(world.GDP_per_capita, world.Life_expectancy, c=colors, s=size, alpha = 0.5)

plt.show()
# Start you answer here
fig=plt.figure(figsize=(15,8))



map_dict = {      

    'Asia':'red',

    'Europe':'green',

    'Africa':'blue',

    'North America':'yellow',

    'South America':'yellow',

    'Oceania':'black'

}

colors = world.Continent.map(map_dict)

fig=plt.figure(figsize=(15,8))

size = world.Population / 1e6



plt.xlabel("GDP_per_capita", size = 20)

plt.ylabel("Population(Billion)", size = 20)

plt.title("Population & GDP_per_capita", size = 20)



plt.scatter(world.GDP_per_capita, world.Population, c=colors, s=size, alpha = 0.5)

plt.grid(True)

plt.show()
# Dummy Dataset

labels=['Python','Java','C++','Ruby']

values=[20,30,40,10]



colors=['yellow','green','red','blue']



plt.pie(values, labels=labels, colors=colors)

plt.title("Pie Chart")

plt.show()
labels=['Python','Java','C++','Ruby']

values=[20,30,40,10]

colors=['yellow','green','red','blue']



# figsize(horizontal size, vertial size)

# but for pie chart, both horizontal and vertial size will be the min(h-size, v-size).

fig=plt.figure(figsize=(5,5))



# Try the following code.

# fig=plt.figure(figsize=(5,50))



plt.pie(values, labels=labels, colors=colors)

plt.title("Pie Chart")

plt.show()
labels=['Python','Java','C++','Ruby']

values=[20,30,40,10]

colors=['yellow','green','red','blue']



fig=plt.figure(figsize=(5,5))



# shadow

plt.pie(values, labels=labels, colors=colors, shadow = True)

plt.title("Pie Chart")

plt.show()
labels=['Python','Java','C++','Ruby']

values=[20,30,40,10]

colors=['yellow','green','red','blue']



# This will separate parts away from the center of the graph.

explode = [0.2, 0, 0, 0]



fig=plt.figure(figsize=(5,5))

plt.pie(values, labels=labels, colors=colors, 

        shadow = True, explode = explode)

plt.title("Pie Chart")

plt.show()
labels=['Python','Java','C++','Ruby']

values=[200,300,400,100] # new value list

colors=['yellow','green','red','blue']

explode = [0.2, 0, 0, 0]



fig=plt.figure(figsize=(5,5))



# autopct will show the percentage of each parts on the graph.

plt.pie(values, labels=labels, colors=colors, 

        shadow = True, explode = explode, autopct='%0.1f%%')

plt.title("Pie Chart")

plt.show()
labels=['Python','Java','C++','Ruby']

values=[20,30,40,10]

colors=['yellow','green','red','blue']

explode = [0.2, 0, 0, 0]



fig=plt.figure(figsize=(5,5))



# pctdistance：change the location of numbers added by autopct.

plt.pie(values, labels=labels, colors=colors, 

        shadow = True, explode = explode, 

        autopct='%0.1f%%', pctdistance=0.3)

plt.title("Pie Chart")

plt.show()
labels=['Python','Java','C++','Ruby']

values=[20,30,40,10]

colors=['yellow','green','red','blue']

explode = [0.2, 0, 0, 0]



fig=plt.figure(figsize=(5,5))



# radius: change the size of pie.

plt.pie(values, labels=labels, colors=colors, 

        shadow = True, explode = explode, 

        autopct='%0.1f%%', pctdistance=0.3,

       radius = 1.5)



plt.title("Pie Chart")

plt.show()
labels=['Python','Java','C++','Ruby']

values=[20,30,40,10]

colors=['yellow','green','red','blue']

explode = [0.2, 0, 0, 0]



fig=plt.figure(figsize=(5,5))



# wedgeprops = {"linewidth":  , "width":  , "edgecolor":  }

plt.pie(values, labels=labels, colors=colors, 

        shadow = True, explode = explode, 

        autopct='%0.1f%%', pctdistance=0.8,

       radius = 1, wedgeprops={"linewidth": 3, "width": 0.4, "edgecolor": "white"})

plt.title("Pie Chart")

plt.show()
# startangle

labels=['Python','Java','C++','Ruby']

values=[20,30,40,10]

colors=['yellow','green','red','blue']

explode = [0.2, 0, 0, 0]



fig=plt.figure(figsize=(5,5))



# startangle

plt.pie(values, labels=labels, colors=colors, 

        shadow = True, explode = explode, 

        autopct='%0.1f%%', pctdistance=0.8,

       radius = 1, wedgeprops={"linewidth": 3, "width": 0.4, "edgecolor": "white"},

       startangle = 180)



plt.title("Pie Chart")

plt.show()
labels=['Python','Java','C++','Ruby']

values=[20,30,40,10]

colors=['yellow','green','red','blue']

explode = [0.2, 0, 0, 0]



fig=plt.figure(figsize=(5,5))



# textprops

plt.pie(values, labels=labels, colors=colors, 

        shadow = True, explode = explode, 

        autopct='%0.1f%%', pctdistance=0.8,

       radius = 1, wedgeprops={"linewidth": 3, "width": 0.4, "edgecolor": "white"},

       startangle = 180, textprops = {"color": "purple", "weight": "bold"})



plt.title("Pie Chart")

plt.show()
# pei chart 1

labels=['Python','Java','C++','Ruby']

values=[20,30,40,10]

colors=['yellow','green','red','blue']

explode = [0.2, 0, 0, 0]



# pie chart 2

labels2=["Amater", "Pro",

       "Amater", "Pro",

       "Amater", "Pro",

       "Amater", "Pro"]

values2 = [10,10,20,10,15,25,3,7]

colors2 = ["greenyellow", "khaki",

          "limegreen", "olive",

          "darkorange", "salmon",

          "skyblue", "violet"]

explode2 = [0.2, 0.2, 0, 0, 0, 0, 0, 0]
# Start you answer here
fig=plt.figure(figsize=(8,8))



# pie 1

plt.pie(values, labels=labels, colors=colors, 

        shadow = False, explode = explode, 

        autopct='%0.1f%%', pctdistance=1.3,

        radius = 1, wedgeprops={"linewidth": 3, "width": 0.4, "edgecolor": "white"},

        textprops = {"fontsize": 13, "color": "purple", "weight": "bold"})



# pie 2

plt.pie(values2, labels=labels2, colors=colors2, 

        shadow = False, explode = explode2, 

        autopct='%0.1f%%', pctdistance=0.7,

        radius = 0.7, wedgeprops={"linewidth": 3, "width": 0.4, "edgecolor": "white"},

        textprops = {"fontsize": 13, "color": "white", "weight": "bold"})



plt.title("Most Popular Programming Language", size=20)

plt.show()