# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import matplotlib.pyplot as plt
## Basic Plot



x = [10, 20, 30, 40, 50]

y = [45, 75, 30, 80, 40]

plt.bar(x,y)

plt.show()
x = ['Ajay', 'Tom', 'Vicky', 'Anouska','Mikhail']

y = [4500, 7500, 3000, 8000, 40000]

plt.title(" Bar graph example") # Name title of the graph

plt.xlabel('Employees') # Assign the name of the x axis

plt.ylabel("Salary") # Assign the name of the y axis

plt.bar(x, y, color='red') # Change bar color

plt.show()
barplot = plt.bar(x, y)

for bar in barplot:

    yval = bar.get_height()

    plt.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), va='bottom') #va: vertical alignment y positional argument

    

plt.title(" Bar graph example")

plt.xlabel('Employees')

plt.ylabel("Salary")
plt.figure(figsize=(7,7))

barplot = plt.bar(x, y)

for bar in barplot:

    yval = bar.get_height()

    plt.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), va='bottom') #va: vertical alignment y positional argument

    

plt.title(" Bar graph example")

plt.xlabel('Employees')

plt.ylabel("Salary")

plt.show()
plt.figure(figsize=(7,7))

barplot = plt.bar(x, y)

for bar in barplot:

    yval = bar.get_height()

    plt.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), va='bottom') #va: vertical alignment y positional argument

    plt.yticks([])

plt.title(" Bar graph example")

plt.xlabel('Employees')

plt.ylabel("Salary")

plt.show()
plt.barh(x,y)

plt.title("Horizontal Bar graph example")

plt.xlabel("Employees")

plt.ylabel('Salary')
y,x = zip(*sorted(zip(y,x)))

plt.barh(x,y)
plt.barh(x,y)

ax=plt.subplot()

ax.invert_yaxis()
print(plt.style.available)
plt.barh(x,y)

plt.title("Horizontal Bar graph example")

plt.xlabel("Employees")

plt.ylabel('Salary')

plt.style.use('dark_background')
plt.barh(x,y)

plt.title("Horizontal Bar graph example")

plt.xlabel("Employees")

plt.ylabel('Salary')

plt.style.use('seaborn-dark-palette')
import pandas as pd



df = pd.DataFrame({"Year" : [2015,2016,2017,2018,2019], 

                  "Salary_Hike" : [2000, 3000, 4000, 3500, 6000]})
# plot line chart

plt.plot(df["Year"], df["Salary_Hike"])

plt.title("Simple Line Plot")

plt.xlabel('Year')

plt.ylabel('Salary_Hike')

plt.style.use('seaborn-white')
ax = df.plot(x="Year", y="Salary_Hike", kind="line", title ="Simple Line Plot", legend=True, style = 'b--')

ax.set(ylabel='Salary_Hike', xlabel = 'Year', xticks =df["Year"])
import pandas as pd



product = pd.DataFrame({"Year" : [2014,2015,2016,2017,2018], 

                  "ProdAManufacture" : [2000, 3000, 4000, 3500, 6000],

                  "ProdBManufacture" : [3000, 4000, 3500, 3500, 5500]})





# Multi line plot

ax = product.plot("Year", "ProdAManufacture", kind="line", label = 'Product A manufacture')

product.plot("Year", "ProdBManufacture", ax= ax , kind="line", label = 'Product B manufacture', title= 'MultiLine Plot') #ax : axes object



# Set axes

ax.set(ylabel='Sales', xlabel = 'Year', xticks =df["Year"])
ax = product.plot("Year", "ProdAManufacture", kind='scatter', color = 'red', title = 'Year by ProductA Manufacture')

ax.set(ylabel='ProdAManufacture', xlabel = 'Year', xticks =df["Year"])

plt.show()
Goals = [20, 12, 11, 4, 3]

players = ['Ronaldo', 'Messi', 'Suarez', 'Neymar', 'Salah', ]

comp = pd.DataFrame({"Goals" : Goals, "players" : players})

ax = comp.plot(y="Goals", kind="pie", labels = comp["players"], autopct = '%1.0f%%', legend=False, title='No of Goals scored')



# Hide y-axis label

ax.set(ylabel='')
ax = comp.plot(y="Goals", kind="pie", labels = comp["players"], startangle = 90, shadow = True, 

        explode = (0.1, 0.1, 0.1, 0, 0), autopct = '%1.0f%%', legend=False, title='No of Goals scored')

ax.set(ylabel='')

plt.show()
# Creating random data

import numpy as np

np.random.seed(1)

mydf = pd.DataFrame({"Age" : np.random.randint(low=20, high=100, size=50)})



# Histogram

ax = mydf.plot(bins= 5, kind="hist", rwidth = 0.7, title = 'Distribution - Marks', legend=False)

ax.set(xlabel="Bins")

plt.show()
labels = ['Amsterdam', 'Berlin', 'Brussels', 'Paris']

x1 = [45, 30, 15, 10]

x2 = [25, 20, 25, 50]



finaldf = pd.DataFrame({"2017_Score":x1, "2018_Score" : x2, "cities" : labels})
# Method 1



fig = plt.figure()



ax1 = fig.add_subplot(121)

ax = finaldf.plot(x="cities",  y="2017_Score", ax=ax1, kind="barh", legend = False, title = "2017 Score")

ax.invert_yaxis()



ax2 = fig.add_subplot(122)

ax = finaldf.plot(x="cities",  y="2018_Score", ax=ax2, kind="barh", legend = False, title = "2018 Score")

ax.invert_yaxis()

ax.set(ylabel='')
#Method 2



fig, (ax0, ax01) = plt.subplots(1, 2)



ax = finaldf.plot(x="cities",  y="2017_Score", ax=ax0, kind="barh", legend = False, title = "2017 Score")

ax.invert_yaxis()



ax = finaldf.plot(x="cities",  y="2018_Score", ax=ax01, kind="barh", legend = False, title = "2018 Score")

ax.invert_yaxis()

ax.set(ylabel='')
fig = plt.figure()

ax1 = fig.add_subplot(211)

ax = finaldf.plot(x="cities",  y="2017_Score", ax=ax1, kind="barh", legend = False, title = "2017 vs 2018 Score")

ax.invert_yaxis()

plt.xticks(range(0,60,10))

ax.set(ylabel='')



ax2 = fig.add_subplot(212)

ax = finaldf.plot(x="cities",  y="2018_Score", ax=ax2, kind="barh", legend = False)

ax.invert_yaxis()

ax.set(ylabel='')
def f(x, y):

    return (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 2 -y ** 2)

n = 256

x = np.linspace(-3, 3, n)

y = np.linspace(-3, 3, n)

X, Y = np.meshgrid(x, y)

plt.contourf(X, Y, f(X, Y), 8, alpha=.75, cmap='jet')

C = plt.contour(X, Y, f(X, Y), 8, colors='black', linewidth=.5)
def f(x, y):

    return (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)

n = 10

x = np.linspace(-3, 3, 4 * n)

y = np.linspace(-3, 3, 3 * n)

X, Y = np.meshgrid(x, y)

plt.imshow(f(X, Y))

plt.show()
fig, ax = plt.subplots()





x_pos = 0

y_pos = 0

x_direct = 1

y_direct = 1





ax.quiver(x_pos, y_pos, x_direct, y_direct)

ax.set_title('Quiver plot with one arrow')





plt.show()
fig, ax = plt.subplots()



x_pos = [0, 0]

y_pos = [0, 0]

x_direct = [1, 0]

y_direct = [1, -1]





ax.quiver(x_pos,y_pos,x_direct,y_direct,

         scale=5)

ax.axis([-1.5, 1.5, -1.5, 1.5])





plt.show()
n = 8

X, Y = np.mgrid[0:n, 0:n]

plt.quiver(X, Y)

plt.show()
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()

ax = Axes3D(fig)

X = np.arange(-4, 4, 0.25)

Y = np.arange(-4, 4, 0.25)

X, Y = np.meshgrid(X, Y)

R = np.sqrt(X**2 + Y**2)

Z = np.sin(R)

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot')
#Polar Axis 

r = np.arange(0, 2, 0.01)

theta = 2 * np.pi * r



ax = plt.subplot(111, projection='polar')

ax.plot(theta, r)

ax.set_rmax(2)

ax.set_rticks([0.5, 1, 1.5, 2])  # less radial ticks

ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line

ax.grid(True)



ax.set_title("A line plot on a polar axis", va='bottom')

plt.show()
x = ['A','B','C', 'D']

y = [100,119,800,900]

plt.bar(x, y)

ax = plt.subplot()

ax.set_ylim(0,1000)

plt.show()
plt.legend(["First_Legend","Second_Legend"])
plt.bar(x, y)

plt.title("Cost of Living", fontsize=18, fontweight='bold', color='blue')

plt.xlabel("Cities", fontsize=16)

plt.ylabel("Score", fontsize=16)