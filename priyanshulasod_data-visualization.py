import matplotlib.pyplot as plt

import numpy as np

import pandas as pd
x = np.arange(0,10)

y = np.random.randint(22, 42, 10)

color = np.random.rand(10)

plt.plot(x, y, c = "green")

plt.scatter(x,y, c = ["red", "green", "pink", "yellow", "blue", "red", "green", "pink", "yellow", "blue"])

plt.xlabel("Day")

plt.ylabel("Temp")

plt.title("Day Vs Temp")

plt.show()
df = pd.read_csv("../input/nba.csv")

plt.hist(df.Age, bins=10)

df.keys()
fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

ax.bar(df.Age,df.Number)

plt.show()
x = np.random.randint(10,40, 6)

y = np.random.randint(2010, 2020, 6)

explode = np.random.randint(0,1,6)

explode = explode.astype(float)

explode[x.argmin()] = 0.2

plt.pie(x, labels=y, explode=explode, autopct="%1.1f%%")

plt.show()
x = [23,56,21,22,13,16,17,10,20,45,39,35,28,40,47]

y = np.random.randint(10,50,15)

plt.scatter(x,y)