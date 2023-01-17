#Cell - 4: Execute code to ensure the data has been imported successfully.

#Output: A list view of the files available.

import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from IPython.display import display

import seaborn as sns

print(os.listdir("../input"))
# Write your code for Q1 and execute the cell

q1=pd.read_csv("../input/quakes.csv")

q1.head(10)
plt.scatter(q1["lat"], q1["long"], alpha=0.50)

plt.show()
# Write your code for Q2 and execute the cell

d = pd.Series([5,7,2,4,5,6,4,5,6,4,3,5,6,5,3])

print("Mean: ", d.mean())

print("Mode: ", d.mode())

print("Median: ", d.median())
# Write your code for Q3 and execute the cell

c1=pd.read_csv("../input/chickwts.csv")

c1.head(10)
w1 = c1["weight"]

display("Mean: ", w1.mean())

display("Median: ", w1.median())

display("80% value: ", w1.quantile(0.8))

display("Weight description: ", w1.describe())
f1=c1["feed"]

print("value count:\n" ,f1.value_counts())

print('------------------------------------------------')

print("proportion:\n", f1.value_counts()/f1.size)

print('------------------------------------------------')

print("proportion:\n", f1.value_counts()/f1.size*100)
sns.countplot(f1)
# Write your code for Q4 and execute the cell

q1=pd.read_csv("../input/quakes.csv")

print("summary for column mag:\n", q1["mag"].describe())

# Write your code for Q5 and execute the cell

o = pd.read_csv("../input/orange.csv")

o1 = o[["age","circumference"]]

print(o1)

print('------------------------------------------------')

print("correlation:\n", o1.corr())

plt.scatter(o1["age"],o1["circumference"], alpha=0.5)

plt.title('scatter plot')

plt.xlabel('age')

plt.ylabel('circumference')

plt.show()
# Write your code for Q6 and execute the cell

m1=pd.read_csv("../input/mtcars.csv")

m1.head()
cf = m1["cyl"].value_counts()

cd = cf.apply(pd.Series)

cd.columns = ["freq"]

print(cd)
# Write your code for Q7 and execute the cell

plt.pie(cd["freq"], labels=cd.index, startangle=90)

plt.show()
# Write your code for Q8 and execute the cell

o = pd.read_csv("../input/orange.csv")

o1 = o[["age"]]

o1["age"].value_counts()
plt.boxplot(o1["age"])

plt.show()
# Write your code for Q9 and execute the cell

o2 = o[["age", "circumference"]]

o2.head()
sns.boxplot(

    x = 'age',

    y = 'circumference',

    data = o2

)