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

df1 = pd.read_csv("../input/quakes.csv")

df1.head(10)
plt.scatter(df1["lat"], df1["long"], alpha=0.5)

plt.show
# Write your code for Q2 and execute the cell

d = pd.Series([5,7,2,4,5,6,4,5,6,4,3,5,6,5,3])

print("Mean:", d.mean())

print("Median:",d.median())

print("Mode:",d.mode())
# Write your code for Q3 and execute the cell

df2 = pd.read_csv("../input/chickwts.csv")

df2.head(10)
wt = df2["weight"]

display("Mean:",wt.mean())

display("Median:",wt.median())

display("80% Value:",wt.quantile(0.8))

display("Weight description:",wt.describe())
fd = df2["feed"]

print("Value Count:\n",fd.value_counts())

print("------------------------------------")

print("Proportion:\n",fd.value_counts()/fd.size)

print("------------------------------------")

print("Percentages:\n",fd.value_counts()/fd.size*100)
sns.countplot(fd)
# Write your code for Q4 and execute the cell

df1 = pd.read_csv("../input/quakes.csv")

print("Summary for column mag:\n", df1["mag"].describe())
print("Summary for dataset:\n", df1.describe())
# Write your code for Q5 and execute the cell

df3 = pd.read_csv("../input/orange.csv")

df4 = df3[["age", "circumference"]]

print(df4)

print("-----------------------------------")

print("Correlation:\n", df4.corr())
plt.scatter(df4["age"], df4["circumference"],alpha=0.50)

plt.title('Scatter Plot',fontsize=16)

plt.xlabel('Age')

plt.ylabel('Circumference')

plt.show()
# Write your code for Q6 and execute the cell

df5 = pd.read_csv("../input/mtcars.csv")

df5.head()
cf = df5["cyl"].value_counts()

cd = cf.apply(pd.Series)

cd.columns = ["freq"]

print(cd)
plt.bar(cd.index,cd["freq"])

plt.show()
# Write your code for Q7 and execute the cell

plt.pie(cd["freq"], labels=cd.index, startangle=90)

plt.show()
# Write your code for Q8 and execute the cell

df3 = pd.read_csv("../input/orange.csv")

df4 = df3[["age"]]

df4["age"].value_counts()
plt.boxplot(df4["age"])

plt.show()
# Write your code for Q9 and execute the cell

df6 = df3[["age", "circumference"]]

df6.head()
sns.boxplot(

    x = "age",

    y = "circumference",

    data=df6

)