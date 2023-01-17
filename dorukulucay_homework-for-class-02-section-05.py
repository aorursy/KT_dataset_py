import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

data = pd.read_csv("../input/pokemon.csv")
data.head()
data.tail()
data.columns
data.shape
data.info()
data["Type 1"].value_counts()
serie = pd.DataFrame([9, 5, 6, 1, 3, 8, 12, 100, 11])
# sorted it is: 1 3 5 6 8 9 11 12 100
serie.describe()
import matplotlib.pyplot as plt

data.boxplot(column="HP")
plt.show()
summary = data.head()
summary
melted = pd.melt(frame=summary, id_vars="Name", value_vars = ["Attack", "Defense"])
melted
melted.pivot(index='Name', columns='variable', values='value')
concated = pd.concat([data.head(), data.tail()], axis=0, ignore_index=True)
concated
concated_h = pd.concat([data["Attack"].head(), data["Defense"].head()], axis=1)
concated_h
# here we fill all the empty columns in data with string 'empty'
data.fillna('empty')
# here we drop every row that contains any empty cell
data.dropna()
# in python, assertion returns nothing if it passes.
try:
    assert 1 == 1
    print("Assertion passed")
except AssertionError as e:
    print("ASSERTION FAILED")
# it throws AssertionError if it does not pass

#i use try-catch so jupyter wont stop on assertion error
try:
    assert 1 == 2
except AssertionError as e:
    print("ASSERTION FAILED")

dt = pd.DataFrame({"Nums":[1,2,3,None,5,6,7]})

# returns if serie does not contain any null values. will return false
dt["Nums"].notnull().all()
# lets drop null values
dt.Nums.dropna(inplace=True)
dt
# and assert if there is no null value remaining after dropping
try:
    assert dt.Nums.notnull().all()
    print("Assertion passed")
except AssertionError as e:
    print("ASSERTION FAILED")
# Passed!