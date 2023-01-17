first5 = train.head(5)
first5.describe(include="all")
def missing_values(x):
    return sum(x.isnull())

print ("Missing values per column:")
print (train.apply(missing_values, axis=0))

print("Missing values per row:")
print(train.apply(missing_values, axis=1))

train1 = train.head(10)
train1.corr()
cats = train.groupby("Category")["Category"].count()
cats = cats.sort_values(ascending=0)
plt.figure()
cats.plot(kind='bar', title="Category Count")
print(cats)
# Largest category is LARCENY/THEFT, let's investigate it further
larceny = train[train["Category"] == "LARCENY/THEFT"]
groups = larceny.groupby("DayOfWeek")["Category"].count()
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
groups = groups[weekdays]
plt.figure()
groups.plot(kind="bar", title="LARCENY/THEFT per weekday")
print(groups)
# Lets group Category by Day of Week
category = train.groupby("Category")["Category"].count()
category.plot(kind="bar", title="Category Count")
catDistrict = train.groupby("PdDistrict")["PdDistrict"].count()
catDistrict.plot(kind="bar", title="PdDistrict vs Crime Category")
train.describe()

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
