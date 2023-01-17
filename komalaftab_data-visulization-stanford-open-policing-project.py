import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
p_d = pd.read_csv("../input/stanford-open-policing-project/police_project.csv")

                  # police data
p_d.head() # by default return the first 5 rows
p_d.tail() # by default return the last 5 rows
p_d.shape  # shape attributes represent the rows and column of the data
p_d.isnull() # detecting missing values

p_d.isnull().sum() # .sum() calculates the sum of each column

# True = 1 ,False =0
p_d.drop("county_name", axis = 1, inplace = True)

#p_d.dropna(axis="columns",how ="all" or how ="any" ,inlpace =True)
p_d.shape
p_d.columns
p_d[p_d.violation=="Speeding"].driver_gender.value_counts()
p_d[p_d.driver_gender == "M"].violation.value_counts(normalize=True)
p_d[p_d.driver_gender == "F"].violation.value_counts(normalize=True)
a=p_d.groupby(["driver_gender"]).violation.value_counts(normalize=True).unstack()
a
plt.figure(figsize=(12, 8))

a.Speeding.plot(kind="bar")
p_d.stop_date
p_d['stop_date'] = pd.to_datetime(p_d.stop_date, format="%Y-%M-%d")

p_d["year"] = p_d.stop_date.dt.year
p_d.dtypes


p_d.year.value_counts()


#plt.figure(figsize=(8, 8))

p_d.year.value_counts().plot(kind="bar")