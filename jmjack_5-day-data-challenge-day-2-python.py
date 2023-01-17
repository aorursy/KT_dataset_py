import pandas as pd

import matplotlib.pyplot as plt

import re
data = pd.read_csv("../input/salaries-by-college-type.csv")

data.dropna(inplace=True)

data.reset_index(inplace=True, drop=True)

data.head()
for column in range(2,8):

    data.iloc[:, column] = [float(re.sub('[$,.]', '', i)[:-2]) for i in data.iloc[:, column]]
data
plt.hist(data['Starting Median Salary'])

plt.title("Histogram of Starting Median Salary")
plt.hist(data['Mid-Career Median Salary'], color='navy')

plt.title("Histogram of Mid-Career Median Salary")
plt.hist(data['Mid-Career 10th Percentile Salary'], color='lightskyblue')

plt.title("Histogram of Mid-Career 10th Percentile Salary")
plt.hist(data['Mid-Career 25th Percentile Salary'], color='blue')

plt.title("Histogram of Mid-Career 25th Percentile Salary")
plt.hist(data['Mid-Career 75th Percentile Salary'], color='g')

plt.title("Histogram of Mid-Career 75th Percentile Salary")
plt.hist(data['Mid-Career 90th Percentile Salary'], color='r')

plt.title("Histogram of Mid-Career 90th Percentile Salary")