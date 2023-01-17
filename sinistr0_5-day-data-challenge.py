import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from scipy.stats import ttest_ind

from scipy.stats import probplot # for a qqplot

import seaborn as sns

import pylab

import os

print(os.listdir("../input"))

data = pd.read_csv("../input/database.csv")
data.head()
data.describe()
plt.hist(data['Incident Year'])

plt.title('Accidents Per Year') 
probplot(data["Incident Month"], dist="norm", plot=pylab)
jan = data["Incident Month"][data["Incident Month"] == 1]

feb = data["Incident Month"][data["Incident Month"] == 2]

mar = data["Incident Month"][data["Incident Month"] == 3]

apr = data["Incident Month"][data["Incident Month"] == 4]

may = data["Incident Month"][data["Incident Month"] == 5]

jun = data["Incident Month"][data["Incident Month"] == 6]

jul = data["Incident Month"][data["Incident Month"] == 7]

aug = data["Incident Month"][data["Incident Month"] == 8]

sep = data["Incident Month"][data["Incident Month"] == 9]

octo = data["Incident Month"][data["Incident Month"] == 10]

nov = data["Incident Month"][data["Incident Month"] == 11]

dec = data["Incident Month"][data["Incident Month"] == 12]

plt.hist(jan, alpha=0.5, label='Jan')

plt.hist(feb, label='Feb')

plt.hist(mar, label='Mar')

plt.hist(apr, label='Apr')

plt.hist(may, label='May')

plt.hist(jun, label='Jun')

plt.hist(jul, label='Jul')

plt.hist(aug, label='Aug')

plt.hist(sep, label='Sep')

plt.hist(octo, label='Oct')

plt.hist(nov, label='Nov')

plt.hist(dec, label='Dev')

plt.legend(loc='upper left')

plt.title('Incidents per Month')
sns.countplot(data["Incident Month"]).set_title("Incidents Per Month")
sns.countplot(data["State"], order=data["State"].value_counts().iloc[:20].index).set_title("Incidents Per State (Top 20)")

plt.xticks(rotation=90)
sns.countplot(data["Species Name"], order=data["Species Name"].value_counts().iloc[:20].index).set_title("Bird Species (Top 20)")

plt.xticks(rotation=90)