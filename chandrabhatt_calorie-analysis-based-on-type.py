# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/cereal.csv")
data.describe()
data.head()
import matplotlib.pyplot as plt
# list all the coulmn names
print(data.columns)

# get the calories column
calories = data["calories"]

# Plot a histogram of claories content in the cereals
plt.hist(calories, bins=10, edgecolor = "red")
plt.title("Calories in our favourite Cereal Brands") # add a title
plt.xlabel("Calories in kCal") # label the x axes 
plt.ylabel("Count")
cereal_names=data['name']
cereal_names
#we will generate an array of length of label and use it on X-axis
index = np.arange(len(cereal_names))
no_calories=data['calories']
def plot_bar_x():
    # this is for plotting purpose
    plt.bar(index, no_calories)
    plt.xlabel('Cereals', fontsize=30)
    plt.ylabel('No of Calories', fontsize=30)
    plt.xticks(index, cereal_names, fontsize=15, rotation=90)
    plt.title('Calorie content for each name Cereal Brand', fontsize=30)
    plt.show()
import matplotlib.font_manager as fm

fig = plt.figure(figsize=(50,20))
fontprop = fm.FontProperties(size=30)
ax = fig.add_subplot(111)
plot_bar_x()
ax.legend(loc=0, prop=fontprop) 
from scipy.stats import ttest_ind # just the t-test from scipy.stats
from scipy.stats import probplot # for a qqplot
import pylab 
# check out the first few lines
data.head()
#There are two types of cereals 'Hot' = 'H' and 'Cold'='C'
data['type'].unique()
# plot a qqplot to check normality. If the varaible is normally distributed, most of the points 
# should be along the center diagonal.
probplot(data["calories"], dist="norm", plot=pylab)
# get the sodium for hot cerals
hotCereals = data["calories"][data["type"] == "H"]
# get the sodium for cold ceareals
coldCereals = data["calories"][data["type"] == "C"]

# compare them
ttest_ind(hotCereals, coldCereals, equal_var=False)
# let's look at the means (averages) of each group to see which is larger
print("Mean calorie content for the hot cereals:")
print(hotCereals.mean())

print("Mean calorie content for the cold cereals:")
print(coldCereals.mean())

fig = plt.figure(figsize=(25,5))
fontprop = fm.FontProperties(size=30)
ax = fig.add_subplot(111)
# plot the cold cereals
plt.hist(coldCereals, alpha=0.5, label='cold') 
# and the hot cereals
plt.hist(hotCereals, label='hot')
# and add a legend
plt.legend(loc='upper right')
calories = data["calories"]
plt.xlabel("Calories in kCal") # label the x axes 
plt.ylabel("Count")
# add a title
plt.title("Calorie content of cereals by type")
ax.legend(loc=0, prop=fontprop) 
print(data[data["type"] == "C"])
