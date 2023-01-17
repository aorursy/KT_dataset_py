# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from scipy.stats import probplot
import pylab
import scipy.stats
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
train.head()
train.describe()
# Plot a histogram of sodium content
plt.hist('Pclass', data = train)
plt.title("Passenger_Class")

# Plot a histogram of sodium content with nine bins, a black edge 
# around the columns & at a larger size
ticket_fare = train.Fare
plt.hist(ticket_fare, bins=5, edgecolor = "black")
plt.title("Ticket Fare") # add a title
plt.xlabel("Fare price") # label the x axes 
plt.ylabel("Count") # label the y axes
train.hist(column= 'Fare', figsize = (12,12))
test = pd.read_csv('../input/test.csv')
test.head()
test.describe()
# plot a qqplot to check normality. If the varaible is normally distributed, most of the points 
# should be along the center diagonal.
probplot(train["Fare"], dist="norm", plot=pylab)
# get the sodium for hot cerals
female = train["Fare"][train["Sex"] == "female"]
# get the sodium for cold ceareals
male = train['Fare'][train['Sex'] == 'male']

# compare them
ttest_ind(female, male, equal_var=False)
# let's look at the means (averages) of each group to see which is larger
print("Mean fare for the female:")
print(female.mean())

print("Mean fare for the male:")
print(male.mean())
# plot the cold cereals
plt.hist(female, alpha=0.5, label='F')
# and the hot cereals
plt.hist(male, label='M')
# and add a legend
plt.legend(loc='upper right')
# add a title
plt.title("Fare between male and female")
sns.countplot(train['Pclass']).set_title('Passenger Class')
sns.countplot(train.Embarked)
scipy.stats.chisquare(train["Sex"].value_counts())
scipy.stats.chisquare(train["Pclass"].value_counts())
contingencyTable = pd.crosstab(train.Sex, train.Pclass)
scipy.stats.chi2_contingency(contingencyTable)
