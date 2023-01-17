# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.simplefilter(action = "ignore", category = FutureWarning)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
blackFriday = pd.read_csv("../input/BlackFriday.csv")
blackFriday.head(5)
# Setting the plot size and style 
plt.figure(figsize=(24,20))

sns.set_style("darkgrid")
# Number of Females and Males who participated in the Black Friday Sales
sns.countplot(blackFriday.Gender)
# Get Percentage of Male and Female who participated in the sale
explode = (0.1,0)  # Slices out the first slice in the pie

ax1 = plt.subplot()

ax1.pie(blackFriday['Gender'].value_counts(), explode=explode,labels=['Male','Female'], autopct='%1.1f%%',

        shadow=True, startangle=90)

# Equal aspect ratio ensures that pie is drawn as a circle

ax1.axis('equal')  

plt.tight_layout()

plt.legend()
# Now that we understand that number of Men who participated in the Black Friday sales is much higher

# It may be safe to assume that Men spent more.

# However lets prove it with a visualisation.
def spendByGroup(group, column, plot):

    blackFriday.groupby(group)[column].sum().sort_values().unstack().plot(kind=plot, stacked = True)

    plt.ylabel(column)
group = ('Age', 'Gender')

spendByGroup(group, 'Purchase', 'bar')
# We should also check how marriage affected Men and Women in making a decision about their purchases.
group = ('Marital_Status', 'Gender')

spendByGroup(group, 'Purchase', 'bar')
# We should also check how marriage and age have influenced people in making a decision about their purchases.
group = ('Age', 'Marital_Status')

spendByGroup(group, 'Purchase', 'bar')
# Lets check which category of cities spent the most
blackFriday.City_Category.unique()
sns.countplot(blackFriday['City_Category'], hue=blackFriday['Gender'])
def sliceOfGroup(column, explode, labels):

    ax1 = plt.subplot()

    ax1.pie(blackFriday[column].value_counts(), explode=explode,labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

    # Equal aspect ratio ensures that pie is drawn as a circle

    ax1.axis('equal')  

    plt.tight_layout()

    plt.legend()
explode = (0.1, 0, 0)

labels = blackFriday.City_Category.unique()
sliceOfGroup('City_Category', explode, labels)
# Spend grouped by Age for the City Category

group = ('Age', 'City_Category')

spendByGroup(group, 'Purchase', 'bar')
# Spend grouped by Marital_Status for the City Category

group = ('City_Category', 'Marital_Status')

spendByGroup(group, 'Purchase', 'bar')
# Spending by product Category grouped by Age, Gender, Marital Status and City Category
# Spending for Product_Category_1 grouped by Age

group = ('Product_Category_1', 'Age')

spendByGroup(group, 'Purchase', 'bar')
# Spending for Product_Category_1 grouped by Gender

group = ('Product_Category_1', 'Gender')

spendByGroup(group, 'Purchase', 'bar')
# Spending for Product_Category_1 grouped by Marital_Status

group = ('Product_Category_1', 'Marital_Status')

spendByGroup(group, 'Purchase', 'bar')
# Spending for Product_Category_1 grouped by City_Category

group = ('Product_Category_1', 'City_Category')

spendByGroup(group, 'Purchase', 'bar')