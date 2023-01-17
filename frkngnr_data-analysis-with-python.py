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
import matplotlib.pyplot as plt 

%matplotlib inline
import pandas as pd 
data = pd.read_csv("../input/winemag-data_first150k.csv", index_col = 0)

data.head()  ## to see first 5 rows.
data.shape ## to get the shape of our dataframe
data.info()   ## to gather information about dataframe
data.describe() # to gather information about columns that have numeric data types.
data.isnull().sum()  ## to get the count of NA values in each column.
## I realized that both country and province columns have 5 Na values and I want to investigate further to check whether they are the same rows or not.
data[data.country.isna() == True]  
# Yes, indeed. I decided to remove these rows.
data[(data.country.isna() == True) & (data.province.isna() == True)]  ##boolean filter for Na values in country and province.
data_clean = data[~((data.country.isna() == True) & (data.province.isna() == True))]   ##select desired rows.
data_clean.shape  # to assert that we removed only 5 rows.
data_clean.columns   ## lets dive more for further investigation.
##I would like to understand how many countries existing in this dataset.
data_clean.country.nunique()
##Lets check how many different types of wine existing in this dataset.
data_clean.variety.nunique()
## Question 1: What are the best 10 wine varieties in terms of points they got from reviewers.
data_clean.groupby("variety")["points"].mean().sort_values(ascending = False).head(10)
# lets visualize best wine varieties in a bar plot which is good for different categories.
s = data_clean.groupby("variety")["points"].mean().sort_values(ascending = False).head(10)
s.plot.bar()
plt.xlabel("wine types")
plt.ylabel("points")
plt.title("10 best wine types")
plt.tight_layout()
plt.ylim(80,100)   ## since points are only in that range, I chose starting value of y axis as 80 and obviously 100 is the highest they can get.
plt.show()
## Question 2 : What are the top 8 best performing countries in terms of wine points ? 
data_clean.groupby("country")["points"].mean().nlargest(8)
#I want to further investigate how many wine reviews each country got ? 
data_clean.country.value_counts()

##I realized that England has only 9 wine reviews and is the best performing country in terms 
#of points it got. Other good performing countries such as  France, Italy have more than 20000 wines reviewed 
# so I decided to drop England from that list since they have relatively few samples.
winner_countries = data_clean.groupby("country")["points"].mean().nlargest(9)[1:]
winner_countries
data_clean.groupby("country")["price"].mean().sort_values(ascending = False).head(5)
## I realized that there is a mistake in country column which has "US-France" value and interestingly 
# it has the top mean value so I decided to remove that row from the data, similary since England has 9 
#analyzed samples I decided to not pick England as the most expensive wine country : 
data_clean.loc[data_clean.country == "US-France"].index

data_clean = data_clean.drop(144054, axis = 0)
data_clean.country.value_counts()
## since England has 9 
#reviewed samples I decided to not pick England as the most expensive wine country : 
data_clean.groupby("country")["price"].mean().sort_values(ascending = False).head(6)[1:6]
##lets make a plot out of it to better understand price differences in average.
average_price_countries = data_clean.groupby("country")["price"].mean().sort_values(ascending = False).head(6)[1:6]

average_price_countries.plot.bar()
plt.xlabel("countries")
plt.ylabel("average wine price")
plt.title("most expensive wine countries")
plt.tight_layout()
plt.show()

## How many wines got 100 points out of 100 ? How many different wine types involved in this list ? 
data_clean[data_clean.points == 100].count()
## There are 24 wines that got 100 points out of 100. Lets check how many different types are there :
data_clean[data_clean.points == 100]["variety"].nunique()

##Interesting ! There are only 10 types of wine that got 100 points out of 100. Lets dive to find which type 
# got 100 points most ? 
data_clean[data_clean.points == 100]["variety"].value_counts()

## Final question of this kernel : Is there a relationship between price and the point that wine got ? 
## to better understand this relationship, we must use scatter plots which are best for understanding two numeric columns relation.
##before plotting I would like to find correlation coefficient : 
data_clean["price"].corr(data_clean["points"])

# we can say that they are moderately correlated. There is no strong correlation. Lets check this by looking plot :
plt.scatter(x = data_clean.price , y = data_clean.points, c = "g", marker = ".")
plt.xlabel("price")
plt.ylabel("points")
plt.title("price vs points")
plt.tight_layout()
plt.show()

##lets make this plot more fancy by showing higher prices with bigger points : 
plt.scatter(x = data_clean.price , y = data_clean.points, c = "g", marker = ".", s = data_clean.price)
plt.xlabel("price")
plt.ylabel("points")
plt.title("price vs points")
plt.tight_layout()
plt.show()
## from second graph price vs points relationship can be better understand but still it is not strongly correlated 
## since correlation coefficient is smaller than 0.7 which is  usually minimum border for a strong correlation.