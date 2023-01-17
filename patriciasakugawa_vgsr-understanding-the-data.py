# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Load the file "Video Game Sales with Ratings"
data = pd.read_csv("../input/Video_Games_Sales_as_at_22_Dec_2016.csv")

print("The header of the 'Video Game Sales with Ratings' is shown below:")
print(data.head())
# Let's analyze the data grouping it by the year of release
# The sales are normalized with respect to the total of global sales

year = data.Year_of_Release.dropna().unique()
data_year = data.groupby("Year_of_Release").sum()

global_sales = data_year.Global_Sales
na_sales = data_year.NA_Sales
eu_sales = data_year.EU_Sales
jp_sales = data_year.JP_Sales
other_sales = data_year.Other_Sales
sales = pd.DataFrame({'North America': na_sales/global_sales.sum(),
                      'Europe': eu_sales/global_sales.sum(),
                      'Japan':jp_sales/global_sales.sum(),
                      'Other': other_sales/global_sales.sum(),
                      'Global': global_sales/global_sales.sum()});
plt.subplot(121)
plt.title('Video Game Unit Sales Normalized per Year')
plt.xlabel('Year')
plt.ylabel('Normalized Unit Sales')
plt.scatter(year,sales['North America'], label = 'North America')
plt.scatter(year,sales['Europe'], label = 'Europe')
plt.scatter(year,sales['Japan'], label = 'Japan')
plt.scatter(year,sales['Other'], label = 'Other')
plt.legend()
plt.axis([1978, 2022, 0, 0.055])

notna_sales = eu_sales + jp_sales + other_sales
plt.subplot(122)
plt.xlabel('Year')
plt.scatter(year,sales['North America'], label = 'North America')
plt.scatter(year,notna_sales/global_sales.sum(), label = 'Others')
plt.legend()
plt.axis([1978, 2022, 0, 0.055])
plt.show()

# North America is responsible for around half of the sales of the world.
# We can see that the relationship between critics and users are approximatly linear
plt.subplot(121)
plt.title('Relationship Between Critic and User with Global Sales')
plt.ylabel('Global Unit Sale')
plt.xlabel('User')
user_year = data_year.User_Count
plt.scatter(user_year/user_year.sum(),global_sales)
plt.axis([0, 0.15, 0, 800])
plt.subplot(122)
plt.xlabel('Critic')
critic_year = data_year.Critic_Count
plt.scatter(critic_year/critic_year.sum(),global_sales)
plt.axis([0, 0.15, 0, 800])
# Filter the Top 10 Developers with most Global Sales
data_developer = data.dropna().groupby("Developer").sum()
dps = data_developer.Global_Sales.sort_values()[-10:].reset_index()
plt.barh(dps.Developer,dps.Global_Sales)
plt.title('Top 10 Global Sales')
plt.ylabel('Developer')
plt.xlabel('Global Sales')
# Developers with the highest  mean score from critics
dev_score = data_developer.Critic_Score
dev_count = data.dropna().groupby("Developer").Name.count().rename('Count')
dev_score_count = (dev_score/dev_count).rename('Critic_Mean_Score')
dps2 = dev_score_count.sort_values()[-10:].reset_index()
plt.barh(dps2.Developer,dps2.Critic_Mean_Score)
plt.title('Top 10 Mean Score')
plt.ylabel('Developer')
plt.xlabel('Mean Score')
# Relationship between Sales and Mean Score by Developer
plt.title('Mean Global Sales x Mean Score')
plt.subplot(121)
X1 = data_developer.Global_Sales.sort_values()[-10:]/dev_count[dps.Developer]
plt.scatter(X1,dev_score_count[dps.Developer])
plt.ylabel('Score')
plt.xlabel('Global Sales')
plt.axis([0, 8, 0, 100])
plt.subplot(122)
X2 = data_developer.Global_Sales[dps2.Developer]/dev_count[dps2.Developer]
plt.scatter(X2,dps2.Critic_Mean_Score)
plt.axis([0, 8, 0, 100])