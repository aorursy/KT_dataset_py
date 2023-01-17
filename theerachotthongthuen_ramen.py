# Import file and packages
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt     # for visualisation
import seaborn as sns     # for visualisation
from wordcloud import WordCloud    # for create word cloud
import random    # for use in random color in word cloud

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

ramen_data = pd.read_csv('/kaggle/input/ramen-ratings/ramen-ratings.csv')
ramen_data.Country.value_counts()
# View head
ramen_data.head(11)

ramen_data= pd.read_csv('/kaggle/input/ramen-ratings/ramen-ratings.csv')
listbrand  = ramen_data["Brand"]
print(listbrand)
ramen_data.groupby('Country')['Brand'].max()
ramen_data.Country.value_counts()
# Count the amount of brand that got review
ramen_brand = ramen_data.groupby(['Brand','Country']).agg({'Review #':'count'})
ramen_brand = ramen_brand.reset_index() 
ramen_brand = ramen_brand.sort_values('Review #', ascending = False)
# Count brand from each country that got review
ramen_coun = ramen_brand.groupby('Country').agg({'Brand':'count'}).reset_index()
ramen_coun = ramen_coun.rename(columns = {'Brand':'Amount of brand'})
ramen_coun = ramen_coun.sort_values(['Amount of brand', 'Country'], ascending = [False, True])
# View the top 10 countries which have the most amount of ramen brand
ramen_coun.head(10)
# Bar chart of the amount of ramen brands in each country that got review
plt.figure(figsize=(15, 5))
plt.bar('Country', 'Amount of brand', data = ramen_coun, color = 'gold')
plt.title( 'The amount of ramen brands in each country', fontsize=14)
plt.ylabel('Number of brands')
plt.xticks(rotation = 90)
plt.show()
# Present the variety of each countries that got reviewed
ramen_variety = ramen_data.groupby(['Country']).agg({'Variety':'count'})
ramen_variety = ramen_variety.reset_index() 
ramen_variety = ramen_variety.sort_values(['Variety','Country'], ascending = [False, True])
ramen_variety = ramen_variety.rename(columns = {'Variety': 'Country variety'})
# Bar chart of the amount of ramen products in each country that got reviewed
plt.figure(figsize=(15, 5))
plt.bar('Country', 'Country variety', data = ramen_variety, color = 'peru')
plt.title( 'The amount of ramen product in each country', fontsize=14)
plt.ylabel('Number of product')
plt.xticks(rotation = 90)
plt.show()
# Rank ramen by Stars column
ramen_sort = ramen_data.sort_values('Stars').dropna(subset = ['Stars'])

# Split into top 100 and bottom 100
ramen_top = ramen_sort.head(100)
ramen_bottom = ramen_sort.tail(100)
ramen_bottom.head(10)