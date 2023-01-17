# Importing required packages

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from wordcloud import WordCloud

import random 
def overview():

        

    data = pd.read_csv('../input/ramen-ratings/ramen-ratings.csv')

    # Print the first 5 lines of data

    print("First 5 lines of data \n\n")

    print(data.head())



    # Print data type

    print("\n\n\nDatatype\n")

    print(data.dtypes)



    # Print number of null values 

    print("\n\n\nNumber of null values\n")

    print(data.isnull().sum())



    # Print data summary

    print("\n\n\nData summary\n")

    print(data.describe())



    # Print data shape

    print("\n\n\nData shape\n")

    print("Data has {} rows and {} columns".format(data.shape[0], data.shape[1]))



    return data



data = overview()
# Dropping 'Top Ten' column 

data.drop(columns = ['Top Ten'], inplace = True)



# Convert 'Stars' column to numeric

data['Stars'] = pd.to_numeric(data['Stars'],errors='coerce')



# Identify the NaN values in 'Style' column and replace them with the right style

data[data['Style'].isna()]
# Replace NaN with 'Pack'

data['Style'].fillna('Pack', inplace = True)



# Check if NaN is replaced

target = ['E Menm Chicken', '100 Furong Shrimp']

data.loc[data['Variety'].isin(target)]
top_reviews = data['Brand'].value_counts().head(10)

print(top_reviews)



# Using visualisation

sns.countplot(y="Brand", data=data, palette="Oranges_r",

              order=data.Brand.value_counts().iloc[:10].index)
ramen_coun = data.groupby('Country').agg({'Brand':'count'}).reset_index()

ramen_coun = ramen_coun.rename(columns = {'Brand':'Amount of brand'})

ramen_coun = ramen_coun.sort_values(['Amount of brand', 'Country'], ascending = [False, True])

print(ramen_coun)

# Visualising

sns.barplot(y="Country", x = 'Amount of brand', data=ramen_coun, palette="Blues_r",

              order=data.Country.value_counts().iloc[:10].index)
top_style = data['Style'].value_counts()

top_style
# Rank ramen by Stars column

ramen_sort = data.sort_values('Stars', ascending = False).dropna(subset = ['Stars'])



# Showing top 100 

ramen_top = ramen_sort.head(100)

ramen_top
# Join the top 100 ramen product name into a string

ramen_top_str = ramen_top['Variety'].str.cat(sep=',')



# For generate color

def color_func(word, font_size, position, orientation, random_state=None,\

                    **kwargs):

    return "hsl(%d, 100%%, 60%%)" % random.randint(20, 55)



# Plot word cloud of the top 100

stopword_list = ['Noodle', 'Noodles', 'Instant Noodle', 'Instant', 'Flavor', 'Flavour', 'Ramen', 'With']

plt.figure(figsize=(10,6))

top_wordcloud = WordCloud(max_font_size= 50, background_color='black', \

                      prefer_horizontal = 0.7, stopwords = stopword_list).generate(ramen_top_str)

plt.imshow(top_wordcloud.recolor(color_func = color_func, random_state = 3), interpolation='bilinear')

plt.axis('off')

plt.show()