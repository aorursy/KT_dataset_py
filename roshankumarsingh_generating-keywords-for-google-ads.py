# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# List of words to pair with products

words = ['buy', 'price', 'discount', 'promotion', 'promo', 'shop', 

         'buying', 'prices', 'pricing', 'shopping', 'discounts', 

         'promos', 'ecommerce', 'e commerce', 'buy online',

         'shop online', 'cheap', 'best price', 'lowest price',

         'cheapest', 'best value', 'offer', 'offers', 'promotions',

         'purchase', 'sale', 'bargain', 'affordable',

         'cheap', 'low', 'budget', 'inexpensive', 'economical','amazon','e-commerce']



# Printing list of words

print(words)
products = ['sofas', 'convertible sofas', 'love seats', 'recliners', 'sofa beds']



# an empty list

keywords_list = []



# Looping through products

for product in products:

    # Looping through words

    for word in words:

        # Appending combinations

        keywords_list.append([product, product + ' ' + word])

        keywords_list.append([product, word + ' ' + product])

        

# Inspecting keyword list

from pprint import pprint

pprint(keywords_list)


import pandas as pd



# Creating a DataFrame from list

keywords_df = pd.DataFrame.from_records(keywords_list)



# Printing the keywords DataFrame to explore it

keywords_df.head()
# Renaming the columns of the DataFrame

keywords_df = keywords_df.rename(columns={0: "Ad Group",1: "Keyword"})

keywords_df.head()
# Adding a campaign column

keywords_df['Campaign']='SEM_Sofas'
# Adding a criterion type column

keywords_df['Criterion Type']='Exact'
# Making a copy of the keywords DataFrame

keywords_phrase = keywords_df.copy()



# Changing criterion type match to phrase

keywords_phrase['Criterion Type']='Phrase'



# Appending the DataFrames

keywords_df_final = keywords_df.append(keywords_phrase)
# Saving the final keywords to a CSV file

keywords_df.to_csv('keyords.csv', index=False)

# Viewing a summary of our campaign work

summary = keywords_df_final.groupby(['Ad Group', 'Criterion Type'])['Keyword'].count()

print(summary)