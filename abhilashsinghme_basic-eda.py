# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Lets import all the libraries



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# Reading the tables



resto = pd.read_csv('/kaggle/input/zomato-restaurants-hyderabad/Restaurant names and Metadata.csv')

review = pd.read_csv('/kaggle/input/zomato-restaurants-hyderabad/Restaurant reviews.csv')
resto.head()
review.head()
# Joining the two tables for further analysis



df = pd.merge(resto, review, left_on = 'Name', right_on = 'Restaurant', how = 'left')

df.head()
# Removing unwanted columns



df.drop(['Restaurant', 'Links'], axis = 1, inplace = True)

df.head()
cost_df = df[['Name', 'Cost']].drop_duplicates()

cost_df
cost_df.info()
# Cost seems to be a non-null object.



# Convert it into float.



cost_df['Cost'] = cost_df['Cost'].apply(lambda x : x.replace(',',''))

cost_df['Cost'] = cost_df['Cost'].astype('float')
# Probabality Mass Function (PMF)



def PMF(data):

    

    return data.value_counts().sort_index()/len(data)





pmf = PMF(cost_df['Cost'])





# Plotting PMF

pmf.plot.bar()

plt.xlabel('Cost')

plt.ylabel('PMF')
# Lets define a function for ECDF - Cumulative Density Function



def ecdf(data):

    

    y = (np.arange(1, len(data) + 1))/len(data)

    x = np.sort(data)

    plt.plot(x,y, marker = '.', linestyle = 'none')

    



ecdf(cost_df['Cost'])
# What are the available cuisines in Hyderabad's restaurants?



Cuisines = df[['Name', 'Cuisines']].drop_duplicates()





# Lets make a dictionary for the unique cuisines in the dataset



cuisines_dict = {}

for cuisine in Cuisines['Cuisines']:

    for name in cuisine.split(', '):

        if name in cuisines_dict:

            cuisines_dict[name] = cuisines_dict[name] + 1

        else:

            cuisines_dict[name] = 1



cuisines_dict
# Lets plot the top 10 available cuisines



cuisine_df = pd.DataFrame(cuisines_dict.items(), columns = ['Cuisines', 'Count']).sort_values(['Count'], ascending = False).head(10)





chart = sns.barplot(x = 'Cuisines', y = 'Count', data = cuisine_df, palette = 'OrRd')

chart.set_xlabel('Cuisines')

chart.set_ylabel('Number of Availability')

chart.set_xticklabels(chart.get_xticklabels(),

                      rotation = 45)