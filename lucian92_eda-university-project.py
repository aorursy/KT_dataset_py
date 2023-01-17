# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.graph_objs as gro
from plotly import tools
import chart_studio.plotly as ply


# encoder for the nominal categorical values
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

data = pd.read_csv("/kaggle/input/google-play-store-apps/googleplaystore.csv")
data.head()
data.shape
data.info()
#we see that although there are 10841 rows to the dataset, there are null values amongst
#we want to reduce in order to be able to use the data for prediction
# missing data
total = data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum()/data.count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent * 100], axis=1, keys=['Total', 'Percent'])
missing_data.head(15)
cleaned_data = data.dropna()

# cleaned data
total2 = cleaned_data.isnull().sum().sort_values(ascending=False)
percent2 = (cleaned_data.isnull().sum()/cleaned_data.count()).sort_values(ascending=False)
missing_data2 = pd.concat([total2, percent2], axis=1, keys=['Total', 'Percent'])
missing_data2.head(15)
# total duplicated entries by the "App" & "Current Ver" columns 1959
t_a_d = cleaned_data[cleaned_data.duplicated(['App', 'Current Ver'])]

print("Number of duplicates {}".format(t_a_d.shape))
print("Expected number of rows after cleanup: {}".format(cleaned_data.shape[0] - t_a_d.shape[0]))
# Return DataFrame with duplicate rows removed, optionally only considering certain columns. Indexes, including time indexes are ignored.
cleaned_data = cleaned_data.drop_duplicates(['App', 'Current Ver'])
print("Final table shape without duplicates in 'App' and 'Current Ver': __{}__".format(cleaned_data.shape))
# Just replace whitespaces with underscore in the headers
cleaned_data.columns = cleaned_data.columns.str.replace(' ', '_')
# display first 10 number of reviews
cleaned_data['Reviews'].head(10)
# Let's change each review to int
cleaned_data['Reviews'] = cleaned_data['Reviews'].apply(lambda review: int(review))
# display first 10 sizes
cleaned_data['Size'].head(10)
# Size Frequencies
cleaned_data.Size.value_counts().head(10) # 10 rows selected 
redundant_string = 'Varies with device'

print("There are a total of {} rows with the 'Size' column containing '{}' string.".format(cleaned_data.loc[cleaned_data['Size'] == redundant_string].shape[0], redundant_string))

# Let's replace the redundant string with 'NaN'
cleaned_data['Size'] = cleaned_data['Size'].apply(lambda x: str(x).replace(redundant_string, 'NaN') if redundant_string in str(x) else x)

print("There are a total of {} rows with the 'Size' column containing '{}' string.".format(cleaned_data.loc[cleaned_data['Size'] == redundant_string].shape[0], redundant_string))
cleaned_data['Size'] = cleaned_data['Size'].apply(lambda size: str(size).replace('M', '') if 'M' in str(size) else size)
cleaned_data['Size'] = cleaned_data['Size'].apply(lambda size: str(size).replace(',', '') if 'M' in str(size) else size)

# we remove k but we also divide it by 1000 to have a standardized column for the application sizes
cleaned_data['Size'] = cleaned_data['Size'].apply(lambda size: float(str(size).replace('k', '')) / 1000 if 'k' in str(size) else size)
cleaned_data['Size'] = cleaned_data['Size'].apply(lambda size: float(size))
# display first 10 total installs
cleaned_data['Installs'].head(10)
# Install Frequencies
cleaned_data.Installs.value_counts().head(10) # 10 rows selected 
# We remove the signs from the installs column

cleaned_data['Installs'] = cleaned_data['Installs'].apply(lambda x: x.replace('+', '') if '+' in str(x) else x)
cleaned_data['Installs'] = cleaned_data['Installs'].apply(lambda x: x.replace(',', '') if ',' in str(x) else x)
cleaned_data['Installs'] = cleaned_data['Installs'].apply(lambda x: int(x))
# display first 5 prices
print(cleaned_data['Price'].head(5))

# display first 5 prices with a value bigger than 0
print(cleaned_data[cleaned_data['Price'] != 0]['Price'].head(5))
paid_apps = cleaned_data[cleaned_data.Price != '0'].shape[0]
print("Total free apps {}".format(cleaned_data.shape[0]- paid_apps))
print("Total paid apps {}".format(paid_apps))
cleaned_data['Price'] = cleaned_data['Price'].apply(lambda x: str(x).replace('$', '') if '$' in str(x) else str(x))
cleaned_data['Price'] = cleaned_data['Price'].apply(lambda x: float(x))
# display first 5 genres
print(cleaned_data['Genres'].head(5))
result = cleaned_data[cleaned_data.Genres == 'Adventure;Action & Adventure']
genres = np.unique(cleaned_data['Genres'])
print("There are a total of {} initial genres".format(genres.shape))
# The idea is that we want to reduce redudant genres or combine 
# the ones which are of the same genre but are marked with different ones;
# More can be added / mixed together; There is no rule but it is about having some sort of a
# correct intuition which is needed in order to logically group them together;
new_genres = {
'Adventure;Action & Adventure'  :  'Action;Action & Adventure',
'Educational;Action & Adventure' : 'Action & Adventure',
'Adventure;Brain Games'   :  'Adventure',
'Adventure;Education'   : 'Adventure',
'Arcade;Pretend Play'   : 'Arcade',
'Art & Design;Pretend Play' : 'Art & Design;Creativity',
'Board;Pretend Play'  : 'Board;Brain Games',
'Books & Reference'  : 'Education',
'Communication;Creativity' : 'Communication',
'Educational;Education'   : 'Education',
'Educational' : 'Education',
'Educational;Brain Games': 'Education;Brain Games',
'Educational;Creativity': 'Education;Creativity',
'Educational;Pretend Play': 'Education;Pretend Play',
'Music;Music & Video' : 'Music',
'Lifestyle;Pretend Play': 'Lifestyle',
'Simulation;Education': 'Simulation',
'Simulation;Pretend Play' : 'Simulation' 
}

for old, new in new_genres.items():
    print("Replacing [{}] GENRE with [{}] GENRE".format(old, new))
    cleaned_data['Genres'] = cleaned_data['Genres'].apply(lambda x: x.replace(old, new) if old in str(x) else x)

# just checking the results here / nothing special
cleaned_data[cleaned_data.Genres == 'Art & Design;Creativity']
final_data = cleaned_data.copy()
installs = final_data['Installs'][final_data.Installs !=0]
reviews = final_data['Reviews'][final_data.Reviews !=0]
app_type = final_data['Type']
price = final_data['Price']

# we will use log to better represent the number of installs and reviews
p = sns.pairplot(pd.DataFrame(list(zip(final_data['Rating'], 
                                       final_data['Size'], 
                                       np.log(installs), 
                                       np.log10(reviews), 
                                       app_type, 
                                       price)), 
                columns=['Rating','Size', 'Installs', 'Reviews', 'Type', 'Price']), 
                hue='Type', palette="Set1")
plt.hist(final_data['Rating'], bins = 25)
plt.xlim((1, 6))
plt.show()
plt.figure()
redundant = cleaned_data.boxplot(['Rating'])
list_of_categories = final_data['Category'].unique().tolist()
fig, axes = plt.subplots(nrows = 17, ncols = 2)

fig.set_figheight(160)
fig.set_figwidth(40)

c = 0

for row in range(0, axes.shape[0]):
    for column in range(0, axes.shape[1]):
        if c == len(list_of_categories):
            break
        current_subplot = axes[row, column]
        category = list_of_categories[c]
        filtered = final_data.loc[final_data['Category'] == category]['Rating']
        current_subplot.hist(filtered)
        current_subplot.title.set_text(category)
        current_subplot.grid()
        c += 1

counts = cleaned_data['Category'].value_counts()
dict_counts=counts.to_dict()
z = plt.pie([float(v) for v in dict_counts.values()], labels=[str(k) for k in dict_counts],radius=5, autopct='%.2f')

final_data['Rating']
#gro.Histogram(final_data.Rating, xbins = {'start':1, 'size':0.1, 'end':5})

avg_rating = gro.Histogram(
    
    x=final_data.Rating,
    name='Average Rating',
    xbins = {'start': 1, 'size': 0.1, 'end' :5},
    marker=dict(
        color='#546FDE',
    ),
    opacity=0.75
)

fig = tools.make_subplots(rows=1, cols=1)
fig.append_trace(avg_rating, 1, 1)
fig.show()

print('Average Rating {}'.format(np.mean(final_data['Rating'])))
