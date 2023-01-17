import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import scipy.stats as stats

%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Reading the data 

goog_filepath = '../input/google-playstore-apps/Google-Playstore-32K.csv'

goog_data = pd.read_csv(goog_filepath)
# Initial look at the data

goog_data
# Size of the data

goog_data.shape
# Columns

goog_data.columns
# Types of each feature

goog_data.dtypes
#check for duplicates

print('The number of duplicated apps are {:n}'.format(goog_data.duplicated(keep='first').sum()))
# what is the duplicate

dup_app = goog_data[goog_data.duplicated(keep='first')]['App Name']

print('The duplicate app is {}'.format(dup_app.iloc[0]))
# remove duplicate from data

g_data = goog_data.drop_duplicates(keep='first').reset_index().drop('index',axis=1)
#check

g_data.duplicated().sum()
# Categories

g_data['Category'].value_counts()
# Find the 3 odd data points

odd_cat = [' Channel 2 News', ')',' Podcasts']

test = g_data['Category'].isin(odd_cat)

odd_cat_ind = []

for i in range(len(test)):

    if test[i] == True : 

        odd_cat_ind.append(i)

g_data.iloc[odd_cat_ind]
# correcting the false 3 data points

g_data.iloc[odd_cat_ind[0],1:7] = list(g_data.iloc[odd_cat_ind[0],4:10])

g_data.iloc[odd_cat_ind[0],7:] = np.nan

g_data.iloc[odd_cat_ind[1],1:9] = list(g_data.iloc[odd_cat_ind[1],2:10])

g_data.iloc[odd_cat_ind[1],9:] = np.nan

g_data.iloc[odd_cat_ind[2],1:9] = list(g_data.iloc[odd_cat_ind[2],2:10])

g_data.iloc[odd_cat_ind[2],9:] = np.nan

g_data.iloc[odd_cat_ind]
# Price data type

g_data['Price'].value_counts(normalize=True)
# Append a new feature (Free/Paid) to the dataset

if g_data['Price'].dtype == 'object' : 

    g_data['Price'] = g_data['Price'].apply(lambda x : x.strip('$')).astype(float)

free_paid = ['Free' if i == 0 else 'Paid' for i in g_data['Price']]

free_paid_ser = pd.Series(free_paid,name = 'Free/Paid')

g_data['Free/Paid'] = free_paid_ser

g_data
# Dropping 'Last Updated', 'Minimum Version', and 'Latest Version'

g_data = g_data.drop(['Last Updated','Minimum Version','Latest Version'],axis=1)
# Changing the 'Rating' and 'Reviews' data types

g_data['Rating'] = g_data['Rating'].astype(float)

g_data['Reviews'] = g_data['Reviews'].astype(int)
# Plot of the ratings

plt.figure(figsize=(8,8))

plt.title('Ratings distribution')

sns.distplot(g_data['Rating'],kde=True,color='orange',fit=stats.norm)

plt.legend(['Normal Distribution','Ratings',])
# Normality check for ratings distribution

ratings = g_data['Rating']

norm_rating = (ratings-ratings.mean())/ratings.std() # Normalize the ratings first

print('The p-value for Kolmogorov-Smirnov Test is {}'.format(stats.kstest(norm_rating,'norm',N = len(norm_rating)).pvalue))
# Check if data fits lognormal distribution

sns.distplot(np.log(ratings),fit=stats.norm,kde=False)
# dropping data with < 1000 reviews in an attempt if it has an effect on normality of ratings

more_1000_reviews_ind = [i for i,x in enumerate(g_data['Reviews'] >= 1000) if x]

data = g_data.loc[more_1000_reviews_ind].reset_index().drop('index',axis=1)

plt.title('Ratings Distribution (> 1000 Reviews)')

sns.distplot(data['Rating'],fit=stats.norm,color='orange')
# Categories

data['Category'].value_counts()
# Change all game categories into 'GAME'

game_ind = [i for i,x in enumerate(data['Category'].str.contains('GAME')) if x]

data_2 = data.copy()

data_2.loc[game_ind,'Category'] = 'GAME'
# Boxplot of the ratings for each category

plt.figure(figsize=(13,8)); 

plt.title('Boxplot of the Ratings of each Category');

sns.boxplot(x=data_2['Category'],y=data['Rating'],showmeans=True)

plt.xticks(rotation=90);
# Group the dataset by Category and sort the values by their average rating

group_cat = data_2.groupby('Category')

sorted_rating_by_cat = group_cat['Rating'].mean().sort_values(ascending=False)

sorted_rating_by_cat
# Taking the top 5 categories and the bottom 5 categories

top_5_cat = sorted_rating_by_cat.index[0:5]

bot_5_cat = sorted_rating_by_cat.index[-5:]

print('The top 5 rated categories are {},{},{},{},and {}'.format(*list(top_5_cat)))

print('The bottom 5 rated categories are {},{},{},{}, and {}'.format(*list(bot_5_cat)))
# Making a dataset consisting of only apps from the top 5 and bottom 5 categories

top_5_cat_ind = [i for i,x in enumerate(data_2['Category'].isin(top_5_cat)) if x]

bot_5_cat_ind = [i for i,x in enumerate(data_2['Category'].isin(bot_5_cat)) if x]

top_5_cat_data = data_2.iloc[top_5_cat_ind].reset_index().drop('index',axis=1)

bot_5_cat_data = data_2.iloc[bot_5_cat_ind].reset_index().drop('index',axis=1)

top_bot_cat_data = pd.concat([top_5_cat_data,bot_5_cat_data],axis=0)

top_bot_cat_data
# Boxplot of these categories' ratings

plt.figure(figsize=(13,8))

plt.title('Boxplots of Top 5 and Bottom 5 Categories\' Ratings')

sns.boxplot(x='Category',y='Rating',data=top_bot_cat_data,showmeans=True)

plt.xticks(rotation=90);
# Group the data according to Free/Paid Apps

group_price = data_2.groupby('Free/Paid')

print('The average rating for the Free apps are {}.'.format(group_price['Rating'].mean().loc['Free']))

print('The average rating for the Paid For apps are {}'.format(group_price['Rating'].mean().loc['Paid']))
# Boxplot of free vs paid apps

plt.figure(figsize=(8,5))

plt.title('Boxplot of Ratings of Free and Paid for Apps')

sns.boxplot(x='Free/Paid',y='Rating',data=data_2,showmeans=True)