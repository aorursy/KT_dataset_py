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
# import packages

import seaborn as sns

import matplotlib.pyplot as plt
user_reviews = pd.read_csv("../input/googleplaystore_user_reviews.csv")

playstore_data = pd.read_csv("../input/googleplaystore.csv")
user_reviews.head()



playstore_data.head()

print('Dimentions of user_review {}'.format(user_reviews.shape))

print('Dimentions of google_playstore {}'.format(playstore_data.shape))

print('Unique Stores {}'.format(len(playstore_data['App'].unique())))
# In total there are around 64295 reviews further there are 9660 apps and apps can have almost same names with slight change and different maker
# Checking for missing values



print('Missing values in user_reviews {}'.format(user_reviews.isnull().sum()))
print('Missing values in Google_play_store {}'.format(playstore_data.isnull().sum()))
# removing the missing values



user_reviews.dropna(inplace=True)
# remove duplicates in play store



#playstore_data.drop_duplicates(inplace=True)

playstore_data= playstore_data.drop_duplicates(subset='App')
playstore_data.shape
print('Missing values in Google_play_store {}'.format(playstore_data.isnull().sum()))
# as data is not capurted in scraping i will be removing the missing values other then rating for better analysis
playstore_data.dropna(inplace=True,subset=['Type','Content Rating','Current Ver','Android Ver'])
playstore_data.shape
playstore_data['Rating'].describe()
# filling rating in missing areas by not captured (0)



playstore_data.fillna(0,inplace=True)
playstore_data[playstore_data['Rating'] == 0].head()
# Descriptive analytics on numerical variables



print(user_reviews.columns)

user_reviews.describe()





# about 75 percentile of the time sentiment polarity is around 0.4, which is positive

# subjectivity is between 0 and 1, median subjectivity that was understood was 0.51

# descriptive analysis on play store



print(playstore_data.describe())

# density plot shape

rating = playstore_data[playstore_data['Rating'] != 0 ]

print("After removing the missing values in ratings")

print(rating.describe())

sns.kdeplot(shade=True,data=rating['Rating'])



from scipy.stats import kurtosis, skew



x = np.random.normal(0, 2, 10000)

print( 'excess kurtosis of  distribution : {}'.format( kurtosis(rating['Rating']) ))

print( 'skewness of distribution: {}'.format( skew(rating['Rating']) ))

# Rating is  Left skewed(Negatively skewed) and median value is around 4.3 from discriptive analysis and density plot

#Kurtosis is around 5.5 that means data is above normal distribution 

# this a actually a good this a rating is above 4 thats a good indication that most of the apps are liked by users

# Univariate analysis
# Bar plot on categorical variable



df1 = user_reviews['Sentiment'].value_counts()

df1 = df1.reset_index()

def bar_plot(x,y,y_label,title,color):

    objects = x.values

    y_pos = np.arange(len(objects))

    plt.figure(figsize=(10,5))

    bar = plt.bar(x,y,color=color)

    plt.xticks(y_pos, objects)

    plt.ylabel(y_label)

    plt.title(title)

    

    return bar

    
df1['index'].values
bar_plot(x = df1['index'],y = df1['Sentiment'],color='g',y_label = 'Sentiment_Freq',title = 'Bar Plot on Sentiment')

playstore_data.head(2)
# visualize the following

# 1. how many apps are free vs paid

# 2. how many genres are there

# 3. represent installs

# 4. represent Content Rating







playstore_data.columns
list_1 = ['Category', 'Installs', 'Type',

        'Content Rating']
def bar_plot(x,y,y_label,x_label,title,color,ax):

    # plt.figure(figsize=(10,5))

    bar = sns.barplot(x = x,y=y,ax=ax,orient='h')

    plt.ylabel(y_label)

    plt.xlabel(x_label)

    plt.title(title)

    for i, v in enumerate(x.values):

        ax.text(v + 3, i + .25, str(v), color='black', fontweight='bold')

    return bar
fig = plt.figure(figsize=(14,18))

fig.subplots_adjust(hspace=0.4, wspace=0.4)

i = 1

for names in list_1:

    ax1 = fig.add_subplot(2, 2, i)

    df2 = playstore_data[names].value_counts()

    df2 = df2.reset_index()

    bar_plot(x = df2[names],y = df2['index'],y_label = 'Freq',title = 'Bar Chart On {}'.format(names),color='red',ax=ax1,x_label=names)

    i += 1
# from the above viz we can see that free apps are more then paid apps in this dataset

# from installs we find that there are 20 apps which have like 1 billion downloads

# 100 Million installs are around for 188 apps and majority of installs are in 1 million and above 10 million installs

list_2 = ['Genres']
def bar_plot(x,y,y_label,x_label,title,color,ax=None):

    plt.figure(figsize=(5,8))

    bar = sns.barplot(x = x,y=y,orient='h')

    plt.ylabel(y_label)

    plt.xlabel(x_label)

    plt.title(title)

    for i, v in enumerate(x.values):

        bar.text(v + 3, i + .25, str(v), color='black', fontweight='bold')

    return bar

df2 = playstore_data['Genres'].value_counts()

df2 = df2.reset_index()

df2 = df2[df2['Genres'] > 100]

bar_plot(x = df2['Genres'],y = df2['index'],y_label = 'Freq',title = 'Bar Chart On Gerner',color='red',x_label='Gerner')

   
# highest apps are made on tools, entertainment and education
""" Analysis on Apps most popular apps in terms of installs

% of free and paid apps

10 rated apps with installs of 100 million

update year wise

Avg app size by genre"""

# Apps with 1 billion downloads

playstore_data[playstore_data['Installs'] == '1,000,000,000+']['App']
genres=  list(df2['index'][1:10])
d = pd.DatetimeIndex(playstore_data['Last Updated'])

playstore_data['year'] = d.year

playstore_data['month'] = d.month
for i in genres:

    

    play = playstore_data[(playstore_data['Installs'] != '1,000,000,000+') & (playstore_data['Genres'] == i) & (playstore_data['Rating'] >= 4.5) & (playstore_data['year'] == 2018)]['App']

    print('')

    print('Printing 10 Apps with 100 million installs and Rating >= 4.5 and Year = 2018 in {}'.format(i))

    print('--------------------------------------------------')

    print(play[0:10])


# % free vs paid apps



size=[8895,753]

sentiment = ['Free', 'Paid']

colors = ['g', 'pink']

plt.pie(size, labels=sentiment, colors=colors, startangle=180, autopct='%.1f%%')

plt.title('% Free vs Paid Apps')

plt.show()

# avg app size

size = playstore_data[playstore_data['Size'] != 'Varies with device']

size_m = []

size_n = []

for i in size['Size']:

    size_m.append(i[-1])

    size_n.append(float(i[0:-1]))

    

size['size_m'] = size_m

size['size_n'] = size_n

size['size_m'] = size['size_m'].replace('k',1000)

size['size_m'] = size['size_m'].replace('M',1000000)



size['bites'] = size['size_n'] * size['size_m']
# avg app size



grouped = size.groupby('Category').agg({'bites': [min, max]})

grouped.columns = grouped.columns.droplevel(level=0)

grouped.rename(columns={"min": "min_size", "max": "max_size"})

grouped.head(10)

size.groupby('Category')['bites'].mean().head(10)

# app update



print('# of apps not been updated since 2016 {}'.format(len(playstore_data[playstore_data['year'] < 2016])))

print('# of apps not been updated since 2015 {}'.format(len(playstore_data[playstore_data['year'] < 2015])))

print('# of apps not been updated since 2014 {}'.format(len(playstore_data[playstore_data['year'] < 2014])))                                                       

                                                        
# there are around 801 apps that are not been updated since 2 years,these apps might not be in service
# analysis on paid apps



paided = playstore_data[playstore_data['Type'] == 'Paid']
df3 = paided['Category'].value_counts()

df3 = df3.reset_index()

df3 = df3[:10]

plt.figure(figsize=(10,5))

plt.pie(x = list(df3['Category']), labels=list(df3['index']), autopct='%1.0f%%', pctdistance=0.8, labeldistance=1.2)

plt.title('% Distribution of Paided Apps Categories')
# Medial apps have  13 % of share, further analysis is required understand medical apps 

# Top rated paid apps with installs 1,000,000+



paided[(paided['Rating'] > 4.7) & (paided['Installs'] == '100,000+') ]['App']


