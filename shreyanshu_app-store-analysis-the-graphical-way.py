# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt
# Import the dataset and delete unwanted columns

data = pd.read_csv('../input/AppleStore.csv')

df=data

df=df.drop(['Unnamed: 0','track_name','currency','ver','vpp_lic'],axis=1)

df_desc=pd.read_csv('../input/appleStore_description.csv')

df.head()
# Check data-type for each column

df.dtypes
# Getting the mean user ratings for the different App genre categories

mean_user_ratings=df.groupby('prime_genre')['user_rating'].mean().reset_index().sort_values(by=['user_rating'])



# plotting values for Average User rating vs App Genre

plt.figure(figsize = (10, 8), facecolor = None)

sns.set_style("darkgrid")

plot = sns.barplot(x="prime_genre", y="user_rating", data=mean_user_ratings,order=mean_user_ratings['prime_genre'])



plot.set_xticklabels(mean_user_ratings['prime_genre'], rotation=90, ha="center")

plot.set(xlabel='App prime genre',ylabel='Average user ratings')

plot.set_title('App Genre vs Average User rating')
# count of apps for different app_genre categories in app store

app_count=df["prime_genre"].value_counts().reset_index()



# Plot of app counts for the various app_genre categories

plt.figure(figsize = (10, 8), facecolor = None)

sns.set_style("darkgrid")

plot1 = sns.barplot(x="index", y="prime_genre", data=app_count)



plot1.set_xticklabels(app_count['index'], rotation=90, ha="center")

plot1.set(xlabel='App prime genre',ylabel='App count in App store')

plot1.set_title('App Genre vs App Count')
# Creating word cloud of the descriptive words from appleStore_description file



stopwords = set(STOPWORDS)

plt.figure(figsize = (10, 8), facecolor = None)

wordcloud = WordCloud(width = 1000, height = 1000,

                background_color ='white',

                stopwords = stopwords,

                max_words=200,

                max_font_size=150, 

                random_state=50,

                min_font_size = 30).generate(str(df_desc['app_desc']))



# plot the WordCloud image                       

plt.figure(figsize = (10, 10), facecolor = None)

plt.imshow(wordcloud)

plt.axis("off")
# Calculating total rating count for the app_genre categories

tot_ratings=df.groupby('prime_genre')['rating_count_tot'].sum().reset_index().sort_values(by=['rating_count_tot'])

tot_ratings



# Plotting total ratings for different app_gene categories

plt.figure(figsize = (10, 8), facecolor = None)

sns.set_style("darkgrid")

plot1 = sns.barplot(x="prime_genre", y="rating_count_tot", data=tot_ratings)



plot1.set_xticklabels(tot_ratings['prime_genre'], rotation=90, ha="center")

plot1.set(xlabel='App prime genre',ylabel='Total ratings')

plot1.set_title('App Genre vs Total ratings')
# Removing the + suffix from the rows of column cont_rating and converting the column data type to numeric

df['cont_rating'] = df.apply(lambda row: row['cont_rating'][:-1], axis=1)

df['cont_rating'] = pd.to_numeric(df['cont_rating'])



# Calculating the mean of cont_ratings for the different app_genre categories

avg_cont_ratings=df.groupby('prime_genre')['cont_rating'].mean().reset_index().sort_values(by=['cont_rating'])

avg_cont_ratings 



# Plotting content ratings for different app_gene categories

plt.figure(figsize = (10, 8), facecolor = None)

sns.set_style("darkgrid")

plot1 = sns.barplot(x="prime_genre", y="cont_rating", data=avg_cont_ratings)



plot1.set_xticklabels(avg_cont_ratings['prime_genre'], rotation=90, ha="center")

plot1.set(xlabel='App prime genre',ylabel='Content Rating')

plot1.set_title('App Genre vs Content rating')