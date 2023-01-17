# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
store = "../input/AppleStore.csv"
data = pd.read_csv(store) # Data is the name of the dataframe containing all apps!!!!!!!!!!
data.head(10)
# Sort the dataframe by number of ratings
most_ratings = data.sort_values(by=['rating_count_tot'], ascending=False)

# Store the top 100 most rated apps into a new dataframe
top100 = most_ratings.head(100)
top100
best100 = top100.sort_values(by = ['user_rating'], ascending=False)
best100
# Create a dataframe where the user ratings are greater than or equal to 4.5
toprated = best100.loc[best100['user_rating'] >= 4.5]

# Sort the apps with 4.5+ ratings by the number of ratings they have
bestsellers = toprated.sort_values(by = ['rating_count_tot'], ascending=False)
bestsellers
print('Of the top 100 most rated apps, only', len(bestsellers), 'have a rating of 4.5 or above and are considered best seller apps in our analysis.')
# Create a dataframe of paid apps
bestsellers_paid = bestsellers.loc[bestsellers['price'] != 0.00]
bestsellers_paid
# Count how many rows of paid bestselling apps
best_paid = len(bestsellers_paid)
print('There are', best_paid, 'paid best selling apps.')
# Create a dataframe of free apps
bestsellers_free = bestsellers.loc[bestsellers['price'] == 0.00]
bestsellers_free
# Count how many rows of free bestselling apps
best_free = len(bestsellers_free) 
print('There are', best_free, 'free best selling apps.')
# Using the bestsellers DF, group by price and count the number of apps at each price level
best_distribution = bestsellers['price'].groupby(bestsellers['price']).count()

# Display a bar chart of app counts grouped by price
best_distribution.plot(kind='bar')
plt.xlabel('Price')
plt.ylabel('Count')
plt.title('Best Selling App Prices')
# List unique values in the prime_genre column for the bestsellers dataframe
best_genres = bestsellers.prime_genre.unique()
best_genres
# List allllllll app genres
all_genres = data.prime_genre.unique()
all_genres
print('From our analysis, we can determine that there are', len(all_genres), 'different app genres. However, best selling apps only consist of', len(best_genres), 'different genres.')
# Group all best seller apps by genre and count the number of apps in each genre
best_genres_viz = bestsellers['prime_genre'].groupby(bestsellers['prime_genre']).count().sort_values(ascending=False)

# Display a bar chart of the grouped data above
best_genres_viz.plot(kind='bar')
plt.xlabel('App Genre')
plt.ylabel('Count')
plt.title('Best Selling App Genres')
# Create a new column to calculate app sizes in megabytes
data['size_MB'] = data['size_bytes'] / (1000 * 1000.0)
bestsellers['size_MB'] = bestsellers['size_bytes'] / (1000 * 1000.0)
# Determine the average app size for both the bestselling dataframe and all the apps
avg_size = data['size_MB'].mean()
best_avg_size = bestsellers['size_MB'].mean()
print('The average size of all apps within our dataset is', round(avg_size, 2), 'megabytes.')
print('The average size of best seller apps is', round(best_avg_size, 2), 'megabytes.')
plt.hist(bestsellers['size_MB'], bins='auto')
plt.title('Size Frequency of the Best Selling Apps')
plt.ylabel('Frequency')
plt.xlabel('App Size in MB')
#Pull out all the paid apps from the data 

paid_apps = data.loc[data['price'] != 0.00]
paid_apps
num_paid_apps = len(paid_apps)
print('The total number of paid apps in our dataset is', num_paid_apps)
max_price = paid_apps.max()
max_price
max_price = paid_apps.max()
average_price = paid_apps["price"].mean()
print('The average price of all of the paid apps is $', round(average_price, 2))
paid_avg_user_rating = paid_apps["user_rating"].mean()
print('The average user rating of paid apps is ', round(paid_avg_user_rating, 2))

free_apps = data.loc[data['price'] == 0.00]
free_avg_user_rating = free_apps["user_rating"].mean()
print('The average user rating of free apps is ', round(free_avg_user_rating, 2))
# PAID APP RATING VIZ

# Take the paid apps, group by user rating and count the number of apps for each rating
paid_ratings = paid_apps['user_rating'].groupby(paid_apps['user_rating']).count()

# Display the data in a bar chart, make sure you edit the axis and title
paid_ratings.plot(kind='bar')
plt.xlabel('App Ratings')
plt.ylabel('Count')
plt.title('Paid App Ratings')
# FREE APP RATING VIZ

free_rating = free_apps['user_rating'].groupby(free_apps['user_rating']).count()

free_rating.plot(kind = 'bar')
plt.xlabel('App Ratings')
plt.ylabel('Count')
plt.title('Free App Ratings')


music_apps = data.loc[data['prime_genre'] == 'Music']
num_music = len(music_apps)
print("The number of applications for the Music genre is: " , num_music)

book_apps = data.loc[data['prime_genre'] == 'Book']
num_book = len(book_apps)
print("The number of applications for the Book genre is: ", num_book)

business_apps = data.loc[data['prime_genre'] == 'Business']
num_business = len(business_apps)
print("The number of applications for the Business genre is: ", num_business)

catalogs_apps = data.loc[data['prime_genre'] == 'Catalogs']
num_catalogs = len(catalogs_apps)
print("The number of applications for the Catalogs genre is: ", num_catalogs)

education_apps = data.loc[data['prime_genre'] == 'Education']
num_education = len(education_apps)
print("The number of applications for the Education genre is: ", num_education)

entertainment_apps = data.loc[data['prime_genre'] == 'Entertainment']
num_entertainment = len(entertainment_apps)
print("The number of applications for the Entertainment genre is: ", num_entertainment)

finance_apps = data.loc[data['prime_genre'] == 'Finance']
num_finance = len(finance_apps)
print("The number of applications for the Finance genre is: ", num_finance)

fooddrink_apps = data.loc[data['prime_genre'] == 'Food & Drink']
num_fooddrink = len(fooddrink_apps)
print("The number of applications for the Food & Drink genre is: ", num_fooddrink)

games_apps = data.loc[data['prime_genre'] == 'Games']
num_games = len(games_apps)
print("The number of applications for the Games genre is: ", num_games)

healthfitness_apps = data.loc[data['prime_genre'] == 'Health & Fitness']
num_healthfitness = len(healthfitness_apps)
print("The number of applications for the Health & Fitness genre is: ", num_healthfitness)

lifestyle_apps = data.loc[data['prime_genre'] == 'Lifestyle']
num_lifestyle = len(lifestyle_apps)
print("The number of applications for the Lifestyle genre is: ", num_lifestyle)

medical_apps = data.loc[data['prime_genre'] == 'Medical']
num_medical = len(medical_apps)
print("The number of applications for the Medical genre is: ", num_medical)

navigation_apps = data.loc[data['prime_genre'] == 'Navigation']
num_navigation = len(navigation_apps)
print("The number of applications for the Navigation genre is: ", num_navigation)

news_apps = data.loc[data['prime_genre'] == 'News']
num_news = len(news_apps)
print("The number of applications for the News genre is: ", num_news)

photovideo_apps = data.loc[data['prime_genre'] == 'Photo & Video']
num_photovideo = len(photovideo_apps)
print("The number of applications for the Photo & Video genre is: ", num_photovideo)

productivity_apps = data.loc[data['prime_genre'] == 'Productivity']
num_productivity = len(productivity_apps)
print("The number of applications for the Productivity genre is: ", num_productivity)

reference_apps = data.loc[data['prime_genre'] == 'Reference']
num_reference = len(reference_apps)
print("The number of applications for the Reference genre is: ", num_reference)

shopping_apps = data.loc[data['prime_genre'] == 'Shopping']
num_shopping = len(shopping_apps)
print("The number of applications for the Shopping genre is: ", num_shopping)

socialNetworking_apps = data.loc[data['prime_genre'] == 'Social Networking']
num_socialNetworking = len(socialNetworking_apps)
print("The number of applications for the Social Networking genre is: ", num_socialNetworking)

sports_apps = data.loc[data['prime_genre'] == 'Sports']
num_sports = len(sports_apps)
print("The number of applications for the Sports genre is: ", num_sports)

travel_apps = data.loc[data['prime_genre'] == 'Travel']
num_travel = len(travel_apps)
print("The number of applications for the Travel genre is: ", num_travel)

utilities_apps = data.loc[data['prime_genre'] == 'Utilities']
num_utilities = len(utilities_apps)
print("The number of applications for the Utilities genre is: ", num_utilities)

weather_apps = data.loc[data['prime_genre'] == 'Weather']
num_weather = len(weather_apps)
print("The number of applications for the Weather genre is: ", num_weather)

#weather_avg = weather_apps['user_rating'].mean()
#print('The avg user rating for weather apps is', weather_avg)
# APPS GENRES VIZ

# Visualize the count of app genres with a bar chart
# Group by app genres and count the number of apps in each genre


# Take the DF with all apps, group by user rating and count the number of apps for each rating
all_ratings = data['prime_genre'].groupby(data['prime_genre']).count().sort_values(ascending=False)

# Display the data in a bar chart, make sure you edit the axis and title
all_ratings.plot(kind='bar')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.title('App Genre Distribution')
price_per_genre = data[['prime_genre', 'price']].groupby('prime_genre').mean()['price'].sort_values(ascending=False)
price_per_genre.plot(kind='bar')
plt.xlabel('Genre')
plt.ylabel('Price ($)')
plt.title('Average App Price per Genre')
# GAMES APPS PRICE DISTRIBUTION

# Make a histogram of paid app prices
plt.hist(games_apps['price'], bins='auto')
plt.axis([0, 30, 0, 2300])
plt.xlabel('Price ($)')
plt.ylabel('Count')
plt.title('Gaming Apps Price Distribution')

# FIRST, determine the max price so you know the range of your x-axis. 
# games_apps['price'].max() = 27.99
# So above in this code: plt.axis([0, 30, 0, 2300]) the second number value '30' is the x axis max
# DELETE MAX PRICE ONCE YOU HAVE IT DETERMINED

# !!!HOW TO FIX AXIS TO SHOW ALL DATA, DO TRIAL AND ERROR JUST LIKE WE WOULD IN EXCEL OR SOMETHING!!!
# plt.axis([x-axis min, x-axis max, y-axis min, y-axis max])
# ENTERTAINMENT APPS PRICE DISTRIBUTION

hist = plt.hist(entertainment_apps['price'], bins='auto')
plt.axis([0, 10, 0, 350])
plt.xlabel('Price ($)')
plt.ylabel('Count')
plt.title('Entertainment Apps Price Distribution')

# entertainment_apps['price'].max()
# EDUCATION APPS PRICE DISTRIBUTION

edu_hist = plt.hist(education_apps['price'], bins='auto')
plt.axis([0, 30, 0, 200])
plt.xlabel('Price ($)')
plt.ylabel('Count')
plt.title('Education Apps Price Distribution')
# PHOTO & VIDEO APPS PRICE DISTRIBUTION

plt.hist(photovideo_apps['price'], bins='auto')
plt.axis([0, 30, 0, 200])
plt.xlabel('Price ($)')
plt.ylabel('Count')
plt.title('Photo&Video Apps Price Distribution')
# UTILITIES APPS PRICE DISTRIBUTION

plt.hist(utilities_apps['price'], bins='auto')
plt.axis([0, 30, 0, 125])
plt.xlabel('Price ($)')
plt.ylabel('Count')
plt.title('Utilities Apps Price Distribution')
#List of Categories
categories = ['Music', 'Book', 'Business', 'Catalogs', 'Education', 'Entertainment', 'Finance', 'Food & Drink', 'Games',
              'Health & Fitness', 'Lifestyle','Medical','Navigation', 'News','Photo & Video', 'Productivity','Reference',
             'Shopping','Social Networking', 'Sports', 'Travel' ,'Utilities','Weather']

#variables to store information about the best category and its correspoding average rating
best_category = ''
best_avg_rating = 0

#For loop to iterate through each category
for each_category in categories:
    #get dataset of the current category
    current_dataset = data.loc[data['prime_genre'] == each_category ]
    
    #calculate average rating for the selected category
    current_avg_rating = current_dataset['user_rating'].mean()
    
    #if current category has higher average rating than the best average rating, then store its information in the aformentioned variables
    if(current_avg_rating > best_avg_rating):
        best_category = each_category
        best_avg_rating = current_avg_rating

#print the output statement
print('The leading category in terms of highest average rating would be', best_category,'with an average rating of', str(best_avg_rating))
price_per_genre = data[['prime_genre', 'user_rating']].groupby('prime_genre').mean()['user_rating'].sort_values(ascending=False)
price_per_genre.plot(kind='bar')
plt.xlabel('Genre')
plt.ylabel('Rating')
plt.title('Average User Rating per Genre')
# PRODUCTIVITY APPS PRICE DISTRIBUTION

hist = plt.hist(productivity_apps['price'], bins='auto')
plt.axis([0, 100, 0, 100])
plt.xlabel('Price')
plt.ylabel('Count')
plt.title('Entertainment Apps Price Distribution')

# productivity_apps['price'].max()
size_Q1 = data.size_MB.quantile(.25)
size_Q2 = data.size_MB.quantile(.5)
size_Q3 = data.size_MB.quantile(.75)
size_IQR = size_Q3 - size_Q1
print('The first quartile of app sizes is', round(size_Q1,2))
print('The second quartile of app sizes is', round(size_Q2,2))
print('The third quartile of app sizes is', round(size_Q3,2))
print('The inter quartile range of app sizes is', round(size_IQR,2))
plt.hist(data['size_MB'], bins='auto')
plt.title('Size Distribution of the All Apps')
plt.ylabel('Frequency')
plt.xlabel('App Size in MB')
test = paid_apps[['prime_genre', 'size_MB']].groupby('prime_genre').mean()['size_MB'].sort_values(ascending=False)
test.plot(kind='bar')
plt.xlabel('Genre')
plt.ylabel('Size (MB)')
plt.title('Mean App Sizes of Paid Apps')
test = free_apps[['prime_genre', 'size_MB']].groupby('prime_genre').mean()['size_MB'].sort_values(ascending=False)
test.plot(kind='bar')
plt.xlabel('Genre')
plt.ylabel('Size (MB)')
plt.title('Mean App Sizes of Free Apps')