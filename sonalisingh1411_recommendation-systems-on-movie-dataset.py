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
#Import libraries



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('white')

%matplotlib inline
#Dataframe-1 

column_names = ['user_id', 'item_id', 'rating', 'timestamp']

df = pd.read_csv('../input/movie-dataset123/u.csv', names=column_names)

df.head()
#Movie_titles(Dataframe-2)

movie_titles = pd.read_csv("../input/movie-title/Movie_Id_Titles.csv")

movie_titles.head()

#Merging or joinning two dataframe based on Item_id..........

df = pd.merge(df,movie_titles,on='item_id')

df.head()



#Visualize highest user rating..........

df['rating'].value_counts().plot(kind='bar',color='green')



#Checking the unique title in df dataset....... 

df1 =pd.DataFrame(df['title'].unique())

df1.count()
#Checking the unique user's in df dataset....... 

df2 =pd.DataFrame(df['user_id'].unique())

df2.count()



# =============================================================================

# # EDA

# 

# Let's explore the data a bit and get a look at some of the best rated movies.

# =============================================================================

# 

# Let's create a ratings dataframe with average rating and number of ratings:

# =============================================================================

Mean_df= df.groupby('title')['rating'].mean().sort_values(ascending=False).head()

Mean_df.head()
#Checking the user's who have viewed Movie most.... 

Count_df = df.groupby('title')['rating'].count().sort_values(ascending=False).head()

Count_df.head()
#Taking out the mean of the rating based on unique title movie.........

ratings = pd.DataFrame(df.groupby('title')['rating'].mean())

ratings.head()



#Now set the number of ratings column

ratings['num of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())

ratings.head()
#Visualizing the num of rating(Count of movie who have give rating to it)

plt.figure(figsize=(10,4))

ratings['num of ratings'].hist(bins=70)




#Visualizing the rating

plt.figure(figsize=(10,4))

ratings['rating'].hist(bins=70)


#Or you can even try this plot used to view two different plot(Hist,scatter plot) 

sns.jointplot(x='rating',y='num of ratings',data=ratings,alpha=0.5)



# =============================================================================

# 

# Okay! Now that we have a general idea of what the data looks like, 

# let's move on to creating a simple recommendation system:

# 

# Recommending Similar Movies

# 

# Now let's create a matrix that has the user ids on one access and the movie title

# on another axis. Each cell will then consist of the rating the user gave to that movie. 

#Note there will be a lot of NaN values, because most people have not seen most of the movies.

# =============================================================================





moviemat = df.pivot_table(index='user_id',columns='title',values='rating')



moviemat.head()





#Lets see the top rated movies based on user interest..........

ratings.sort_values('num of ratings',ascending=False).head(10)





ratings.head()





#Now if we sort the dataframe by correlation, we should get the most similar

#movies, however note that we get some results that don't really make sense.

#This is because there are a lot of movies only watched once by users who also

#watched star wars (it was the most popular movie).



# =============================================================================

# Working with star_wars movie .......................

# =============================================================================



#Now let's grab the user ratings for those two movies:

starwars_user_ratings = moviemat['Star Wars (1977)']

starwars_user_ratings.head()







#We can then use corrwith() method to get correlations between two pandas series:

similar_to_starwars = moviemat.corrwith(starwars_user_ratings)



#Drop nan:----(STARWARS)

corr_starwars = pd.DataFrame(similar_to_starwars,columns=['Correlation'])

corr_starwars.dropna(inplace=True)

corr_starwars.head()



corr_starwars.sort_values('Correlation',ascending=False).head(10)


#Let's fix this by filtering out movies that have less than 100 reviews 

#(this value was chosen based off the histogram from earlier).

corr_starwars = corr_starwars.join(ratings['num of ratings'])

corr_starwars.head()





#Now sort the values and notice how the titles make a lot more sense:

corr_starwars[corr_starwars['num of ratings']>100].sort_values('Correlation',ascending=False).head()







# =============================================================================

# Working with LiarLiar movie............

# =============================================================================





liarliar_user_ratings = moviemat['Liar Liar (1997)']

liarliar_user_ratings.head()





similar_to_liarliar = moviemat.corrwith(liarliar_user_ratings)




#Now the same for the comedy Liar Liar:

corr_liarliar = pd.DataFrame(similar_to_liarliar,columns=['Correlation'])

corr_liarliar.dropna(inplace=True)



corr_liarliar.sort_values('Correlation',ascending=False).head(10)



#Let's fix this by filtering out movies that have less than 100 reviews 

#(this value was chosen based off the histogram from earlier).

corr_liarliar = corr_liarliar.join(ratings['num of ratings'])



#Now sort the values and notice how the titles make a lot more sense:

corr_liarliar[corr_liarliar['num of ratings']>100].sort_values('Correlation',ascending=False).head()







# =============================================================================

# Working with Contact movie.............

# =============================================================================



#Now let's grab the user ratings for those two movies:

Contact_user_ratings = moviemat['Contact (1997)']

Contact_user_ratings.head()


similar_to_Contact = moviemat.corrwith(Contact_user_ratings)
corr_Contact = pd.DataFrame(similar_to_Contact,columns=['Correlation'])

corr_Contact.dropna(inplace=True)



corr_Contact.sort_values('Correlation',ascending=False).head(10)



#Let's fix this by filtering out movies that have less than 100 reviews 

#(this value was chosen based off the histogram from earlier).

corr_Contact = corr_Contact.join(ratings['num of ratings'])

#Now sort the values and notice how the titles make a lot more sense:

corr_Contact[corr_Contact['num of ratings']>100].sort_values('Correlation',ascending=False).head()
