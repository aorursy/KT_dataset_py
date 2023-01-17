import numpy as np
import pandas as pd
data=pd.read_csv('../input/movie-recommendations/U.csv')

# This is the first file that we are going to read using pandas. We are going to name it 'data' as it is the first in deal.
# This piece of code reads this comma separaed file and then converts it into a Dataframe.
data.head(n=5)

# Lets load first five observations of our dataset.
movie_titles= pd.read_csv('../input/movie-recommendations/Movie_Id_Titles.csv')
movie_titles.head(n=5)
Movie_imdb_data= pd.merge(data,movie_titles,on='item_id')
Movie_imdb_data.head(n=5)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
Movie_imdb_data.groupby(by='title')['Rating'].mean()

# This would produce means of all the ratings related to a specific title.
# Lets sort them and see which movie has the highest mean of ratings.

Movie_imdb_data.groupby(by='title')['Rating'].mean().sort_values(ascending=False)
Movie_imdb_data.groupby(by='title')['Rating'].count().sort_values(ascending=False)
Movie_imdb_data.groupby(by='title')['Rating'].mean()
New_data=pd.DataFrame(Movie_imdb_data.groupby(by='title')['Rating'].mean())
New_data.head()
New_data['No. of people Rated']=Movie_imdb_data.groupby(by='title')['Rating'].count()
New_data.head(n=5)
New_data.info()
sns.jointplot(x=New_data['Rating'],y=New_data['No. of people Rated']);
sns.jointplot(x=New_data['Rating'],y=New_data['No. of people Rated'],kind='kde');
sns.distplot(New_data['Rating']);
Movie_imdb_data.head(n=5)

# This was the actual dataset that we made initially.
user_all_ratings=Movie_imdb_data.pivot_table(index='user_id',columns='title',values='Rating')

# This inbuilt function from pandas will easily helps us to create a PIVOT TABLE within no time.
# Lets load this dataframe that we have just created.

user_all_ratings.head()
# Now iam going to get the column of that specified movie and would correlate it with the other movie columns.

Toy_story_similar_movies=user_all_ratings['Toy Story (1995)']
X=user_all_ratings.corrwith(Toy_story_similar_movies)
X.head()
Similar_movies=pd.DataFrame(X,columns=['Correlation'])
Similar_movies.sort_values('Correlation',ascending=False).head(n=5)
def Sim_mov_recomm():
    name=input('Please enter the name of the movie you like:')
    y=user_all_ratings[name]
    z=user_all_ratings.corrwith(y)
    df=pd.DataFrame(z,columns=['Correlation'])
    b=df.sort_values(by='Correlation',ascending=False).head(n=5)
    c=list(b.index)
    print('Movies you would also like:')
    for i in c:
        print(i)
    
    
Sim_mov_recomm()