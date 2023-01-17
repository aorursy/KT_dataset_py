# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from IPython.display import clear_output



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# I am on loading combine_data_1

df1=pd.read_csv('/kaggle/input/netflix-prize-data/combined_data_1.txt', header = None, names = ['Cust_Id', 'Rating'], usecols = [0,1])

#df2=pd.read_csv('/kaggle/input/netflix-prize-data/combined_data_2.txt', header = None, names = ['Cust_Id', 'Rating'], usecols = [0,1])

#df3=pd.read_csv('/kaggle/input/netflix-prize-data/combined_data_3.txt', header = None, names = ['Cust_Id', 'Rating'], usecols = [0,1])

#df4=pd.read_csv('/kaggle/input/netflix-prize-data/combined_data_4.txt', header = None, names = ['Cust_Id', 'Rating'], usecols = [0,1])

#df=[df1,df2,df3,df4]

#df=pd.concat(df,axis=0)



# The rows with customr Id has ':' is the movie id

# For example the first row the '1:' is the movie id

# On 548 row the customer id '2:' is the movie id



print(df1.head())

print(df1[df1.Rating.isna()])
# Hence add column movie id and fill that witht the movie id

# For example '1' for index 0 to 547

# Then drop all rows with NaN 





n=df1.loc[pd.isnull(df1.Rating)].index

df1['Movie_id']=np.zeros(df1.shape[0])

for i in range(len(n)-1):

    df1.iloc[np.arange(n[i],n[i+1]),-1]=int(df1.iloc[n[i],0][:-1])

df1.dropna(axis=0,inplace=True)

df1[['Cust_Id','Rating','Movie_id']]=df1[['Cust_Id','Rating','Movie_id']].astype('int32')

print(df1.iloc[545:550,:])
# Corelation function

### Filtered array: array containing all the customers who watched atleast n number of movies as the user

### c1: array containing all the movies watched by a user

### ids of all the customers







def corr_function(filtered_array,c1,ids):



    corr_list=np.zeros(ids.shape[0])



    for i in np.arange(ids.shape[0]):

      id2=ids[i]

      c2=filtered_array[np.where(filtered_array[:,0]==id2)]

      n2=c2[np.where(np.in1d(c2[:,1],c1[:,0]))]

      n1=c1[np.where(np.in1d(c1[:,0],c2[:,1]))]

      corr_list[i]=np.corrcoef(n2[:,2],n1[:,1])[0,1]

      

    return corr_list





###Cupy is found to be much faster than numpy



#import cupy as cp

#def corr_function_cp(filtered_array,c1,ids):

#  cp.cuda.Device().synchronize()

# with cp.cuda.Device(0):

#    corr_list=cp.zeros(ids.shape[0])

    

#    for i in cp.arange(ids.shape[0]):

#      id2=ids[i]

#      c2=filtered_array[cp.where(filtered_array[:,0]==id2)]

#      n2=c2[np.where(cp.in1d(c2[:,1],c1[:,0]))]

#      n1=c1[np.where(cp.in1d(c1[:,0],c2[:,1]))]

#      corr_list[i]=cp.corrcoef(n2[:,2],n1[:,1])[0,1]

      

#  return corr_list
req_id=822109

## Find all the customers who watched the same movies as the required user



cust_unique=df1.loc[df1.Cust_Id==req_id]

df_filtered=df1.loc[df1.Movie_id.isin(cust_unique.Movie_id.values)]



# Num of minimum common movies between the user and any given customer

# bls contains all the ids of customers who have minimum 30 movies or greater common

num_of_movies=15



bls=df_filtered.groupby(['Cust_Id']).apply(lambda x: len(x) >= num_of_movies)

bls=bls[bls.values==True].index

bls=bls[bls!=req_id]

print(['Total customers with minimum 30 movies: '+str(len(bls))])





filtered_array_values=df_filtered[['Cust_Id','Movie_id','Rating']].dropna(axis=0).values





# Dataframe values where req_id is present

c1=df_filtered.loc[df_filtered.Cust_Id==req_id,['Movie_id','Rating']].dropna(axis=0).values



correlation_values=corr_function(filtered_array_values,c1,np.array(bls))

### Using Cupy in case of GPU

#with cp.cuda.Device(0):

 # filtered_array_cupy=cp.asarray(filtered_array)

  #arc1=df_filtered.loc[df_filtered.Cust_Id==req_id,['Movie_id','Rating']].dropna(axis=0).values

  #arc1_cupy=cp.asarray(arc1)

  #start_time=time.time()



#cProfile.run('corr_function_cp(filtered_array_cupy,arc1_cupy,cp.array(bls[range(1000)]))')

#correlation_values=corr_function_cp(filtered_array_cupy,arc1_cupy,cp.array(bls[range(10)]))
### 

#Ids=corr_function_cp(filtered_array_cupy,arc1_cupy,cp.array(bls))
## Correlated Ids: Ids of the coustomers who have correlation greater than 0.7

# we can take the negative also, but for in this case I am neglecting, since I found only few ids based

#on my experience of running 



# 0.7 is arbitrary value I took, but I guess we can use 0.75 or 0.8 based on our experience



Correlated_Ids=[]

for x,y in zip(bls,correlation_values):

  if y>0.7:

    #print(y)

    Correlated_Ids.append(x)

len(Correlated_Ids)
# The function finds the movies which are not watched by the user but rated high by the customer

# pass the customer id to get the moovies list



def Recommended_Movies_from_an_Id(id):

  Movie_Test=df1.loc[(df1.Cust_Id==id)&(df1.Rating>3),['Movie_id']].values

  Movie_Test=list(Movie_Test)

  for each in df_filtered.Movie_id.unique():

    try:

      Movie_Test.remove(each)

    except:

      pass

  Movie_Test=df1.loc[(df1.Cust_Id==id)&(df1.Movie_id.isin(Movie_Test)),['Movie_id','Rating']]

  return Movie_Test

## The list of all recommended movies for a particular user

All_Recommended_Movies=[]

for each in Correlated_Ids:

  All_Recommended_Movies.append(Recommended_Movies_from_an_Id(each))

All_Recommended_Movies=pd.concat(All_Recommended_Movies)

All_Recommended_Movies.head()
## Select only the movies which are recommended more than 'filter_count' times, in this case

# I am passing the values 5



def Repeated_Recomendation(movies_list,filter_count):

  records_array = np.array(movies_list)

  vals, inverse, count = np.unique(records_array, return_inverse=True,

                                return_counts=True)



  idx_vals_repeated = np.where(count > filter_count)[0]

  vals_repeated = vals[idx_vals_repeated]

  return vals_repeated
vals_repeated=Repeated_Recomendation(All_Recommended_Movies.Movie_id.values,5)
## Load the movies titles to dataframe list of movies

list_of_movies=pd.read_csv('/kaggle/input/netflix-prize-data/movie_titles.csv', usecols = [0,1,2],encoding = "ISO-8859-1",header=None,names=['id','year','Name'])

list_of_movies.head()
# Find the movies which are repeated



movies_repeated=list_of_movies.loc[list_of_movies.id.isin(vals_repeated),['Name','year']]

# Recomendation is simply based on the sum of ratings 

movies_repeated['recomendation']=All_Recommended_Movies.loc[All_Recommended_Movies.Movie_id.isin(vals_repeated)].groupby(['Movie_id'])['Rating'].sum().sort_values(ascending=False).values

movies_repeated['recomendation']=movies_repeated['recomendation']/movies_repeated['recomendation'].max()

decimals = 2    

# Lets set the  the maximum sum to 1, the rest are fraction of that 

movies_repeated['recomendation'] = movies_repeated['recomendation'].apply(lambda x: round(x, decimals))

print(movies_repeated.shape)

print(movies_repeated.head(50))
Imdb=pd.read_csv('/kaggle/input/imdbdata/AllMoviesDetailsCleaned.csv',encoding = "ISO-8859-1",error_bad_lines=False,sep=';')
list_of_movies.Name=list_of_movies.Name.apply(lambda x:x.lower())

Imdb['year']=pd.to_datetime(Imdb.release_date)

Imdb.title=Imdb.title.astype(str)

Imdb['title']=Imdb.title.apply(lambda x:x.lower())

Imdb.year=Imdb.year.apply(lambda x:x.year)

Imdb.year=Imdb.year.fillna(0)

Imdb.year.astype(int)
## Find the genres for movies in imdb list

## I am merging both dataframes

list_of_movies.rename({'Name':'title'},axis=1,inplace=True)

Movies_in_Imdb_set=list_of_movies.merge(Imdb,left_on=['title',"year"],right_on=['title','year'])

print('Number of movies available in Imdb Dataset: '+ str(Movies_in_Imdb_set.shape[0]))
## Create a random id in database for you

## Make sure Id doesn't exist in current database



new_user_id=np.random.randint(1,2649429,1)

while new_user_id in df1.Cust_Id.values:

  new_user_id=np.random.randint(1,2649429,1)

print(new_user_id)

## Lets create a dataframe to attach to the main dataset

df_input=[]