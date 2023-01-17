# import libraries 
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
data = pd.read_csv('/kaggle/input/movies-on-netflix-prime-video-hulu-and-disney/MoviesOnStreamingPlatforms_updated.csv')
data.head()
# total number of non-missing values in each column:
data.count()
# replace missing value in "Age" column:
# if the moive in Disney+, target group is 13+, otherwise is 18+
data.loc[data['Age'].isnull() & data['Disney+']==1,"Age"]="13+"
data.head()
data['Age'].fillna("18+",inplace=True)
data.head()
# replace missing value in "Runtime" column by its mean value:
mean_rt = round(data["Runtime"].mean(),0)
data["Runtime"].fillna(mean_rt, inplace=True)
data.count()
# replace mising value in "IMDb" column by its mean value:
mean_imdb = round(data['IMDb'].mean(),1)
data['IMDb'].fillna(mean_imdb,inplace=True)
data.count()
# check if any duplicate rows:
dupe = data.duplicated()
dupe.sum()
# total number of movie that each platform has:
num_netflix = data['Netflix'].sum()
print(num_netflix)
num_hulu = data['Hulu'].sum()
print(num_hulu)
num_prime = data['Prime Video'].sum()
print(num_prime)
num_disney = data['Disney+'].sum()
print(num_disney)
num_platform = (num_netflix,num_hulu,num_prime,num_disney)
col_names = ('Netflix','Hulu','Prime Video','Disney+')
PlatformList = list(zip(col_names,num_platform))
PlatformCounts = pd.DataFrame(data=PlatformList,columns=['Platform','Number of Movie'])
PlatformCounts
sns.barplot(x=PlatformCounts['Platform'],y=PlatformCounts['Number of Movie'])
plt.pie(num_platform,labels = col_names,autopct='%1.1f%%')
plt.title("Number of Movie")
sum_age = data.groupby("Age").sum()
sum_age[["Netflix","Hulu","Prime Video","Disney+"]]
# number of moive produce each year per platform:
sum_year = data.groupby("Year").sum()
sum_year[["Netflix","Hulu","Prime Video","Disney+"]]
df = data.sort_values("IMDb",ascending=False)
# top 10 movies with highest IMDb rate on Netflix
df.loc[df['Netflix']==1][['Title','IMDb','Genres']].head(10)
df.loc[df['Prime Video']==1][['Title','IMDb','Genres']].head(10)
df.loc[df['Hulu']==1][['Title','IMDb','Genres']].head(10)
df.loc[df['Disney+']==1][['Title','IMDb','Genres']].head(10)
# create dataframe with not null Rotten Tomatoes values:
df_roto = data[data['Rotten Tomatoes'].notnull()]
df_roto.dtypes
# convert str to float type for "Rotten Tomatoes" column:
df_roto['Rotten Tomatoes'] = df_roto['Rotten Tomatoes'].str.rstrip('%').astype('float') 
df_roto.head()
# average IMDb rate on Disney+:
no_disney = df_roto['Disney+'].sum()
disney_avg_imdb = round(df_roto.loc[df_roto['Disney+']==1]['IMDb'].mean(),1)
# average IMDb rate on Hulu:
no_hulu = df_roto['Hulu'].sum()
hulu_avg_imdb = round(df_roto.loc[df_roto['Hulu']==1]['IMDb'].mean(),1)
# average IMDb rate on Netflix:
no_netflix = df_roto['Netflix'].sum()
netflix_avg_imdb = round(df_roto.loc[df_roto['Netflix']==1]['IMDb'].mean(),1)
# average IMDb rate on Prime Video:
no_prime = df_roto['Prime Video'].sum()
prime_avg_imdb = round(df_roto.loc[df_roto['Prime Video']==1]['IMDb'].mean(),1)
# average Rotten Tomatoes rate on Disney+:
disney_avg_roto = round(df_roto.loc[df_roto['Disney+']==1]['Rotten Tomatoes'].mean(),1)
# average Rotten Tomatoes rate on Hulu:
hulu_avg_roto = round(df_roto.loc[df_roto['Hulu']==1]['Rotten Tomatoes'].mean(),1)
# average Rotten Tomatoes rate on Netflix:
netflix_avg_roto = round(df_roto.loc[df_roto['Netflix']==1]['Rotten Tomatoes'].mean(),1)
# average Rotten Tomatoes rate on Prime Video:
prime_avg_roto = round(df_roto.loc[df_roto['Prime Video']==1]['Rotten Tomatoes'].mean(),1)
# create dataframe:
no_platform = (no_netflix,no_hulu,no_prime,no_disney)
col_names = ('Netflix','Hulu','Prime Video','Disney+')
avg_imdb = (netflix_avg_imdb,hulu_avg_imdb,prime_avg_imdb,disney_avg_imdb)
avg_roto = (netflix_avg_roto,hulu_avg_roto,prime_avg_roto,disney_avg_roto)
List = list(zip(col_names,no_platform,avg_imdb,avg_roto))
Counts =  pd.DataFrame(data=List,columns=['Platform','Number of Movie','Average IDMb rate','Average % Rotten Tomattoes rate'])
Counts