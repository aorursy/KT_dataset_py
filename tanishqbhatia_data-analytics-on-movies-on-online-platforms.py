import pandas as pd

from pandas import Series,DataFrame

import matplotlib.pyplot as plt

import seaborn as sns

import ipywidgets as widgets

from ipywidgets import interact, interact_manual
df=pd.read_csv('../input/online-streaming/Mo.csv',index_col=False)

df1=pd.read_csv('../input/online-streaming/Mo.csv',index_col=False)

df.head()
df.shape
df.drop_duplicates(inplace=True)
df.info()
df.isnull().sum()
df.drop('Type',axis=1,inplace=True)
df.describe()
s = df['Genres'].str.split(',').apply(Series, 1).stack()

s.index = s.index.droplevel(-1)

s.name = 'Genres'

del df['Genres']

df_Genre = df.join(s)
d= df1['Language'].str.split(',').apply(Series, 1).stack()

d.index = d.index.droplevel(-1)

d.name = 'Language'

del df1['Language']

df1 = df1.join(d)
df1.head()
df_Genre.head()
Netflix=df['Netflix'].sum()

print("Number of movies on Netflix:",Netflix)
Hulu=df['Hulu'].sum()

print("Number of movies on Hulu:",Hulu)
Prime_Video=df['Prime Video'].sum()

print("Number of movies on Amazon Prime Video:",Prime_Video)
Disney=df['Disney+'].sum()

print("Number of movies on Disney+:",Disney)
Total=Netflix+Hulu+Prime_Video+Disney

print("The total number of movies on these online platform:",Total)
num_platform = (Netflix,Hulu,Prime_Video,Disney)

col_names = ('Netflix','Hulu','Prime Video','Disney+')

PlatformList = list(zip(col_names,num_platform))

PlatformCounts = pd.DataFrame(data=PlatformList,columns=['Platform','Number of Movie'])

PlatformCounts
print("The movies available in these platforms are from the year:",df['Year'].min(),"to:",df['Year'].max())
print("The genre of the movies are:",df_Genre['Genres'].unique())

print("The number of unique genres are:",df_Genre['Genres'].nunique())
print("Average run time of movies on these platforms:",df['Runtime'].mean())
df.hist(color='brown',figsize=(10,10))
df['Age'].replace("13+",13,inplace=True)

df['Age'].replace("18+",18,inplace=True)

df['Age'].replace("7+",7,inplace=True)

df['Age'].replace("all",0,inplace=True)

df['Age'].replace("16+",16,inplace=True)
df_year = pd.DataFrame(df.groupby(df['Year']).Title.nunique())

df_year.head()
df_year.nlargest(5,'Title')
df_year.plot(title='Movies made per year',color='red',kind='line')
df_Genre['Genres'].value_counts().plot(kind='barh',figsize=(10,10))
df['Age'].value_counts().plot(kind='barh')
print("The unique languages",df1['Language'].unique())

print("The number of unique languages:",df1['Language'].nunique())
@interact

def show_articles_more_than( x=3.1):

    return df.loc[df['IMDb'] >= x]

#interact will help you in toggling the age through the toggle button, it will only work in jupyter notebook
@interact

def show_articles_more_than(x=1000):

    print("The number of movies in the year:",x,"are:",df.loc[df['Year']==x].shape[0])

    return df.loc[df['Year'] ==x]
@interact

def show_articles_more_than(x=10):

    return df.loc[df['Age']<=x]

#age=0 means every individual can see that movie
df['Rotten Tomatoes'] = df['Rotten Tomatoes'].str.rstrip('%').astype('float') 
disney_avg_rt= round(df.loc[df['Disney+']==1]['Rotten Tomatoes'].mean(),1)

hulu_avg_rt=round(df.loc[df['Hulu']==1]['Rotten Tomatoes'].mean(),1)

netflix_avg_rt=round(df.loc[df['Netflix']==1]['Rotten Tomatoes'].mean(),1)

prime_avg_rt=round(df.loc[df['Prime Video']==1]['Rotten Tomatoes'].mean(),1)
disney_avg_imdb= round(df.loc[df['Disney+']==1]['IMDb'].mean(),1)

hulu_avg_imdb=round(df.loc[df['Hulu']==1]['IMDb'].mean(),1)

netflix_avg_imdb=round(df.loc[df['Netflix']==1]['IMDb'].mean(),1)

prime_avg_imdb=round(df.loc[df['Prime Video']==1]['IMDb'].mean(),1)
Net=df.loc[df['Netflix']==1]

Net.head()
Hu=df.loc[df['Hulu']==1]

Hu.head()
Pr=df.loc[df['Prime Video']==1]

Pr.head()
# create dataframe:

no_platform = (Netflix,Hulu,Prime_Video,Disney)

col_names = ('Netflix','Hulu','Prime Video','Disney+')

avg_imdb = (netflix_avg_imdb,hulu_avg_imdb,prime_avg_imdb,disney_avg_imdb)

avg_roto = (netflix_avg_rt,hulu_avg_rt,prime_avg_rt,disney_avg_rt)

List = list(zip(col_names,no_platform,avg_imdb,avg_roto))

Counts =  pd.DataFrame(data=List,columns=['Platform','Number of Movie','Average IMDb rate','Average % Rotten Tomattoes rate'])

Counts
sns.barplot(x='Platform',y='Number of Movie',data=PlatformCounts)
df['Rotten_t']=df['Rotten Tomatoes']/10
df.plot.scatter(x='IMDb', y='Rotten_t',title='Profit vs Vote Avg',color='DarkBlue',figsize=(6,5));