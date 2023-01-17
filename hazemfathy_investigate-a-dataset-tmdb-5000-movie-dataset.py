#  all of the packages i plan to use

import pandas as pd

import numpy as nb

import matplotlib.pyplot as plt



from pandas import Series, DataFrame



# Load your data and print out a few lines. Perform operations to inspect data

df=pd.read_csv('../input/tmdb-movies-dataset/tmdb_movies_data.csv')

df.head(3)

#   number of rows and columns

df.shape
df.describe()
df.head()
# missing data

df.isnull().sum()
#  explore data types

df.dtypes
# convert release date from string ro date

df.release_date=pd.to_datetime(df.release_date) 

df.dtypes
#Datafram df_Q1 for Question 1

df_Q1=df 

#Datafram  df_Q2 for Question2 

df_Q2=df
df_Q2.shape
# drop null values in genres 

df_Q1.genres.dropna(inplace=True)

## delete rows with 0 

df_Q1.query('popularity==0').size 
#  duplicated movie titles

df_Q1.original_title.duplicated().sum()
# delete duplicated movie titles

df_Q1.original_title.drop_duplicates(keep='first',inplace=True)

df_Q1.original_title.duplicated().sum()
s = df_Q1['genres'].str.split('|').apply(Series, 1).stack()

s.index = s.index.droplevel(-1)

s.name = 'genres'

del df_Q1['genres']

dt=df_Q1.join(s)

dt.head(20)

 
df_Q2.shape
# explor movies without revenue

df_Q2[df_Q2['revenue']==0].shape
# for qustion 2 i have to delete All Rows with revenue =0 

df_Q2.drop(df_Q2[df_Q2['revenue']==0].index,inplace=True)

df_Q2.query('revenue ==0').size
# explor movie without Budjet

df_Q2.query('budget_adj==0').size
# delete all rows with budget 0

df_Q2.drop(df_Q2[df_Q2['budget_adj']==0].index,inplace=True)

df_Q2.shape # my sample is 3855 from 10866
# Divid years to decade to facilitate data analysing

#   

def decade(x):

    if ((x>= 1960) and (x<1970)): x= "Decade 60_70"

    elif((x>=1970) and (x<1980)): x="Decade 70_80"

    elif((x>=1980) and (x<1990)): x="Decade 80_90"

    elif((x>=1990) and (x<2000)): x="Decade 90_2000"

    elif((x>=2000) and (x<2010)): x="Decade 2000_2010"

    else: x="from 2010 until 2014"

    return x;

        

dt['Decade']=dt['release_year'].apply(lambda x : decade(x))

dt.head(5)

# calculate the average of popularity for each movie kindes and Decades

df_Q1_EX=dt.groupby(['Decade','genres'],as_index=False).popularity.mean()

df_Q1_EX.shape
##divide date fram to slices each of them represent specific decade

df_Q1_EX_decade1=df_Q1_EX.query('Decade=="Decade 60_70"')

df_Q1_EX_decade2=df_Q1_EX.query('Decade=="Decade 70_80"')

df_Q1_EX_decade3=df_Q1_EX.query('Decade=="Decade 80_90"')

df_Q1_EX_decade4=df_Q1_EX.query('Decade=="Decade 90_2000"')

df_Q1_EX_decade5=df_Q1_EX.query('Decade=="Decade 2000_2010"')

df_Q1_EX_decade6=df_Q1_EX.query('Decade=="from 2010 until 2014"')



df_Q1_EX_decade6.head()
#drowing plot represents genres which are  the most popular from year to year using dataframe slices

%matplotlib inline

labels =df_Q1_EX_decade1['genres'].values

x=nb.arange(len(labels))

width = 0.15

y=df_Q1_EX_decade1['popularity'].values

fig, ax = plt.subplots(figsize=(12, 12))

decade1=ax.barh(x-(2*width),y,width,label='Decade 60_70')

############################################################################################

y2=df_Q1_EX_decade2['popularity'].values

decade2=ax.barh(x-width,y2,width,label='Decade 70_80')

############################################################################################

y3=df_Q1_EX_decade3['popularity'].values

decade3=ax.barh(x,y3,width,label='Decade 80_90')

###########################################################################################

y4=df_Q1_EX_decade4['popularity'].values

decade4=ax.barh(x+width,y4,width,label='Decade 90_2000')

######################################################

y5=df_Q1_EX_decade5['popularity'].values

decade4=ax.barh(x+(2*width),y5,width,label='Decade 2000_2010')

###########################################################

y6=df_Q1_EX_decade6['popularity'].values

decade6=ax.barh(x+(3*width),y6,width,label='from 2010 until 2014')

ax.set_xlabel('popularity')

ax.set_title('popularity and genres')

ax.set_yticks(x)

ax.set_yticklabels(labels)

ax.legend(loc='best', fontsize=10)

ax.xaxis.grid(True, linestyle='--', which='major',

                   color='grey', alpha=.25)









# To get movies with highest revenues i took highest 300 Revenues

# df_Q2R data frame relate to question 2

df_Q2R=df_Q2[['revenue','budget_adj','original_title','vote_average']]

df_Q2R=df_Q2R.nlargest(200,['revenue'])

df_Q2R.head(20)
df_Q2R.plot.line(x='budget_adj',y='revenue',figsize=(12, 10), linewidth=2.5, color='maroon')

plt.ylabel("Revenue", labelpad=15)

plt.xlabel("Budget")

plt.title("Relation between Revenue and Budget associated with movies that have high Revenues (highest 300)", y=1.02,fontsize=15)
# The graph to vote on the top 300 films in terms of revenue

df_Q2R.hist(column='vote_average')