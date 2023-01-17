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
import pandas as pd

pd.set_option('mode.chained_assignment', None)      # To suppress pandas warnings.

pd.set_option('display.max_colwidth', -1)           # To display all the data in each column

pd.options.display.max_columns = 50                 # To display every column of the dataset in head()



import warnings

warnings.filterwarnings('ignore')                   # To suppress all the warnings in the notebook.



import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns

sns.set(style='whitegrid', font_scale=1.3, color_codes=True)      # To apply seaborn styles to the plots.

# Making plotly specific imports

# These imports are necessary to use plotly offline without signing in to their website.



from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

import chart_studio.plotly as py

from plotly import tools

init_notebook_mode(connected=True)
moviesdf=pd.read_csv('/kaggle/input/imdb-data/IMDB-Movie-Data.csv')

moviesdf.head()
moviesdf.shape
moviesdf.columns
moviesdf.info()
moviesdf.describe()
#Dropping the column - Description

moviesdf.drop(['Description'], 1, inplace=True)
#Changing columns names for Revenue(Millions) to Revenue and Runtime(minutes) to Runtime

moviesdf.rename(columns={'Revenue (Millions)':'Revenue','Runtime (Minutes)':'Runtime'}, inplace=True)
moviesdf.head()
#Count of missing values in each column

moviesdf.isna().sum()
#Checking the count where Revenue has missing value

moviesdf[moviesdf['Revenue'].isnull()].isnull().sum()
#Checking the count where Metascore has missing value

moviesdf[moviesdf['Metascore'].isnull()].isnull().sum()
moviesdf[moviesdf['Revenue'].isnull() & moviesdf['Metascore'].isnull()].isnull().sum()
moviesdf[moviesdf['Title']=='The Host']
revmoviesdf=moviesdf.dropna(subset=['Revenue'])

revmoviesdf.isna().sum()
#Top 10 earning movies

revmoviesdf.sort_values(by=['Revenue'], ascending=False)[:10]
revmoviesdf.groupby(['Title'])['Revenue'].mean().sort_values(ascending=False)[:10].plot(kind='bar', figsize=(15,8), fontsize=13, color='green')

plt.ylabel('Revenue')

plt.title("Highest earning movies")
#10 lowest earning movies

revmoviesdf.sort_values(by=['Revenue'])[:10]
revmoviesdf.groupby(['Title'])['Revenue'].mean().sort_values(ascending=True)[:10].plot(kind='bar', figsize=(15,8), fontsize=13, color='red')

plt.ylabel('Revenue')

plt.title("Lowest earning movies")
#Highest earning movies year-wise

revmoviesdf.sort_values(by=['Revenue'], ascending=False).groupby('Year').first()
tmp=revmoviesdf.sort_values(by=['Revenue'], ascending=False).groupby('Year').first()

tmp.groupby(['Title','Year'])['Revenue'].sum().sort_values().plot.bar(x='Title', y='Revenue', figsize=(10,8))

plt.ylabel('Revenue in Millions(USD)')

plt.title("Highest earning movies by years")
revmoviesdf.groupby(['Year'])['Revenue'].sum().plot(kind='bar', figsize=(15,8), fontsize=13, color='blue')

plt.ylabel('Revenue (Milllion USD)')

plt.title("Total Revenue By Years")
revmoviesdf.groupby(['Year'])['Revenue'].sum().plot(kind='line', figsize=(15,8), fontsize=13, color='blue')

plt.ylabel('Revenue')

plt.title("Total Revenue By Years")
#Revenue distribution

sns.distplot(revmoviesdf['Revenue']).set_title("Revenue distribution for movies")
moviesdf.groupby(['Year'])['Title'].count().plot(kind='bar', figsize=(15,8), fontsize=13, color='yellow')

plt.ylabel('Number of Titles')

plt.title("Number of Movies released by Years")
revmoviesdf.groupby(['Year'])['Revenue'].sum()
revmoviesdf.groupby(['Year'])['Revenue'].mean()
revmoviesdf.groupby(['Year'])['Revenue'].mean().plot(kind='bar', figsize=(15,8), fontsize=13, color='orange')

plt.ylabel('Revenue')

plt.title("Average Revenue By Years")
revmoviesdf.groupby(['Year'])['Revenue'].mean().plot(kind='line', figsize=(15,8), fontsize=13, color='orange')

plt.ylabel('Revenue')

plt.title("Average Revenue By Years")
corr = moviesdf.corr()



figure = plt.figure(figsize=(15,10))



sns.heatmap(data=corr, annot=True,cmap='viridis',xticklabels=True, yticklabels=True).set_title("Relation betweem Movie dataset fields")
sns.pairplot(moviesdf)
moviesdf.groupby(['Year'])['Votes'].sum()
#Number of votes over the years

moviesdf.groupby(['Year'])['Votes'].sum().plot(kind='line', figsize=(15,8), fontsize=13, color='orange')

plt.ylabel('Number of votes')

plt.title("Number of Votes by Years")
moviesdf.groupby(['Year'])['Votes'].mean()
#Average number of votes over the years

moviesdf.groupby(['Year'])['Votes'].mean().plot(kind='line', figsize=(15,8), fontsize=13, color='orange')

plt.ylabel('Number of Votes')

plt.title("Average number of Votes by Years")
plt.figure(figsize=(10,6))

plt.title("Votes and Year relation")

sns.regplot(data=moviesdf, x="Year", y="Votes", color='orange')

plt.ylabel("Number of Votes")
moviesdf.groupby(['Year'])['Rating'].count()
moviesdf.groupby(['Year'])['Rating'].mean()
#Average ratings over the years

moviesdf.groupby(['Year'])['Rating'].mean().plot(kind='line', figsize=(15,8), fontsize=13, color='violet')

plt.ylabel('Ratings')

plt.title("Average Ratings by Years")
plt.figure(figsize=(10,6))

plt.title("Ratings and Year relation")

sns.regplot(data=moviesdf, x="Year", y="Rating", color='violet')

plt.ylabel("Ratings")
#Drop the rows with missing metascore values

metamoviedf=moviesdf.dropna(subset=['Metascore'])
metamoviedf.groupby(['Year'])['Metascore'].mean()
#Average Metascore over the years

metamoviedf.groupby(['Year'])['Metascore'].mean().plot(kind='line', figsize=(15,8), fontsize=13, color='blue')

plt.ylabel('Metascore')

plt.title("Average Metascore by Years")
plt.figure(figsize=(10,6))

plt.title("Metascore and Year relation")

sns.regplot(data=metamoviedf, x="Year", y="Metascore", color='blue')

plt.ylabel("Metascore")
tmp=moviesdf.sort_values(by=['Rating'], ascending=False).groupby('Year').first()

tmp.groupby(['Title','Year'])['Rating'].mean()
tmp=moviesdf.sort_values(by=['Rating'], ascending=False).groupby('Year').first()

tmp.groupby(['Title','Year'])['Rating'].mean().sort_values().plot.bar(x='Title', y='Rating', figsize=(10,8), color='purple')

plt.ylabel('Ratings')

plt.title("Highest rated movies by years")
tmp=metamoviedf.sort_values(by=['Metascore'], ascending=False).groupby('Year').first()

tmp.groupby(['Title','Year'])['Metascore'].mean()
tmp=metamoviedf.sort_values(by=['Metascore'], ascending=False).groupby('Year').first()

tmp.groupby(['Title','Year'])['Metascore'].mean().sort_values().plot.bar(x='Title', y='Metascore', figsize=(10,8), color='violet')

plt.ylabel('Metascore')

plt.title("Highest Metascore movies by years")
plt.figure(figsize=(10,6))

plt.title("Revenue vs votes")

sns.regplot(data=revmoviesdf, x="Revenue", y="Votes")

plt.ylabel("Votes")
plt.figure(figsize=(10,6))

plt.title("Revenue vs ratings")

sns.regplot(data=revmoviesdf, x="Revenue", y="Rating", color='orange')

plt.ylabel("Rating")
tmp=moviesdf.dropna(subset=['Revenue','Metascore'])
plt.figure(figsize=(10,6))

plt.title("Revenue vs Metascore")

sns.regplot(data=tmp, x="Revenue", y="Metascore", color='green')

plt.ylabel("Metascore")
plt.figure(figsize=(10,6))

plt.title("Revenue vs Runtime")

sns.regplot(data=revmoviesdf, x="Revenue", y="Runtime")

plt.ylabel("Runtime")
plt.figure(figsize=(10,6))

plt.title("Ratings vs Votes")

sns.regplot(data=moviesdf, x="Rating", y="Votes", color='red')

plt.ylabel("Votes")
plt.figure(figsize=(10,6))

plt.title("Metascores vs Votes")

sns.regplot(data=metamoviedf, x="Metascore", y="Votes", color='violet')

plt.ylabel("Votes")
plt.figure(figsize=(10,6))

plt.title("Metascores vs Ratings")

sns.regplot(data=metamoviedf, x="Metascore", y="Rating", color='purple')

plt.ylabel("Rating")
#get genre list sorted by revenue and stored in the list

genre_list=moviesdf.sort_values(by='Revenue', ascending=False).Genre

genre_list
#using counter to count the occurence of each unique genre element in the list genrated above.

from collections import Counter 



genlist=[]

for genre in genre_list:

  tmp=[]

  tmp=genre.split(',')

  genlist.extend(tmp)

  

#print(genlist)



mycounter=Counter(genlist)

print(mycounter)



#print(mycounter.keys())

#print(mycounter.values())
#Empty dictionary

genre_dict=dict.fromkeys(genlist,0)

print(genre_dict)



#print(type(genre_dict.keys()))

#print(type(genre_dict.values()))
#Traversing dataframe and storing data in dictionary, key-genre, value-total revenue for genre calcualted over the years

genredict=dict()

for idx in moviesdf.index:

  if (moviesdf['Revenue'][idx]>=0):

    if moviesdf['Genre'][idx] in genredict:

      genredict[moviesdf['Genre'][idx]]+=moviesdf['Revenue'][idx]

    else:

       genredict[moviesdf['Genre'][idx]]=moviesdf['Revenue'][idx]



for k,v in genre_dict.items():

  for key, val in genredict.items():

    tmplist=[]

    tmplist.extend(key.split(','))

    if (k in tmplist):

      genre_dict[k]+=val



for k,v in genre_dict.items():

  print ("Genre : {}, Revenue : {}".format(k,v))
list(genre_dict.keys())
tuple(genre_dict.values())
fig = go.Figure([go.Bar(x=list(genre_dict.keys()), y=tuple(genre_dict.values()))])



fig.update_layout(

    title="Genre with highest revneue",

    xaxis_title="Genre",

    yaxis_title="Revenue(in Millions USD)")



fig.show()
#Getting unique year from the list from above section

yeararr=revmoviesdf['Year'].unique()

yeararr=np.sort(yeararr)

yeararr



first=True



#for each year, we are traversing the genre element in each movie and storing the revenue generated by each element along 

#with genre element in a dictionary and further creating dataframe from the dictionary

for year in yeararr:

    genrevdict=dict()

    genrevdict=dict.fromkeys(genlist,0)

    genrevdict['Year']=year

    

    tmpdf=revmoviesdf[revmoviesdf['Year']==year]



    total=0

    for idx in tmpdf.index:

        revlist=[]

        revlist=tmpdf['Genre'][idx].split(',')

        for genre in revlist:

            if genre in genrevdict.keys():

                genrevdict[genre]+=tmpdf['Revenue'][idx]

            else:

                   genrevdict[genre]=tmpdf['Revenue'][idx]

        total+=tmpdf['Revenue'][idx]

        genrevdict["Total"]=total



    if (first==True):

        revenuedf=pd.DataFrame(genrevdict, index=[0])

        first=False

    else:

         revenuedf=revenuedf.append(genrevdict, ignore_index=True)

            

revenuedf  
year=[2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016]





g1=revenuedf.groupby(['Year'])['Total'].sum().array

g2=revenuedf.groupby(['Year'])['Action'].sum().array

g3=revenuedf.groupby(['Year'])['Adventure'].sum().array

g4=revenuedf.groupby(['Year'])['Fantasy'].sum().array

g5=revenuedf.groupby(['Year'])['Sci-Fi'].sum().array

g6=revenuedf.groupby(['Year'])['Crime'].sum().array

g7=revenuedf.groupby(['Year'])['Drama'].sum().array

g8=revenuedf.groupby(['Year'])['Animation'].sum().array

g9=revenuedf.groupby(['Year'])['Comedy'].sum().array

g10=revenuedf.groupby(['Year'])['Thriller'].sum().array

g11=revenuedf.groupby(['Year'])['Mystery'].sum().array

g12=revenuedf.groupby(['Year'])['Family'].sum().array

g13=revenuedf.groupby(['Year'])['Biography'].sum().array

g14=revenuedf.groupby(['Year'])['Horror'].sum().array

g15=revenuedf.groupby(['Year'])['Sport'].sum().array

g16=revenuedf.groupby(['Year'])['War'].sum().array

g17=revenuedf.groupby(['Year'])['Romance'].sum().array

g18=revenuedf.groupby(['Year'])['Music'].sum().array

g19=revenuedf.groupby(['Year'])['History'].sum().array

g20=revenuedf.groupby(['Year'])['Western'].sum().array

g21=revenuedf.groupby(['Year'])['Musical'].sum().array



plt.bar(year, g1, color = '#eec900')

plt.bar(year, g2, color = '#44c9c6', bottom=g1)

plt.bar(year, g3, color = '#58dae4', bottom=g1+g2)

plt.bar(year, g4, color = '#39af8e', bottom=g1+g2+g3)

plt.bar(year, g5, color = '#3e4f6a', bottom=g1+g2+g3+g4)

plt.bar(year, g6, color = '#2eaf57', bottom=g1+g2+g3+g4+g5)

plt.bar(year, g7, color = '#eee7ea', bottom=g1+g2+g3+g4+g5+g6)

plt.bar(year, g8, color = '#6ca0c5', bottom=g1+g2+g3+g4+g5+g7)

plt.bar(year, g9, color = '#1ba1e2', bottom=g1+g2+g3+g4+g5+g7+g8)

plt.bar(year, g10, color = '#008080', bottom=g1+g2+g3+g4+g5+g7+g8+g9)

plt.bar(year, g11, color = '#420420', bottom=g1+g2+g3+g4+g5+g7+g8+g9+g10)

plt.bar(year, g12, color = '#110044', bottom=g1+g2+g3+g4+g5+g7+g8+g9+g10+g11)

plt.bar(year, g13, color = '#110011', bottom=g1+g2+g3+g4+g5+g7+g8+g9+g10+g11+g12)

plt.bar(year, g14, color = '#333300', bottom=g1+g2+g3+g4+g5+g7+g8+g9+g10+g11+g12+g13)

plt.bar(year, g15, color = '#688248', bottom=g1+g2+g3+g4+g5+g7+g8+g9+g10+g11+g12+g13+g14)

plt.bar(year, g16, color = '#cda1ac', bottom=g1+g2+g3+g4+g5+g7+g8+g9+g10+g11+g12+g13+g14+g15)

plt.bar(year, g17, color = '#cc0066', bottom=g1+g2+g3+g4+g5+g7+g8+g9+g10+g11+g12+g13+g14+g15+g16)

plt.bar(year, g18, color = '#ff003c', bottom=g1+g2+g3+g4+g5+g7+g8+g9+g10+g11+g12+g13+g14+g15+g16+g17)

plt.bar(year, g19, color = '#b05f1b', bottom=g1+g2+g3+g4+g5+g7+g8+g9+g10+g11+g12+g13+g14+g15+g16+g17+g18)

plt.bar(year, g20, color = '#f9d7c0', bottom=g1+g2+g3+g4+g5+g7+g8+g9+g10+g11+g12+g13+g14+g15+g16+g17+g18+g19)

plt.bar(year, g21, color = '#b87624', bottom=g1+g2+g3+g4+g5+g7+g8+g9+g10+g11+g12+g13+g14+g15+g16+g17+g18+g19+g20)





plt.legend(labels=('Total', 'Action', 'Adventure','Fantasy', 'Sci-Fi', 'Crime', 'Drama', 'Animation', 'Comedy', 'Thriller',

                  'Mystery', 'Family', 'Biography', 'Horror', 'Sport', 'War', 'Romance', 'Music', 'History', 'Western', 'Musical'))

plt.xlabel("Year")

plt.ylabel("Total Revenue")



fig_size = plt.rcParams["figure.figsize"]

print ("Current size:", fig_size)

fig_size[0] = 20

fig_size[1] = 20

plt.rcParams["figure.figsize"] = fig_size



plt.show()
revmoviesdf.groupby(['Director'])['Revenue'].sum().sort_values(ascending=False)
revmoviesdf.groupby(['Director'])['Revenue'].sum().sort_values(ascending=False)[:20].plot(kind='bar', figsize=(15,8), fontsize=13, color='orange')

plt.ylabel('Revenue(in Millions USD)')

plt.title("Director-wise revenue")
revmoviesdf[revmoviesdf['Director']=='J.J. Abrams']
revmoviesdf[revmoviesdf['Director']=='David Yates']
revmoviesdf[revmoviesdf['Director']=='Christopher Nolan']
moviesdf.groupby(['Director'])['Votes'].sum().sort_values(ascending=False)[:20]
moviesdf.groupby(['Director'])['Votes'].sum().sort_values(ascending=False)[:20].plot(kind='bar', figsize=(15,8), fontsize=13, color='red')

plt.ylabel('Votes')

plt.title("Directors with maximum votes received")
moviesdf.groupby(['Director'])['Rating'].mean().sort_values(ascending=False)[:20]
moviesdf.groupby(['Director'])['Rating'].mean().sort_values(ascending=False)[:20].plot(kind='bar', figsize=(15,8), fontsize=13, color='green')

plt.ylabel('Ratings')

plt.title("Directors with maximum Ratings received")
metamoviedf.groupby(['Director'])['Metascore'].mean().sort_values(ascending=False)[:20]
metamoviedf.groupby(['Director'])['Metascore'].mean().sort_values(ascending=False)[:20].plot(kind='bar', figsize=(15,8), fontsize=13, color='yellow')

plt.ylabel('Metascore')

plt.title("Directors with maximum Metascores received")