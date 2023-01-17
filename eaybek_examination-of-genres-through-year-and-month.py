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


import json
import pprint
pp = pprint.PrettyPrinter(indent=2,width=80)
import matplotlib.pyplot as plt

%matplotlib inline
credits=pd.read_csv("../input/tmdb_5000_credits.csv")
movies=pd.read_csv("../input/tmdb_5000_movies.csv")
from termcolor import colored
def ht_title(title='',length=80, color='blue'):
    """
    helper
    hashtag title function
    this function is print title in center of print(length*"#")
    title:  Message what you say
    length: Column number
    
    """
    if length < 2:
        raise Exception("length must greater than 2")
    len_of_title = len(title)            
    if len_of_title == 0:
        left_space=length/2
        rigth_space=length/2
    elif len_of_title < 0 or len_of_title > length - 2:
        raise Exception("Out of range")
    elif len_of_title % 2 == 0:
        left_space=(length - len_of_title) / 2
        if length % 2==0:
            rigth_space=left_space
        else:
            rigth_space=left_space+1
    elif len_of_title % 2 == 1:
        left_space=(length - len_of_title) / 2
        if length % 2 == 0:
            rigth_space=left_space+1
        else:
            rigth_space=left_space
    print(colored(length*"#", color))
    print(colored("#", color),end='')
    print((int(left_space)-1)*" ",end='')
    print(colored(title.upper(), color),end='')
    print((int(rigth_space)-1)*" ",end='')
    print(colored("#",color))
    print(colored(length*"#",color))
        
#ht_title(length=1)
ht_title('OK!',length=10)
ht_title('OK',length=9,color='yellow')
ht_title('OK',length=10,color='green')
ht_title('OK!',length=9,color='red')
ht_title("credits info")
pp.pprint(credits.info())
spp = pprint.PrettyPrinter(indent=2,width=30)
ht_title("credits features")
spp.pprint(list(credits.columns))
ht_title("credits dtypes")
pp.pprint(credits.dtypes)
ht_title("credits describe")
credits.describe()
ht_title("movies info")
pp.pprint(movies.info())
ht_title("movies features")
pp.pprint(list(movies.columns))
ht_title("movies dtypes")
pp.pprint(movies.dtypes)
ht_title("movies describe")
movies.describe()
movies["release_month"]=[str(each)[5:7] if str(each) != "nan" else "00" for each in movies.release_date]
movies["release_month_int"]=[int(each) for each in movies.release_month]

movies["release_year"]=[str(each)[0:4] if str(each) != "nan" else "1915" for each in movies.release_date]#1915 is nan value 
movies["release_year_int"]=[int(each) for each in movies.release_year]
movies.head()
plt.figure(figsize=(24,6))
plt.title("Histogram of released movies through year")
plt.grid(True)
plt.xticks(range(1915,2021),rotation=90)
plt.yticks(range(0,251,10),rotation=25)
plt.xlabel('year')
plt.ylabel('movie count')
plt.hist(movies.release_year_int,bins=100)
plt.show()
months=['January', 'Feburary', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
plt.figure(figsize=(24,6))
plt.title("Histogram of released movies through month")
plt.grid(True)
plt.xticks(range(1,13),months,rotation=90)
plt.yticks(rotation=25)
plt.xlabel('month')
plt.ylabel('movie count')
plt.hist(movies.release_month_int,bins=18)
plt.show()
genre_tbl=pd.DataFrame({"genre":[],"original_title":[],"release_year":[],"release_month":[]})
index=1
for i,each in enumerate(movies.genres):
    genre_list=json.loads(each)
    for genre in genre_list:
        genre_tbl=pd.concat([genre_tbl,pd.DataFrame({
            "genre":genre["name"],
            "original_title":movies.original_title[i],
            "release_year":movies.release_year[i],
            "release_month":movies.release_month[i]
        },index=[index])],axis=0)
        index+=1

genre_tbl.head()
genre_tbl.tail()
x=genre_tbl.genre.unique()
y=genre_tbl.release_year.sort_values().unique()
z=genre_tbl.release_month.sort_values().unique()
pairs_of_year=np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])
pairs_of_month=np.transpose([np.tile(x, len(z)), np.repeat(z, len(x))])



pairs_of_year
pairs_of_month
year_hist={}
for genre,year in pairs_of_year:
    if genre not in year_hist:
        year_hist[genre]=[]
    year_hist[genre].append((year,genre_tbl[(genre_tbl.genre==genre) & (genre_tbl.release_year==year)].count()[0]))
#year_hist
month_hist={}
for genre,month in pairs_of_month:
    if genre not in month_hist:
        month_hist[genre]=[]
    month_hist[genre].append((month,genre_tbl[(genre_tbl.genre==genre) & (genre_tbl.release_month==month)].count()[0]))
#month_hist
plt.figure(figsize=(24,6))
plt.title("released movie genres through year")
c=3
for genre,values in year_hist.items():
    x=[]
    y=[]
    c=+20
    for i in values:
        x.append(i[0])
        y.append(i[1])
    plt.plot(x,y,label=str(genre))

plt.legend()
plt.grid(True)
plt.yticks(range(0,131,10))
plt.xticks(rotation=90)
plt.ylabel("how many movies are released")
plt.xlabel("year")
plt.show()
plt.figure(figsize=(12,12))
plt.title("released movie genres pie chart in all years")
c=3
x=[]
y=[]
for genre,values in year_hist.items():
    x.append(genre)
    s=0
    for i in values:
        s+=i[1]
    y.append(s)
v=[(i/sum(y))*100 for i in y]
explode=(0,0,0,0,0,0.1,0,0,0,0,0.1,0,0,0,0,0,0,0,0,0)

plt.pie(v,explode=explode,labels=x,autopct='%1.1f%%',)
plt.show()
plt.figure(figsize=(24,6))
plt.title("released movie genres through month")
c=3
for genre,values in month_hist.items():
    x=[]
    y=[]
    c=+20
    for i in values:
        x.append(i[0])
        y.append(i[1])
    plt.plot(x,y,label=str(genre))
months=['January', 'Feburary', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

plt.legend()
plt.grid(True)
plt.yticks()
plt.xticks(range(1,13),months,rotation=90)
plt.ylabel("how many movies are released")
plt.xlabel("month")
plt.show()