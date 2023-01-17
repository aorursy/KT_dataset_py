# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

data = pd.read_csv('../input/netflix-shows/netflix_titles.csv')
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Feature Engineering - Adding extra columns
data['date_added'] = pd.to_datetime(data['date_added'])
data['month'] = data['date_added'].dt.month #Extracting Month
data['year_added'] = data['date_added'].dt.year #Extracting Year
data_na = data.dropna(subset=['month']) #removing records where Month is NA  
data_na = data_na.dropna(subset=['country']).reset_index(drop=True)
country_list = data_na['country']
newlist = []
for d in country_list:
    d = d.split(",")
    newlist.extend(d) 

newlist = [x.strip() for x in newlist]
from collections import Counter
c = Counter(newlist)
df = pd.DataFrame.from_dict(c, orient = 'index').reset_index()
df.rename(columns = {'index': 'Country',0:'Occurences'},inplace=True)
df.sort_values(by = ['Occurences'], inplace=True,ascending = False)
top_10 = df.head(10)

#Creating pallette
pal = sns.color_palette("Blues_d", len(top_10['Occurences']))
rank = top_10["Occurences"].argsort().argsort() 
plt.style.use("seaborn-pastel")
plt.figure(figsize = (20,10))
sns.barplot(x='Country',y='Occurences',data=top_10, palette=np.array(pal[::-1])[rank])
plt.xlabel("")
plt.ylabel("")
plt.title("Top 10 Countries by Content Count",fontsize = 15)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
# plt.title()
plt.show()
pie_chart = data.groupby(['type']).count()['show_id'].reset_index()
total = len(data)
percent_l = []
for i in range((len(pie_chart))):
    percent_l.append((pie_chart['show_id'][i])/total)
pie_chart['Percentage'] = percent_l
plt.style.use('fivethirtyeight')
plt.figure(figsize=(7,7))
plt.rcParams.update({'font.size': 18})
plt.pie(pie_chart['Percentage'], labels=pie_chart['type'], shadow=True,
        startangle=90, autopct='%1.1f%%',
        wedgeprops={'edgecolor': 'black'})
plt.title("Content Type")
plt.tight_layout()
plt.show()
#Top 5 Countries
data_country = data.groupby(['country']).count()['show_id'].to_frame().reset_index()
data_country = data_country.sort_values('show_id',ascending=False)
data_country = data_country.head(5).reset_index(drop=True)
list_c = data_country['country']
top_8_data = data[data['country'].isin(list_c)]
year_with_show = top_8_data.groupby(['country','type']).count()['show_id'].to_frame().reset_index()
year_with_show = year_with_show.sort_values(by = 'show_id', ascending= False)
plt.style.use('seaborn-pastel')
plt.figure(figsize=(20,10))
sns.barplot(x="country", y="show_id", hue="type",data= year_with_show)
plt.xlabel("")
plt.title('Shows by Country')
plt.ylabel("")
plt.show()
plt.style.use('seaborn-pastel')
plt.figure(figsize=(20,10))
a = data.groupby(['year_added','type']).count()['show_id'].to_frame().reset_index()
a = a[a["year_added"]!=2020]
sns.lineplot(x="year_added",y= "show_id", hue = "type",data = a)
plt.xlabel("Year")
plt.ylabel("Number of contents")
plt.show()
months_v = data.groupby('month').count()['show_id'].to_frame().reset_index(drop=True)
months_l = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep' ,'Oct','Nov','Dec']
tuples = list(zip(months_l , months_v['show_id']))
months= pd.DataFrame(tuples , columns = ['Month','Values'])
plt.style.use('seaborn-pastel')
plt.figure(figsize=(20,10))
sns.barplot(x = months['Month'], y= months['Values'])
plt.xlabel("")
plt.ylabel("")
plt.title("Content Count by Month")
plt.show()
month_wise = data.groupby(['country']).count()['show_id'].to_frame().reset_index()
top_countries = data[data['country'].isin(list_c)]
temp_1 = top_countries.groupby(['country','month']).count()['show_id'].to_frame().reset_index()
temp_2 = top_countries.groupby(['country','month','type']).count()['show_id'].to_frame().reset_index()
list_temp_1 = ["United States","India", "United Kingdom", "Japan","Canada"]
for i in range(len(list_temp_1)):
    a = temp_1[temp_1["country"] == list_temp_1[i]]
    plot = sns.catplot(data=a, x="month",y='show_id',kind='bar',row='country',height=4,aspect=4,linewidth=2.5)

    plot.set_axis_labels("", "").set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep' ,'Oct','Nov','Dec'])

plt.show()
for i in range(len(list_temp_1)):
    a = temp_2[temp_2["country"] == list_temp_1[i]]
    ax5   = sns.catplot(data=a, x="month",y='show_id',hue = "type",kind='bar',
                        row='country',height=4,aspect=4,linewidth=2.5)
    ax5.set_axis_labels("", "Content Count").set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep' ,'Oct','Nov','Dec'])
rating_data = data.groupby("rating").count()['show_id'].to_frame().reset_index()
rating_data = rating_data.sort_values(by = 'show_id', ascending = False)
plt.style.use('seaborn-pastel')
plt.figure(figsize=(20,20))
sns.barplot(x="show_id",y="rating", data = rating_data)
plt.xlabel("")
plt.title("Count by Ratings")
plt.ylabel("")
plt.show()
temp_3 = top_countries.groupby(['country','rating']).count()['show_id'].to_frame().reset_index()

for i in range(len(list_temp_1)):
    a = temp_3[temp_3["country"] == list_temp_1[i]].sort_values(by = 'show_id', ascending=False)
    ax1 = sns.catplot(data=a, x="rating",y="show_id",
       kind='bar',row='country',height=4,aspect=4,
       linewidth=2.5)
    ax1.set_axis_labels("", "")
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
list_temp = []

description_list = data.dropna(subset=['description']).reset_index(drop=True)
description_list = description_list['description']
list_temp = []
for b in description_list:
    b=b.split(",")
    list_temp.extend(b)

list_temp = pd.Series(list_temp)
plt.figure(figsize=(20,20))
stopwords = set(STOPWORDS)
stopwords.update(["turn", "one", "two", "become", "three","take","new","four","must","takes","make","find","finds"])
# Create and generate a word cloud image:
wordcloud = WordCloud(stopwords=stopwords, background_color="black").generate(','.join(list_temp))

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
data_movie=data.loc[(data['type']=="Movie")]
data_movie = data_movie.dropna(subset = ['rating'])
data_movie[['min','rest']]=data_movie.duration.str.split(" ",expand = True)
data_movie = data_movie.drop(['rest'],axis=1)
data_movie['min']=data_movie['min'].astype(float)
plt.figure(figsize=(20,20))
sns.boxplot(x='rating', y='min', data=data_movie)
sns.swarmplot(x='rating', y='min', data=data_movie,size=8,alpha=0.2,color=".2")
data_movie.groupby(['rating'])['min'].median()
list_director = []
data_new = data[data['director'].notnull()]
d_list = data_new['director']
for a in d_list:
    a= a.split(",")
    list_director.extend(a)

f= Counter(list_director)

df_d = pd.DataFrame.from_dict(f, orient = 'index').reset_index()
df_d.rename(columns = {'index': 'Director',0:'Occurences'},inplace=True)
df_d.sort_values(by = ['Occurences'], inplace=True,ascending = False)
top_10_d = df_d.head(10)
plt.figure(figsize=(20,10))
sns.barplot(x="Occurences",y="Director", data = top_10_d)
plt.title("Top Directors with Content Count")
plt.xlabel("")
plt.ylabel("")
plt.xlim(1,19)
plt.show()