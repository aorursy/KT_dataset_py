import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

#ignore warning messages

import warnings

warnings.filterwarnings('ignore')

import plotly

plotly.offline.init_notebook_mode(connected=True)

import plotly.graph_objs as go

from wordcloud import WordCloud
df=pd.read_csv("../input/google-play-store-apps/googleplaystore.csv")

df
#total no of Apps before cleaning

print(df.shape)

#Checking null value

print(df.isnull().values.any())
#dropping the rows which has null value 

df = df.dropna(how='any',axis=0) 
#printing shape after removing the rows which has null values

print(df.shape)
#removing the dupilcate apps

df.drop_duplicates(subset='App', inplace=True)
print(df.shape)
print(df.shape)
Category=df['Category'].value_counts()

print(Category.shape)

Category
data = [go.Pie(

        labels = Category.index,

        values = Category.values,

        hoverinfo = 'label+value'

    

)]

plotly.offline.iplot(data, filename='active_category')
Content_Rating=df['Content Rating'].value_counts().sort_values(ascending=False)

print(Content_Rating.shape)

Content_Rating
plt.figure(figsize=(12,12))

sns.barplot(Content_Rating.index, Content_Rating.values, alpha=0.8)

plt.title('Content Rating vs No Apps')

plt.ylabel('Apps')

plt.xlabel('Content Rating')

plt.show()
Type=df['Type'].value_counts().sort_values(ascending=False)

print(Type.shape)

Type
plt.figure(figsize=(8,8))

sns.barplot(Type.index, Type.values, alpha=0.8)

plt.title('Content Rating vs No Apps')

plt.ylabel('Apps')

plt.xlabel('Content Rating')

plt.show()
print('Average app rating = ', np.mean(df['Rating']))

sns.distplot(df['Rating'].values)
g = sns.catplot(x="Category",y="Rating",data=df, kind="box", height = 10 ,

palette = "Set1")

g.despine(left=True)

g.set_xticklabels(rotation=90)

g.set( xticks=range(0,34))

g = g.set_ylabels("Rating")

plt.title('Boxplot of Rating VS Category',size = 20)
Apps1000K=df[df['Installs'] =="1,000,000,000+"]

Apps500K=df[df['Installs'] == "500,000,000+"]

Apps100K=df[df['Installs'] == "100,000,000+"]

fames=[Apps1000K, Apps500K, Apps100K]

topApp = pd.concat(fames)
MostDownloadedApp=[]

for x in Apps1000K['App']:

    MostDownloadedApp.append(x.replace(" ",""))

wordcloud = WordCloud(width = 800, height = 800,

            background_color ='white').generate(str(MostDownloadedApp))

plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

plt.show()
Category=Apps1000K['Category'].value_counts()

plt.figure(figsize=(12,12))

g=sns.barplot(Category.index, Category.values, alpha=0.8)

g.set_xticklabels(rotation=90,labels=Category.index)

plt.title('Content Rating vs No Apps')

plt.ylabel('Number of Apps')

plt.xlabel('Category')

plt.show()
MostDownloadedApp=[]

for x in Apps500K['App']:

    MostDownloadedApp.append(x.replace(" ",""))

wordcloud = WordCloud(width = 800, height = 800, 

            background_color ='white').generate(str(MostDownloadedApp))

plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

plt.show()

Category=Apps500K['Category'].value_counts()

plt.figure(figsize=(12,12))

g=sns.barplot(Category.index, Category.values, alpha=0.8)

g.set_xticklabels(rotation=90,labels=Category.index)

plt.title('Content Rating vs No Apps')

plt.ylabel('Number of Apps')

plt.xlabel('Category')

plt.show()
MostDownloadedApp=[]

for x in Apps100K['App']:

    MostDownloadedApp.append(x.replace(" ",""))

wordcloud = WordCloud(width = 800, height = 800, 

            background_color ='white').generate(str(MostDownloadedApp))

plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

plt.show()

Category=Apps100K['Category'].value_counts()

plt.figure(figsize=(12,12))

g=sns.barplot(Category.index, Category.values, alpha=0.8)

g.set_xticklabels(rotation=90,labels=Category.index)

plt.title('Content Rating vs No Apps')

plt.ylabel('Number of Apps')

plt.xlabel('Category')

plt.show()
MostDownloadedApp=[]

for x in topApp['App']:

    MostDownloadedApp.append(x.replace(" ",""))

wordcloud = WordCloud(width = 800, height = 800, 

            background_color ='white').generate(str(MostDownloadedApp))

plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

plt.show()

Category=topApp['Category'].value_counts()

plt.figure(figsize=(12,12))

g=sns.barplot(Category.index, Category.values, alpha=0.8)

g.set_xticklabels(rotation=90,labels=Category.index)

plt.title('Content Rating vs No Apps')

plt.ylabel('Number of Apps')

plt.xlabel('Category')

plt.show()
Type=topApp['Type'].value_counts()
data = [go.Pie(

        labels = Type.index,

        values = Type.values,

        hoverinfo = 'label+value'

    

)]

plotly.offline.iplot(data, filename='active_category')
g = sns.kdeplot(topApp.Rating, color="Red", shade = True)

g.set_xlabel("Rating")

g.set_ylabel("Frequency")

topRated=df[df['Rating']>4.5]
MostRatedApp=[]

for x in topRated['App']:

    MostRatedApp.append(x.replace(" ",""))

wordcloud = WordCloud(width = 800, height = 800, 

            background_color ='white').generate(str(MostRatedApp))

plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

plt.show()
Category=topRated['Category'].value_counts()

plt.figure(figsize=(12,12))

g=sns.barplot(Category.index, Category.values, alpha=0.8)

g.set_xticklabels(rotation=90,labels=Category.index)

plt.title('Content Rating vs No Apps')

plt.ylabel('Number of Apps')

plt.xlabel('Category')

plt.show()
Type=topRated['Type'].value_counts()
data = [go.Pie(

        labels = Type.index,

        values = Type.values,

        hoverinfo = 'label+value'

    

)]

plotly.offline.iplot(data, filename='active_category')
Installs=topApp['Installs'].value_counts()

print(Installs)

plt.figure(figsize=(12,12))

g=sns.barplot(Installs.index, Installs.values, alpha=0.8)

plt.title('Content Rating vs No Apps')

plt.ylabel('Number of Apps')

plt.xlabel('Category')

plt.show()