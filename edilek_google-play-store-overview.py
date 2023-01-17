# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from collections import Counter

import plotly as py # visualization library

from plotly.offline import init_notebook_mode, iplot # plotly offline mode

init_notebook_mode(connected=True) 

import plotly.graph_objs as go 





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Read and Clean Data

data=pd.read_csv("../input/google-play-store-apps/googleplaystore.csv")

data_review=pd.read_csv("../input/google-play-store-apps/googleplaystore_user_reviews.csv")

data.drop_duplicates(subset='App', inplace=True)

data = data[data['Android Ver'] != np.nan]

data = data[data['Android Ver'] != 'NaN']

data = data[data['Installs'] != 'Free']

data = data[data['Installs'] != 'Paid']

data.columns=['App', 'Category', 'Rating', 'Reviews', 'Size', 'Installs', 'Type',

       'Price', 'Content_Rating', 'Genres', 'Last_Updated', 'Current_Ver',

       'Android_Ver']
data.info()
data_review.info()
data['Installs'] = data['Installs'].apply(lambda x: x.replace('+', '') if '+' in str(x) else x)

data['Installs'] = data['Installs'].apply(lambda x: x.replace(',', '') if ',' in str(x) else x)

data['Installs'] = data['Installs'].apply(lambda x: int(x))



data['Size'] = data['Size'].apply(lambda x: str(x).replace('Varies with device', 'NaN') if 'Varies with device' in str(x) else x)



data['Size'] = data['Size'].apply(lambda x: str(x).replace('M', '') if 'M' in str(x) else x)

data['Size'] = data['Size'].apply(lambda x: str(x).replace(',', '') if 'M' in str(x) else x)

data['Size'] = data['Size'].apply(lambda x: float(str(x).replace('k', '')) / 1000 if 'k' in str(x) else x)





data['Size'] = data['Size'].apply(lambda x: float(x))

data['Installs'] = data['Installs'].apply(lambda x: float(x))



data['Price'] = data['Price'].apply(lambda x: str(x).replace('$', '') if '$' in str(x) else str(x))

data['Price'] = data['Price'].apply(lambda x: float(x))



data['Reviews'] = data['Reviews'].apply(lambda x: int(x))





x = data['Rating'].dropna()

y = data['Size'].dropna()

z = data['Installs'][data.Installs!=0].dropna()

p = data['Reviews'][data.Reviews!=0].dropna()

t = data['Type'].dropna()

price = data['Price']

Category=data["Category"]

APP=data["App"]

Content=data["Content_Rating"]

data.info()
f,ax = plt.subplots(figsize=(5, 5))

sns.heatmap(data.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.2f',ax=ax)

plt.show()
log=pd.DataFrame(list(zip(APP,Category,x, y, np.log(z), np.log10(p), t, price)))

log.columns=["App","Category","Rating","size","Installs","Reviews","Type", "Price"]

log

#Installs VS Reviews of app

log.Installs.plot(color = 'r',label = 'Installs',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

log.Reviews.plot(color = 'b',label = 'Reviews',linewidth=1, alpha = 0.5,grid = True,linestyle = "dotted")

plt.legend(loc='upper right')

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.title('Line Plot')

plt.show()
log.plot(kind='scatter', x='size', y='Rating',alpha = 0.5,color = 'red')

plt.xlabel('Size')              

plt.ylabel('Rating')

plt.title('How do Sizes impact the app rating?')          

plt.show()

log.Rating.plot(kind = 'hist',label="Rating", bins = 5,figsize = (8,8))

plt.title("Rating-Histogram")

plt.show()
data['Category'].unique()

area_list=list(log['Category'].unique())

area_poverty_ratio = []

for i in area_list:

    x = log[log['Category']==i]

    area_poverty_rate = sum(x.Rating)/len(x)

    area_poverty_ratio.append(area_poverty_rate)

data1 = pd.DataFrame({'area_list': area_list,'area_poverty_ratio':area_poverty_ratio})

new_index = (data1['area_poverty_ratio'].sort_values(ascending=False)).index.values

sorted_data = data1.reindex(new_index)

# visualization

plt.figure(figsize=(15,10))

sns.barplot(x=sorted_data['area_list'], y=sorted_data['area_poverty_ratio'])

plt.xticks(rotation= 90)

plt.xlabel('APP Categories')

plt.ylabel('Rating')

plt.title('Customer Rating of App Categories')

plt.show()
log.head(5)
log.info()
area_list = list(log['Category'].unique())

Rating = []

size = []

Installs = []

Reviews = []

Price = []

for i in area_list:

    x = log[log['Category']==i]

    Rating.append(sum(x.Rating)/len(x))

#     size.append(sum(x.size) / len(x))

    Installs.append(sum(x.Installs) / len(x))

    Reviews.append(sum(x.Reviews) / len(x))

    Price.append(sum(x.Price) / len(x))

# visualization

f,ax = plt.subplots(figsize = (9,15))

sns.barplot(x=Rating,y=area_list,color='green',alpha = 0.5,label='Rating' )

sns.barplot(x=Installs,y=area_list,color='cyan',alpha = 0.6,label='Installs')

sns.barplot(x=Reviews,y=area_list,color='yellow',alpha = 0.6,label='Review')

sns.barplot(x=Price,y=area_list,color='red',alpha = 0.6,label='Price')



ax.legend(loc='lower right',frameon = True)     # legendlarin gorunurlugu

ax.set(xlabel='Percentage of Races', ylabel='States',title = "Percentage of Categorical situtation According to numerical values in Data ")


log.head(10).boxplot(column='Installs',by = 'Rating')

plt.show()
melted = pd.melt(frame=log,id_vars = 'App', value_vars= ['Rating','Installs'])

pivot=melted.pivot(index = 'App', columns = 'variable',values='value')

threshold = pivot.Rating.mean()

pivot["Rating_level"] = ["High" if i > threshold else "Low" for i in pivot.Rating]

pivot

sns.countplot(pivot.Rating_level)

plt.title("Rating Level",color = 'blue',fontsize=15)

plt.show()
log.head()
plt.figure(figsize=(20,15))

new_df=pd.DataFrame([log.groupby('Category')['Reviews'].mean().index,log.groupby('Category')['Reviews'].mean().values])

new_df = new_df.transpose()

new_df.columns=["Category", "Reviews_means"]

sns.barplot(x=new_df.sort_values(by=['Reviews_means'], ascending=False).Category,y=new_df.sort_values(by=['Reviews_means'], ascending=False).Reviews_means)

plt.title("Reviews Means of Each Category")

plt.xlabel("Category")

plt.ylabel("Reviews_means")

plt.xticks(rotation=90)

plt.show()
data_review.info()

data_review_log=data_review.copy()

data_review_log=data_review_log[data_review_log["Sentiment_Polarity"]>0]

data_review_log.info()

# data_review_log["App"]=pd.DataFrame(data_review["App"])
data_review_log=data_review_log.sort_values(by=['Sentiment_Polarity'], ascending=False)

data_review_log1=pd.DataFrame()

data_review_log1["Sentiment_Polarity"]=data_review_log.groupby('App')['Sentiment_Polarity'].mean().values

data_review_log1["App"]=data_review_log.groupby('App')['Sentiment_Polarity'].mean().index

data_review_log1["Sentiment_Subjectivity"]=data_review_log.groupby('App')['Sentiment_Subjectivity'].mean().values



# # visualize

f,ax1 = plt.subplots(figsize =(20,10))

sns.pointplot(x="App",y="Sentiment_Polarity",data=data_review_log1.loc[:50],color='lime',alpha=0.8)

sns.pointplot(x='App',y='Sentiment_Subjectivity',data=data_review_log1.loc[:50],color='red',alpha=0.8)

plt.text(4,0.2,'Sentiment_Polarity',color='red',fontsize = 17,style = 'italic')

plt.text(4,0.1,'Sentiment_Subjectivity',color='lime',fontsize = 18,style = 'italic')

plt.xlabel('Applications',fontsize = 15,color='blue')

plt.ylabel('Values',fontsize = 15,color='blue')

plt.xticks(rotation=90)

plt.title('Sentiment_polarity  rate vs sentiment_subjectivity rate of each App',fontsize = 20,color='blue')

plt.grid()
log.head()
a=log.sort_values(by=['Price'], ascending=False)

a=a[a["Type"]=="Paid"]

g = sns.jointplot(a.Price, a.Installs, kind="scatter", size=7)

plt.savefig('graph.png')

plt.show()
sns.swarmplot(x="Installs", y="Type",hue="Price", data=log.loc[:1000])

plt.show()
Translated_Review_df=pd.DataFrame(data_review_log.Translated_Review.value_counts())

Translated_Review_df=Translated_Review_df[Translated_Review_df["Translated_Review"]>70]

Translated_Review_df

labels = Translated_Review_df.index

colors = ['grey','blue','red','yellow','green']

explode = [0,0,0,0,0]

sizes = Translated_Review_df.Translated_Review

plt.figure(figsize = (7,7))

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')

plt.title('Translated Reviews of All Applications ',color = 'blue',fontsize = 15)

plt.show()
pivot.head()
# Installs	Rating	Rating_level

# prepare data frame

df = pivot.iloc[:100,:]



# import graph objects as "go"

import plotly.graph_objs as go



# Creating trace1

trace1 = go.Scatter(

                    x = df.Rating_level,

                    y = df.Installs,

                    mode = "lines",

                    name = "Installs",

                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),

                    text= df.index)

# Creating trace2

trace2 = go.Scatter(

                    x = df.Rating_level,

                    y = df.Rating,

                    mode = "lines+markers",

                    name = "Reviews",

                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),

                    text= df.index)

data = [trace1, trace2]

layout = dict(title = 'Installs and Reviews vs Rating of Top 100 Applications',

              xaxis= dict(title= 'RATING',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
pivot.head()
# prepare data frames

dfhigh = pivot[pivot.Rating_level == "High"].iloc[:100,:]

dflow = pivot[pivot.Rating_level == "Low"].iloc[:100,:]

# import graph objects as "go"

import plotly.graph_objs as go

# creating trace1

trace1 =go.Scatter(

                    x = dfhigh.Installs,

                    y = dfhigh.Rating,

                    mode = "markers",

                    name = "High",

                    marker = dict(color = 'rgba(255, 128, 255, 0.8)'),

                    text= dfhigh.index)

# creating trace2

trace2 =go.Scatter(

                    x = dflow.Installs,

                    y = dflow.Rating,

                    mode = "markers",

                    name = "Low",

                    marker = dict(color = 'rgba(255, 128, 2, 0.8)'),

                    text= dflow.index)

# # creating trace3

# trace3 =go.Scatter(

#                     x = df2016.world_rank,

#                     y = df2016.citations,

#                     mode = "markers",

#                     name = "2016",

#                     marker = dict(color = 'rgba(0, 255, 200, 0.8)'),

#                     text= df2016.university_name)

data = [trace1, trace2]

layout = dict(title = 'Instals vs Rating of top 100 App with their Rating_level',

              xaxis= dict(title= 'Installs',ticklen= 5,zeroline= False),

              yaxis= dict(title= 'Rating',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)