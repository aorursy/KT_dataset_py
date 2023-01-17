import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
df_info=pd.read_csv('../input/google-play-store-apps/googleplaystore.csv')

df_info.head()
df_info.sample(5)
df_info.tail()
df_info.columns
print('App:')

print(df_info['App'].describe())

print()

print('Category:')

print(df_info['Category'].describe())

print()

print('Rating:')

print(df_info['Rating'].describe())

print()

print('Reviews:')

print(df_info['Reviews'].describe())

print()

print('Size:')

print(df_info['Size'].describe())

print()

print('Installs:')

print(df_info['Installs'].describe())

print()

print('Type:')

print(df_info['Type'].describe())

print()

print('Price:')

print(df_info['Price'].describe())

print()

print('Content Rating:')

print(df_info['Content Rating'].describe())

print()

print('Genres:')

print(df_info['Genres'].describe())

print()

print('Last Updated:')

print(df_info['Last Updated'].describe())

print()

print('Current Ver:')

print(df_info['Current Ver'].describe())

print()

print('Android Ver:')

print(df_info['Android Ver'].describe())

print()
df_info.loc[df_info.App=='Tiny Scanner - PDF Scanner App']

df_info[df_info.duplicated(keep='first')]

print(len(df_info))

df_info.drop_duplicates(subset='App', inplace=True)

if(df_info.App=='Life Made WI-Fi Touchscreen Photo Frame').any():

    df_info.drop(10472,inplace=True)

if(df_info.App=='Command & Conquer: Rivals').any():

    df_info.drop(9148,inplace=True)

print(len(df_info))
#App

df_info.App = df_info.App.apply(lambda x: str(x))

#Category

df_info.Category = df_info.Category.apply(lambda x: str(x))

#Rating

df_info.Rating = df_info.Rating.apply(lambda x: float(x))

print('NaN Ratings:')

print(len(df_info.loc[pd.isna(df_info.Rating)]))

#Reviews

df_info.Reviews = df_info.Reviews.apply(lambda x: int(x))

#Size : Remove 'M', Convert 'k'

df_info.Size = df_info.Size.apply(lambda x: str(x))

print('Apps having Varies with device as size:')

print(len(df_info.loc[df_info.Size=='Varies with device']))

df_info.Size = df_info.Size.apply(lambda x: str(x).replace('Varies with device', 'NaN') if 'Varies with device' in str(x) else x)

df_info.Size = df_info.Size.apply(lambda x: str(x).replace('M', '') if 'M' in str(x) else x)

df_info.Size = df_info.Size.apply(lambda x: str(x).replace(',', '') if 'M' in str(x) else x)

df_info.Size = df_info.Size.apply(lambda x: float(str(x).replace('k', '')) / 1000 if 'k' in str(x) else x)

df_info.Size = df_info.Size.apply(lambda x: round(float(x),2))

#Installs: Remove + and ,

df_info.Installs = df_info.Installs.apply(lambda x: x.replace('+', '') if '+' in str(x) else x)

df_info.Installs = df_info.Installs.apply(lambda x: x.replace(',', '') if ',' in str(x) else x)

df_info.Installs = df_info.Installs.apply(lambda x: int(x))

#Type

df_info.Type = df_info.Type.apply(lambda x: str(x))

#Price

df_info.Price = df_info.Price.apply(lambda x: x.replace('$', '') if '$' in str(x) else x)

df_info.Price = df_info.Price.apply(lambda x: int(round(float(x))))

#Content Rating

df_info['Content Rating'] = df_info['Content Rating'].apply(lambda x: str(x))

df_info['Content Rating'] = df_info['Content Rating'].apply(lambda x: x.replace('Everyone 10+', '10+') if 'Everyone 10+' in str(x) else x)

df_info['Content Rating'] = df_info['Content Rating'].apply(lambda x: x.replace('Teen', '13+') if 'Teen' in str(x) else x)

df_info['Content Rating'] = df_info['Content Rating'].apply(lambda x: x.replace('Mature 17+', '17+') if 'Mature 17+' in str(x) else x)

df_info['Content Rating'] = df_info['Content Rating'].apply(lambda x: x.replace('Adults only 18+', '18+') if 'Adults only 18+' in str(x) else x)

df_info.Genres.astype('str')

pd.to_datetime(df_info['Last Updated'])

print('Data shape:')

print(df_info.shape)

df_info.sample(5)

df_info.describe(include=[np.object]).round(1).transpose()
df_reviews=pd.read_csv('../input/google-play-store-apps/googleplaystore_user_reviews.csv')

df_reviews.head()
df_reviews.sample(5)
df_reviews.tail()
print('App:')

print(df_reviews['App'].describe())

print()

print('Translated_Review:')

print(df_reviews['Translated_Review'].describe())

print()

print('Sentiment:')

print(df_reviews['Sentiment'].describe())

print()

print('Sentiment_Polarity:')

print(df_reviews['Sentiment_Polarity'].describe())

print()

print('Sentiment_Subjectivity:')

print(df_reviews['Sentiment_Subjectivity'].describe())

print()
df_reviews.columns
print(df_reviews.shape)

print('NaN Translated_Review:')

print(len(df_reviews.loc[pd.isna(df_reviews.Translated_Review)]))
df_reviews=df_reviews.dropna()

print(df_reviews.shape)
df_reviews.sample(5)
round(df_reviews.describe(),0)
df_reviews.describe(include=[np.object]).round(1)
import math

import scipy.stats as stats

import plotly

import plotly.graph_objs as go

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

plotly.offline.init_notebook_mode(connected=True)

%matplotlib inline

import matplotlib.style

import matplotlib as mpl

mpl.style.use('default')
df=df_info['Category'].value_counts()

df
df_info['Category'].value_counts().plot(kind='barh',figsize=(4,8))

plt.title('No of apps in each category')

plt.xlabel('No of apps')

plt.ylabel('Category')

plt.show()
df.describe()
df_info['Category'].describe()
df_info['Rating'].hist()

plt.xlabel('Rating')

plt.show()
df_info['Rating'].describe()
mean=df_info.Rating.mean()

variance=df_info.Rating.var()

stdDev = math.sqrt(variance)

x = np.linspace(mean - stdDev, mean + stdDev, 100)

plt.plot(x, stats.norm.pdf(x, mean, stdDev))

plt.show()
df_info['Reviews'].hist()
df=df_info[df_info['Reviews']>=10000000]

df['Reviews'].hist()
df_info['Size'].describe()
df_info.Size.hist()
df_info.Size.value_counts(bins=[0,20,40,60,80,100])
df_info.Installs.hist()
df_info.Installs.value_counts(bins=[0,100000000,500000000,1000000000])
df_info.Type.value_counts()
df=df_info[df_info['Price']>0]

df.Price.describe()
df.Price.hist()
df_info['Content Rating'].value_counts()
df_info.Genres.value_counts()
df_info['Android Ver'].value_counts()
np.warnings.filterwarnings('ignore')

sns.set()

sns.pairplot(df_info,hue='Type')

plt.show()
number_of_apps_in_category = df_info['Category'].value_counts().sort_values(ascending=True)



data = [go.Pie(

        labels = number_of_apps_in_category.index,

        values = number_of_apps_in_category.values,

        hoverinfo = 'label+value'    

)]

plotly.offline.iplot(data, filename='active_category')
groups = df_info.groupby('Category').filter(lambda x: len(x) > 292).reset_index()

array = groups['Rating'].hist(by=groups['Category'], sharex=True, figsize=(20,20))
df=df_info.groupby(['Category']).filter(lambda x: len(x) > 292).reset_index()

df=df.groupby(['Category']).mean()

df
groups = df_info.groupby('Category').filter(lambda x: len(x) >= 292).reset_index()

print('Average rating = ', np.nanmean(list(groups.Rating)))

c = ['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 720, len(set(groups.Category)))]



layout = {'title' : 'App ratings across major categories',

        'xaxis': {'tickangle':-40},

        'yaxis': {'title': 'Rating'},

          'plot_bgcolor': 'rgb(250,250,250)',

          'shapes': [{

              'type' :'line',

              'x0': -.5,

              'y0': np.nanmean(list(groups.Rating)),

              'x1': 19,

              'y1': np.nanmean(list(groups.Rating)),

              'line': { 'dash': 'dashdot'}

          }]

          }



data = [{

    'y': df_info.loc[df_info.Category==category]['Rating'], 

    'type':'violin',

    'name' : category,

    'showlegend':False,

    #'marker': {'color': 'Set2'},

    } for i,category in enumerate(list(set(groups.Category)))]



plotly.offline.iplot({'data': data, 'layout': layout})
df['Rating'].sort_values(ascending=True).plot(kind='barh',figsize=(4,4))

plt.title('Average Rating in each category')

plt.xlabel('Average Rating')

plt.ylabel('Category')

plt.show()
df['Price'].sort_values(ascending=True).plot(kind='barh',figsize=(5,5))

plt.title('Average Price in each category')

plt.xlabel('Average Price')

plt.ylabel('Category')

plt.show()
df['Size'].sort_values(ascending=True).plot(kind='barh',figsize=(5,5))

plt.title('Average Size in each category')

plt.xlabel('Average Size')

plt.ylabel('Category')

plt.show()
df['Installs'].sort_values(ascending=True).plot(kind='barh',figsize=(5,5))

plt.title('Average Installs in each category')

plt.xlabel('Average Installs')

plt.ylabel('Category')

plt.show()
df['Reviews'].sort_values(ascending=True).plot(kind='barh',figsize=(5,5))

plt.title('Average Reviews in each category')

plt.xlabel('Average Reviews')

plt.ylabel('Category')

plt.show()
fig, ax = plt.subplots(figsize=(5,8))

s=sns.scatterplot(x="Rating", y="Category", hue="Type", data=df_info, ax=ax);

plt.title('Rating in each category')

plt.draw()

plt.show()
fig, ax = plt.subplots(figsize=(4,8))

s=sns.scatterplot(x="Installs", y="Category", hue="Type", data=df_info, ax=ax);

plt.title('No of Installs in each category')

plt.xlabel('No of Installs')

plt.draw()

plt.show()
fig, ax = plt.subplots(figsize=(5,5))

s=sns.scatterplot(x="Price", y="Rating", hue="Type", data=df_info, ax=ax)

plt.title('Price vs Rating')

plt.draw()

plt.show()
fig, ax = plt.subplots(figsize=(5,5))

s=sns.scatterplot(x="Size", y="Rating", hue="Type", data=df_info);

plt.title('Size vs Rating')

plt.draw()

plt.show()
fig, ax = plt.subplots(figsize=(5,3))

s=sns.scatterplot(x="Rating", y="Reviews", hue="Type", data=df_info, ax=ax);

plt.title('Reviews vs Rating')

plt.draw()

plt.show()
df_info_rating=df_info[df_info.Rating<=2.5]

number_of_apps_in_category = df_info_rating['Category'].value_counts().sort_values(ascending=True)



data = [go.Pie(

        labels = number_of_apps_in_category.index,

        values = number_of_apps_in_category.values,

        hoverinfo = 'label+value'    

)]



plotly.offline.iplot(data, filename='active_category')
fig, ax = plt.subplots(figsize=(5,8))

s=sns.scatterplot(x="Reviews", y="Category", hue="Type", data=df_info, ax=ax);

plt.draw()

plt.show()
fig, ax = plt.subplots(figsize=(5,8))

s=sns.scatterplot(x="Installs", y="Category", hue="Type", data=df_info, ax=ax);

plt.draw()

plt.show()
fig, ax = plt.subplots(figsize=(5,3))

s=sns.scatterplot(x="Price", y="Installs", hue="Type", data=df_info, ax=ax);

plt.title('Price vs Installs')

plt.draw()

plt.show()
df_info['Size']=df_info['Size'].astype(float)

fig, ax = plt.subplots(figsize=(5,3))

s=sns.scatterplot(x="Size", y="Installs", hue="Type", data=df_info, ax=ax);

plt.title('Size vs Installs')

plt.draw()

plt.show()
df_info['Size']=df_info['Size'].astype(float)

fig, ax = plt.subplots(figsize=(5,3))

s=sns.scatterplot(x="Rating", y="Installs", hue="Type", data=df_info, ax=ax);

plt.title('Installs vs Ratings')

plt.draw()

plt.show()
fig, ax = plt.subplots(figsize=(5,3))

s=sns.scatterplot(x="Reviews", y="Installs", hue="Type", data=df_info, ax=ax);

plt.title('Reviews vs Installs')

plt.draw()

plt.show()
df_reviews['Sentiment'].value_counts()
df_reviews['Sentiment'].value_counts().plot(kind='bar',figsize=(3,3))

plt.title('Sentiment Count')

plt.xlabel('Sentiment')

plt.ylabel('Count')
df_reviews['Sentiment_Polarity'].hist()

plt.title('Sentiment_Polarity Count')

plt.xlabel('Sentiment_Polarity')

plt.ylabel('Count')

plt.show()
df_reviews['Sentiment_Subjectivity'].hist()

plt.title('Sentiment_Subjectivity Count')

plt.xlabel('Sentiment_Subjectivity')

plt.ylabel('Count')

plt.show()

plt.show()
text = " ".join(review for review in df_reviews.Translated_Review)

print ("There are {} words in the combination of all review.".format(len(text)))
wordcloud = WordCloud(max_words=50, background_color="white").generate(text)

plt.figure(figsize=(5,5))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
df=df_reviews[df_reviews['Sentiment']=='Positive']

textP = " ".join(review for review in df.Translated_Review)

print ("There are {} words in the combination of all review.".format(len(textP)))
wordcloud = WordCloud(max_words=50, background_color="white").generate(textP)

plt.figure(figsize=[5,5])

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
df=df_reviews[df_reviews['Sentiment']=='Neutral']

textU = " ".join(review for review in df.Translated_Review)

print ("There are {} words in the combination of all review.".format(len(textU)))
wordcloud = WordCloud(max_words=50, background_color="white").generate(textU)

plt.figure(figsize=[5,5])

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
df=df_reviews[df_reviews['Sentiment']=='Negative']

textN = " ".join(review for review in df.Translated_Review)

print ("There are {} words in the combination of all review.".format(len(textN)))
wordcloud = WordCloud(max_words=50, background_color="white").generate(textN)

plt.figure(figsize=[5,5])

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
fig, ax = plt.subplots(figsize=(15,7))

s=sns.scatterplot(x="Sentiment_Subjectivity", y="Sentiment_Polarity", hue="Sentiment", data=df_reviews, ax=ax);

plt.draw()

s.set_xticklabels(ax.get_xticklabels(),rotation=90)

ax.ticklabel_format(style='plain', axis='y')

plt.show()