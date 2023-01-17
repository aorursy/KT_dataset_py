import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')

import os

import re

import functools



pd.options.display.float_format = "{:.2f}".format



# Standard plotly imports

#import plotly_express as px

import plotly.plotly as py

import plotly.graph_objs as go

from plotly.offline import iplot, init_notebook_mode

# Using plotly + cufflinks in offline mode

import cufflinks

cufflinks.go_offline(connected=True)

init_notebook_mode(connected=True)



os.listdir()
kaggle=1

if kaggle==0:

    store=pd.read_csv('googleplaystore.csv')

    review=pd.read_csv('googleplaystore_user_reviews.csv')

else:

    store=pd.read_csv('../input/googleplaystore.csv')

    review=pd.read_csv('../input/googleplaystore_user_reviews.csv')
print(f'Shape of the google playstore data:{store.shape}')

print(f'Shape of the user reviews data:{review.shape}')
review.info()
store.info()
store.loc[store['App']=="Life Made WI-Fi Touchscreen Photo Frame"]
store.loc[store['App']=="Life Made WI-Fi Touchscreen Photo Frame",'Category']='PHOTOGRAPHY'

store.loc[store['App']=="Life Made WI-Fi Touchscreen Photo Frame",'Rating']='1.9'

store.loc[store['App']=="Life Made WI-Fi Touchscreen Photo Frame",'Reviews']='19'

store.loc[store['App']=="Life Made WI-Fi Touchscreen Photo Frame",'Size']='3.0M'

store.loc[store['App']=="Life Made WI-Fi Touchscreen Photo Frame",'Installs']='1,000+'

store.loc[store['App']=="Life Made WI-Fi Touchscreen Photo Frame",'Type']='Free'

store.loc[store['App']=="Life Made WI-Fi Touchscreen Photo Frame",'Price']='0'

store.loc[store['App']=="Life Made WI-Fi Touchscreen Photo Frame",'Content Rating']='Everyone'

store.loc[store['App']=="Life Made WI-Fi Touchscreen Photo Frame",'Genres']='Photography'

store.loc[store['App']=="Life Made WI-Fi Touchscreen Photo Frame",'Last Updated']='February 11, 2018'

store.loc[store['App']=="Life Made WI-Fi Touchscreen Photo Frame",'Current Ver']='1.0.19'

store.loc[store['App']=="Life Made WI-Fi Touchscreen Photo Frame",'Android Ver']='4.0 and up'
store.loc[store['App']=="Life Made WI-Fi Touchscreen Photo Frame"]
store['Size']=store['Size'].apply(lambda x:str(x).replace("M","") if 'M' in str(x) else x)

store['Size']=store['Size'].apply(lambda x:float(str(x).replace("k",""))/1000 if 'k' in str(x) else x)

store['Price']=store['Price'].apply(lambda x:str(x).replace("$","") if "$" in str(x) else x)

store['Installs']=store['Installs'].apply(lambda x:str(x).replace("+","") if "+" in str(x) else x)

store['Installs']=store['Installs'].apply(lambda x:str(x).replace(",","") if "," in str(x) else x)
store=store.astype({'Rating':'float32','Reviews':'float32','Installs':'float64','Price':'float64'})
store.isnull().sum()
store.loc[store['Type'].isna()]
store.loc[store['Type'].isna(),'Price']=1.14

store.loc[store['Type'].isna(),'Type']='Paid'
store.loc[store['Rating'].isna()]
store['Rating'].fillna(0,inplace=True)
store.loc[store['Current Ver'].isna()]
store['Current Ver'].fillna('Varies with device',inplace=True)
store.loc[store['Android Ver'].isna()]
store['Android Ver'].fillna('Varies with device',inplace=True)
store.head()
review.head()
store['Rating'].describe()
plt.figure(figsize=(8,8))

ax=sns.distplot(store['Rating'],bins=40,color="green")

ax.set_xlabel("Rating")

ax.set_ylabel("Distribution Frequency")

ax.set_title("Rating - Distribution")
store['Reviews'].describe()
plt.figure(figsize=(8,8))

ax=sns.distplot(store['Reviews'],color="green")

ax.set_xlabel("Review")

ax.set_ylabel("Distribution Frequency")

ax.set_title("Review - Distribution")
plt.figure(figsize=(8,8))

ax=sns.distplot(np.log1p(store['Reviews']),color="green")

ax.set_xlabel("Review")

ax.set_ylabel("Log - Distribution Frequency")

ax.set_title("Review -Log  Distribution")
store.loc[store['Size']!="Varies with device",['Size']].astype('float32').describe()
plt.figure(figsize=(8,8))

ax=sns.distplot(store.loc[store['Size']!="Varies with device",['Size']].astype('float32'),color="green")

ax.set_xlabel("Size")

ax.set_ylabel("Distribution Frequency")

ax.set_title("Size -Distribution")
store['Installs'].describe()
plt.figure(figsize=(8,8))

ax=sns.distplot(store['Installs'],color="green")

ax.set_xlabel("Installs")

ax.set_ylabel("Distribution Frequency")

ax.get_yaxis().get_major_formatter().set_scientific(False)

ax.set_title("Installs -Distribution")
store['App'].nunique()
pd.concat(g for _,g in store.groupby('App') if len(g)>1)
pd.concat(g for _,g in store.groupby('App') if len(g)>1)['App'].nunique()
pd.concat(g for _,g in store.groupby('App') if len(g)>1)['App'].shape
## Removing the duplicates:

store = store.drop_duplicates()
## Check the number of rows,

store.shape
## Get the rows having the maximum review count ,

#temp=store.loc[store.groupby(['App'])['Reviews'].idxmax()]

store=store.loc[store.groupby(['App','Category'])['Reviews'].idxmax()]
store.shape
## Check the apps having assigned in more than 1 category 

## Taken from my kernel on Olympics - https://www.kaggle.com/gsdeepakkumar/gold-hunters

multi_cat=store.groupby('App').apply(lambda x:x['Category'].unique()).to_frame().reset_index()

multi_cat.columns=['App','Categories']

multi_cat['Count']=[len(c) for c in multi_cat['Categories']]
multi_cat[multi_cat['Count']>1].sort_values('Count',ascending=False)
store['Category'].nunique()
store['Category'].unique()
category_app=store.groupby('Category')['App'].nunique().sort_values(ascending=False).to_frame().reset_index()

category_app.columns=['Category','Total']

category_app['Perc']=category_app['Total']/sum(category_app['Total'])

category_app
store.groupby('Category')['Rating'].mean().sort_values(ascending=False)

avg_rating=store.groupby(['Category','Type'])['Rating'].mean().sort_values(ascending=False).to_frame().reset_index()

plt.figure(figsize=(12,7))

plt.subplot(211)

ax=sns.barplot(x='Category',y='Rating',data=avg_rating.loc[avg_rating['Type']=='Free'],color="blue")

ax.set_xticklabels(ax.get_xticklabels(),rotation=90)

ax.set_xlabel('Category')

ax.set_ylabel('Rating')

ax.set_title("Category and Average Rating for Free Apps")

plt.subplot(212)

ax=sns.boxplot(x='Category',y='Rating',data=store.loc[store['Type']=='Free'],order=avg_rating.loc[avg_rating['Type']=='Free','Category'])

ax.set_xticklabels(ax.get_xticklabels(),rotation=90)

ax.set_xlabel('Category')

ax.set_ylabel('Rating')

ax.set_title("Boxplot of Category and Rating for Free Apps")



plt.subplots_adjust(wspace = 0.8, hspace = 1.2,top = 1.3)



plt.show()

plt.figure(figsize=(12,7))

plt.subplot(211)

ax=sns.barplot(x='Category',y='Rating',data=avg_rating.loc[avg_rating['Type']=='Paid'],color="lightblue")

ax.set_xticklabels(ax.get_xticklabels(),rotation=90)

ax.set_xlabel('Category')

ax.set_ylabel('Rating')

ax.set_title("Category and Average Rating for Paid Apps")



plt.subplot(212)

ax=sns.boxplot(x='Category',y='Rating',data=store.loc[store['Type']=='Paid'],order=avg_rating.loc[avg_rating['Type']=='Paid','Category'])

ax.set_xticklabels(ax.get_xticklabels(),rotation=90)

ax.set_xlabel('Category')

ax.set_ylabel('Rating')

ax.set_title("Boxplot of Category and Rating for Paid Apps")



plt.subplots_adjust(wspace = 0.8, hspace = 1.2,top = 1.3)



plt.show()


store.groupby('Category')['Installs'].mean().sort_values(ascending=False)

avg_installs=store.groupby(['Category','Type'])['Installs'].mean().sort_values(ascending=False).to_frame().reset_index()

plt.figure(figsize=(12,7))

plt.subplot(211)

ax=sns.barplot(x='Category',y='Installs',data=avg_installs.loc[avg_installs['Type']=='Free'],color="blue")

ax.set_xticklabels(ax.get_xticklabels(),rotation=90)

ax.set_xlabel('Category')

ax.set_ylabel('Installs')

ax.get_yaxis().get_major_formatter().set_scientific(False)

ax.set_title("Category and Average Installs for Free Apps")



plt.subplot(212)

ax=sns.boxplot(x='Category',y='Installs',data=store.loc[store['Type']=='Free'],order=avg_installs.loc[avg_installs['Type']=='Free','Category'])

ax.set_xticklabels(ax.get_xticklabels(),rotation=90)

ax.set_xlabel('Category')

ax.set_ylabel('Installs')

ax.get_yaxis().get_major_formatter().set_scientific(False)

ax.set_title("Boxplot of Category and Installs for Free Apps")



plt.subplots_adjust(wspace = 0.8, hspace = 1.2,top = 1.3)



plt.show()



plt.figure(figsize=(12,7))

plt.subplot(211)

ax=sns.barplot(x='Category',y='Installs',data=avg_installs.loc[avg_installs['Type']=='Paid'],color="blue")

ax.set_xticklabels(ax.get_xticklabels(),rotation=90)

ax.set_xlabel('Category')

ax.set_ylabel('Installs')

ax.get_yaxis().get_major_formatter().set_scientific(False)

ax.set_title("Category and Average Installs for Paid Apps")



plt.subplot(212)

ax=sns.boxplot(x='Category',y='Installs',data=store.loc[store['Type']=='Paid'],order=avg_installs.loc[avg_installs['Type']=='Paid','Category'])

ax.set_xticklabels(ax.get_xticklabels(),rotation=90)

ax.set_xlabel('Category')

ax.set_ylabel('Installs')

ax.get_yaxis().get_major_formatter().set_scientific(False)

ax.set_title("Boxplot of Category and Installs for Paid Apps")



plt.subplots_adjust(wspace = 0.8, hspace = 1.2,top = 1.3)



plt.show()

family_app=store[store['Installs']>=4654605].loc[store['Category']=='FAMILY'].sort_values(by='Rating',ascending=False)[0:9]

family_app[['App','Rating','Installs','Type']]
game_app=store[store['Installs']>=14550962].loc[store['Category']=='GAME'].sort_values(by='Rating',ascending=False)[0:9]

game_app[['App','Rating','Installs','Type']]
tool_app=store[store['Installs']>=9774151].loc[store['Category']=='TOOLS'].sort_values(by='Rating',ascending=False)[0:9]

tool_app[['App','Rating','Installs','Type']]
business_app=store[store['Installs']>=1659916].loc[store['Category']=='BUSINESS'].sort_values(by='Rating',ascending=False)[0:9]

business_app[['App','Rating','Installs','Type']]
medical_app=store[store['Installs']>=99224].loc[store['Category']=='MEDICAL'].sort_values(by='Rating',ascending=False)[0:9]

medical_app[['App','Rating','Installs','Type']]
store.loc[store['Size']!='Varies with device'].iplot(

    x='Rating',

    y='Size',

    # Specify the category

    categories='Type',

    xTitle='Rating',

    yTitle='Size',

    title='Rating Vs Size by Type')
store.iplot(

    x='Rating',

    y='Reviews',

    # Specify the category

    categories='Type',

    xTitle='Rating',

    yTitle='Reviews',

    title='Rating Vs Reviews by Type')
# px.scatter(store, x="Installs", y="Rating", color="Rating", facet_col="Type",

#            color_continuous_scale=px.colors.sequential.Viridis, render_mode="webgl")
content_rating=store.groupby('Content Rating')['App'].nunique().sort_values(ascending=False).to_frame().reset_index()

content_rating.columns=['Content Rating','Apps']

content_rating['Perc']=content_rating['Apps']/sum(content_rating['Apps'])

content_rating
plt.figure(figsize=(8,8))

ax=sns.distplot(store.loc[store['Type']=='Paid',['Price']],color='green')

ax.set_xlabel("Price of app")

ax.set_ylabel("Freq Distribution")

ax.set_title("Distribution of Price")
exp_apps=store.loc[store['Type']=='Paid'].sort_values(by='Price',ascending=False)[0:10]

exp_apps[['App','Price']]
plt.figure(figsize=(8,8))

ax=sns.barplot(x='App',y='Price',data=exp_apps)

ax.set_xlabel('App')

ax.set_ylabel('Price')

ax.set_xticklabels(ax.get_xticklabels(),rotation=90)

ax.set_title("Most Expensive Apps")
review.head()
review=review.dropna()
review.head()
review['Sentiment'].value_counts()
pos=review.loc[review['Sentiment']=='Positive']

neg=review.loc[review['Sentiment']=='Negative']

neu=review.loc[review['Sentiment']=='Neutral']
from wordcloud import WordCloud, STOPWORDS

import string

from nltk.stem import WordNetLemmatizer

from sklearn.decomposition import LatentDirichletAllocation

from sklearn.feature_extraction.text import CountVectorizer

import nltk

from nltk.corpus import stopwords

from nltk.stem.porter import *

from nltk.tokenize import word_tokenize, sent_tokenize

from nltk import word_tokenize

eng_stopwords=set(stopwords.words('english'))
### Inspired from https://www.kaggle.com/arthurtok/spooky-nlp-and-topic-modelling-tutorial

plt.figure(figsize=(16,13))

wc = WordCloud(background_color="white", max_words=10000, 

            stopwords=STOPWORDS)

wc.generate(" ".join(pos['Translated_Review'].values))

plt.title("Wordcloud for Positive Reviews", fontsize=20)

plt.imshow(wc.recolor( colormap= 'viridis' , random_state=17), alpha=0.98)

plt.axis('off')
plt.figure(figsize=(16,13))

wc = WordCloud(background_color="white", max_words=10000, 

            stopwords=STOPWORDS)

wc.generate(" ".join(neg['Translated_Review'].values))

plt.title("Wordcloud for Negative Reviews", fontsize=20)

plt.imshow(wc.recolor( colormap= 'viridis' , random_state=17), alpha=0.98)

plt.axis('off')
plt.figure(figsize=(16,13))

wc = WordCloud(background_color="white", max_words=10000, 

            stopwords=STOPWORDS)

wc.generate(" ".join(neu['Translated_Review'].values))

plt.title("Wordcloud for Neutral Reviews", fontsize=20)

plt.imshow(wc.recolor( colormap= 'viridis' , random_state=17), alpha=0.98)

plt.axis('off')
review['num_words']=review['Translated_Review'].apply(lambda x:len(str(x).split()))

review['num_stopwords']=review['Translated_Review'].apply(lambda x:len([w for w in str(x).lower().split() if w in eng_stopwords]))

review['num_punctuations']=review['Translated_Review'].apply(lambda x:len([w for w in str(x) if w in string.punctuation]))
## For better visuals truncate words greater than 120 to 120

review['num_words'].loc[review['num_words']>120]=120



plt.figure(figsize=(8,8))

ax=sns.boxplot(x='Sentiment',y='num_words',data=review)

ax.set_xlabel('Sentiment')

ax.set_xticklabels(ax.get_xticklabels(),rotation=90)

ax.set_ylabel('Number of Words')

ax.set_title("Difference between the Number of Words Vs Sentiment")


plt.figure(figsize=(8,8))

ax=sns.boxplot(x='Sentiment',y='num_stopwords',data=review)

ax.set_xlabel('Sentiment')

ax.set_xticklabels(ax.get_xticklabels(),rotation=90)

ax.set_ylabel('Number of Stop Words')

ax.set_title("Difference between the Number of Stop Words Vs Sentiment")
plt.figure(figsize=(8,8))

ax=sns.boxplot(x='Sentiment',y='num_punctuations',data=review)

ax.set_xlabel('Sentiment')

ax.set_xticklabels(ax.get_xticklabels(),rotation=90)

ax.set_ylabel('Number of Punctuations')

ax.set_title("Difference between the Number of Punctuations Vs Sentiment")
### Inspired from - https://www.kaggle.com/residentmario/exploring-elon-musk-tweets
tokens=review['Translated_Review'].map(word_tokenize)
tokens.head()
def get_reviews_on_token(x):

    x_l = x.lower()

    x_t = x.title()

    return review.loc[tokens.map(lambda sent: x_l in sent or x_t in sent).values]
get_reviews_on_token('Performance')[['App','Translated_Review']][0:5].values.tolist()

get_reviews_on_token('memory')[['App','Translated_Review']][0:5].values.tolist()
get_reviews_on_token('battery')[['App','Translated_Review']][0:5].values.tolist()
get_reviews_on_token('productivity')[['App','Translated_Review']][0:5].values.tolist()