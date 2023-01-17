import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.dates as mdates
import seaborn as sns
sns.set()
sns.set_style("white")
import plotly
import cufflinks as cf
import plotly.graph_objs as go
import plotly.plotly as py
import plotly.tools as tls
from plotly import tools
from plotly.graph_objs import Scatter, Layout
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
from nltk import word_tokenize, regexp_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import sentiment_analyzer, SentimentAnalyzer, SentimentIntensityAnalyzer
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
file = "../input/winemag-data_first150k.csv"
df = pd.read_csv(file)
df.drop("Unnamed: 0", axis=1, inplace=True)
#Create Missing Values function to help determine what columns to keep and drop
def missing_data(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum().div(df.isnull().count()).sort_values(ascending=False))
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data

missing_data(df)
#Assess the unique values within each column to give the data more context and meaning
for i in df.columns:
    print ("There are: " + str(df[i].nunique()) + " unique {}".format(i) + " values")
#Remove the 5 NA values in the country column
df.dropna(subset=["country"], inplace=True)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(sharey=False, nrows=2, ncols=2, sharex=False, figsize=(20,15), squeeze=False)

df['country'].value_counts()[:25].plot(kind='bar', width=0.8, ax=ax1, grid=True, colormap='Dark2')
ax1.set_title("Review Total by Country", fontsize=18)
ax1.tick_params(axis='y', labelsize=18, rotation='auto')
ax1.tick_params(axis='x', labelsize=18)
ax1.set_ylabel('Review Count', fontsize=16)

df['province'].value_counts()[:25].plot(kind='bar', width=0.8, ax=ax2, grid=True, colormap='Dark2')
ax2.set_title("Review Total by Province", fontsize=18)
ax2.tick_params(axis='y', labelsize=18, rotation='auto')
ax2.tick_params(axis='x', labelsize=18)
ax2.set_ylabel('Review Count', fontsize=18)

df['winery'].value_counts()[:25].plot(kind='bar', width=0.8, ax=ax3, grid=True, colormap='Dark2')
ax3.set_title("Review Total by Winery", fontsize=18)
ax3.tick_params(axis='y', labelsize=18, rotation='auto')
ax3.tick_params(axis='x', labelsize=18)
ax3.set_ylabel('Review Count', fontsize=18)

df['variety'].value_counts()[:25].plot(kind='bar', width=0.8, ax=ax4, grid=True, colormap='Dark2')
ax4.set_title("Review Total by Variety", fontsize=18)
ax4.tick_params(axis='y', labelsize=18, rotation='auto')
ax4.tick_params(axis='x', labelsize=18)
ax4.set_ylabel('Review Count', fontsize=18)

plt.tight_layout()
plt.show()
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(sharey=False, nrows=2, ncols=2, sharex=False, figsize=(20,18))

df.groupby(["country"])['price'].agg('mean').sort_values(
    ascending=False)[:25].plot(kind='bar', width=0.8, ax=ax1, grid=True, colormap='Dark2')
ax1.set_title("Average Price by Country", fontsize=18)
ax1.tick_params(axis='y', labelsize=18, rotation='auto')
ax1.tick_params(axis='x', labelsize=18)
ax1.set_ylabel('Average Price', fontsize=18)

df.groupby(["province"])['price'].agg('mean').sort_values(
    ascending=False)[:25].plot(kind='bar', width=0.8, ax=ax2, grid=True, colormap='Dark2')
ax2.set_title("Average Price by Province", fontsize=18)
ax2.tick_params(axis='y', labelsize=18, rotation='auto')
ax2.tick_params(axis='x', labelsize=18)
ax2.set_ylabel('Average Price', fontsize=18)

df.groupby(["winery"])['price'].agg('mean').sort_values(
    ascending=False)[:25].plot(kind='bar', width=0.8, ax=ax3, grid=True, colormap='Dark2')
ax3.set_title("Average Price by Winery", fontsize=18)
ax3.tick_params(axis='y', labelsize=18, rotation='auto')
ax3.tick_params(axis='x', labelsize=18)
ax3.set_ylabel('Average Price', fontsize=18)

df.groupby(["variety"])['price'].agg('mean').sort_values(
    ascending=False)[:25].plot(kind='bar', width=0.8, ax=ax4, grid=True, colormap='Dark2')
ax4.set_title("Average Price by Variety", fontsize=18)
ax4.tick_params(axis='y', labelsize=18, rotation='auto')
ax4.tick_params(axis='x', labelsize=18)
ax4.set_ylabel('Average Price', fontsize=18)

plt.tight_layout()
plt.show()
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(sharey=False, nrows=2, ncols=2, sharex=False, figsize=(20,18))

df.groupby(["country"])['points'].agg('mean').sort_values(
    ascending=False)[:25].plot(kind='bar', width=0.8, ax=ax1, grid=True, colormap='Dark2')
ax1.set_title("Average Points by Country", fontsize=18)
ax1.tick_params(axis='y', labelsize=18, rotation='auto')
ax1.tick_params(axis='x', labelsize=18)
ax1.set_ylabel('Average Points', fontsize=18)
ax1.set_ylim(85,95)

df.groupby(["province"])['points'].agg('mean').sort_values(
    ascending=False)[:25].plot(kind='bar', width=0.8, ax=ax2, grid=True, colormap='Dark2')
ax2.set_title("Average Points by Province", fontsize=18)
ax2.tick_params(axis='y', labelsize=18, rotation='auto')
ax2.tick_params(axis='x', labelsize=18)
ax2.set_ylabel('Average Points', fontsize=18)
ax2.set_ylim(85,95)

df.groupby(["winery"])['points'].agg('mean').sort_values(
    ascending=False)[:25].plot(kind='bar', width=0.8, ax=ax3, grid=True, colormap='Dark2')
ax3.set_title("Average Points by Winery", fontsize=18)
ax3.tick_params(axis='y', labelsize=18, rotation='auto')
ax3.tick_params(axis='x', labelsize=18)
ax3.set_ylabel('Average Points', fontsize=18)
ax3.set_ylim(85,105)

df.groupby(["variety"])['points'].agg('mean').sort_values(
    ascending=False)[:25].plot(kind='bar', width=0.8, ax=ax4, grid=True, colormap='Dark2')
ax4.set_title("Average Points by Variety", fontsize=18)
ax4.tick_params(axis='y', labelsize=18, rotation='auto')
ax4.tick_params(axis='x', labelsize=18)
ax4.set_ylabel('Average Points', fontsize=18)
ax4.set_ylim(85,100)

plt.tight_layout()
plt.show()
ax1 = plt.subplot(121)
df["points"].value_counts().sort_index().plot(kind='bar', width=0.8, figsize=(25,8), grid=True, colormap='Dark2', ax=ax1)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.title("Counts by Point Intervals", fontsize=16)

ax2 = plt.subplot(122)
price_range = pd.cut(df["price"], range(0,2300, 50))
df.groupby([price_range])["price"].agg('count').plot(kind='bar', width=0.8, figsize=(25,8), grid=True, colormap='Dark2', ax=ax2)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.title("Counts by Price Intervals", fontsize=18)

plt.tight_layout()
plt.show()
print ("Over " + str(round(((df.groupby([price_range])["price"].count()/len(df))[1]), 2)) + "% of total wines reviewed are under $50")
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(sharey=False, nrows=2, ncols=2, sharex=False, figsize=(20,15))

df[df['points'] >= 98].groupby(["country"])['points'].agg('count').sort_values(ascending=False).plot(kind='bar', width=0.8, ax=ax1, grid=True, colormap='Dark2')
ax1.set_title("> 98 Points by Country", fontsize=18)
ax1.tick_params(axis='y', labelsize=18, rotation='auto')
ax1.tick_params(axis='x', labelsize=18)
ax1.set_ylabel('Count > 97 Points', fontsize=18)

df[df['points'] >= 98].groupby(["province"])['points'].agg('count').sort_values(ascending=False).plot(kind='bar', width=0.8, ax=ax2, grid=True, colormap='Dark2')
ax2.set_title("> 98 Points by Province", fontsize=18)
ax2.tick_params(axis='y', labelsize=18, rotation='auto')
ax2.tick_params(axis='x', labelsize=18)
ax2.set_ylabel('Count > 97 Points', fontsize=18)

df[df['points'] >= 98].groupby(["winery"])['points'].agg('count')[:25].sort_values(ascending=False).plot(kind='bar', width=0.8, ax=ax3, grid=True, colormap='Dark2')
ax3.set_title("> 98 Points by Winery", fontsize=18)
ax3.tick_params(axis='y', labelsize=18, rotation='auto')
ax3.tick_params(axis='x', labelsize=18)
ax3.set_ylabel('Count > 97 Points', fontsize=18)

df[df['points'] >= 98].groupby(["variety"])['points'].agg('count').sort_values(ascending=False).plot(kind='bar', width=0.8, ax=ax4, grid=True, colormap='Dark2')
ax4.set_title("> 98 Points by Variety", fontsize=18)
ax4.tick_params(axis='y', labelsize=18, rotation='auto')
ax4.tick_params(axis='x', labelsize=18)
ax4.set_ylabel('Count > 97 Points', fontsize=18)

plt.tight_layout()
plt.show()
df['Value_for_Money'] = df["points"].div(df['price'])
df.dropna(subset=['price'], inplace=True)
def value_for_money(column1, column2):
    new_df = df.sort_values(by='Value_for_Money', 
                   ascending=False)[['Value_for_Money', 
                                     column1, 
                                     column2]].head(20).reset_index().set_index([column1, column2])
    return new_df
ax1 = plt.subplot(121)
value_for_money('country', 'variety')["Value_for_Money"].plot(kind='bar', 
                                                              width=0.8, grid=True, colormap='Dark2', figsize=(20,9), ax=ax1)
plt.ylim(0,25)
plt.ylabel(s='Value for Money', fontsize=18)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.title("Value for Money by Country and Variety", fontsize=18)

ax2 = plt.subplot(122)
df[["price", 'points']].plot(kind='scatter', x='points', y='price', c='price', colormap='Dark2', s=125, figsize=(20,9), ax=ax2)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel(s='Points', fontsize=18)
plt.ylabel(s="Price", fontsize=18)
plt.xlim(79, 101, 1)
plt.title("Price Vs Points", fontsize=18)

plt.tight_layout()
plt.show()
#Count word frequency [split by space] - further cleaning required
result = Counter(" ".join(df["description"]).split(" ")).items()
#Transform dict into list
result = list(result)
#Create a pandas dataframe
word_df = pd.DataFrame(sorted(result, reverse=True), columns=["Word", "Count"])
#Some words use a different typography [IE: The actual character is not an "actual double quote"]. Additional cleaning
word_df['Word'] = word_df['Word'].replace({'(\u201c)':'""', '(\u201d)':'""', ',': "", '!':"", '"':""}, regex=True)

#Create an empty list
review_list = []

#Instantiate WordNetLem
wnl = WordNetLemmatizer()

#Remove all stop words (rows at this stage are single words)
for i in range(len(word_df['Word'])):
    tokens = [w for w in word_tokenize(str(word_df['Word'][i])) if w.isalpha()]
    no_stops = [t for t in tokens if t not in stopwords.words('english')]
    lemmatized = [wnl.lemmatize(t) for t in no_stops]
    review_list.append(lemmatized)
    
word_df.insert(loc=0, column="Revised_Word", value=review_list)
word_df["Revised_Word"] = word_df["Revised_Word"].str[0]
#Make all words upper case
word_df["Revised_Word"] = word_df["Revised_Word"].str.title()

word_df.drop('Word', axis=1, inplace=True)
word_df.dropna(inplace=True)
word_df.groupby(['Revised_Word'])['Count'].sum().sort_values(ascending=False)[:75].plot(kind='bar', 
                                                               grid=True, 
                                                               colormap='Dark2',
                                                               width=0.8, figsize=(25,9))

plt.legend("Word Frequency", fontsize=22)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.tight_layout()
plt.show()

#A few stop words not getting picked up. Will revise.

#for those who want to use Plotly (not working here for some reason):
# word_df.groupby(['Revised_Word'])['Count'].sum().sort_values(ascending=False)[:50].iplot(kind='bar', 
#                                                                showgrid=False, 
#                                                                layout={'width':'1000', 'height':'400'},
#                                                                colorscale='Dark2',
#                                                                width=0.9, title="Word Frequency")
stopwords_1 = set(STOPWORDS)

wordcloud = WordCloud(
                          background_color='black',
                          stopwords=stopwords_1,
                          max_words=150,
                          max_font_size=50, 
                          random_state=42, mode='RGBA', colormap='Blues'
                         ).generate(str(word_df["Revised_Word"]))

plt.figure(figsize=(20,7))
plt.imshow(wordcloud)

plt.grid(False)
plt.tight_layout()
plt.show()
