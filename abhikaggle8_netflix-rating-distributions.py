import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
%matplotlib inline 
import nltk
from io import StringIO
import collections as co

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

df = pd.read_csv('../input/Netflix Shows.csv', encoding='cp437')
df.head()
print('The number of rows with Missing Values are: ')
df.isnull().any(axis=1).sum()
df = df.drop_duplicates(keep="first").reset_index(drop=True)
df.info()
print('In all, there are ',df['rating'].nunique(),'types of ratings in the dataset: ',df['rating'].unique())
#df_rating=df['rating'].unique()
print('In all, there are ',df['release year'].nunique(),'years in the dataset: ',df['release year'].unique())
#df_year=df['release year'].unique()
print("The Year-wise distribution of Netflix shows")
year_no_of_shows=df["release year"].value_counts().sort_values(ascending=False)
plt.figure(figsize=(12,4))
year_no_of_shows.plot(title='Years with the number of Netlfix Shows',kind="bar")
print('The number of Netflix Shows in the dataset are: ',df['title'].nunique())
plt.figure(figsize=(12,12))
df.rating.value_counts().plot(kind='pie',autopct='%1.1f%%')
plt.title('Number of appearances in dataset')
plt.show()
df.describe()
user_rating_score=df.groupby("user rating score")['title'].count().reset_index().sort_values(by='user rating score',ascending=False).reset_index(drop=True)
plt.figure(figsize=(12,6))
sns.barplot(x='user rating score',y='title', data=user_rating_score)
plt.xticks(rotation=45)

print('The Ratings along with their occurence in every year:')
df.groupby((['release year', 'rating'])).size()

plt.subplots(figsize=(12,10))
max_ratings=df.groupby('rating')['rating'].count()
max_ratings=max_ratings[max_ratings.values>50]
max_ratings.sort_values(ascending=True,inplace=True)
mean_shows=df[df['rating'].isin(max_ratings.index)]
piv=mean_shows.groupby(['release year','rating'])['user rating score'].mean().reset_index()
piv=piv.pivot('release year','rating','user rating score')
sns.heatmap(piv,annot=True,cmap='YlGnBu')
plt.title('Average User Score By Rating')
plt.show()
df1=df[df['ratingLevel'].notnull()]
string=StringIO()
df1['ratingLevel'].apply(lambda x: string.write(x))
x=string.getvalue()
string.close()
x=x.lower()
x=x.split()
words = co.Counter(nltk.corpus.words.words())
stopWords =co.Counter( nltk.corpus.stopwords.words() )
x=[i for i in x if i in words and i not in stopWords]
string=" ".join(x)
c = co.Counter(x)
most_common_10=c.most_common(10)
print('The 10 Most Common Words in ratingLevel are: ')
most_common_10
text = string
wordcloud2 = WordCloud(stopwords=STOPWORDS,
                          background_color='white',
                          width=8000,
                          height=5000,
                          relative_scaling = 1.0).generate(text)
plt.imshow(wordcloud2)
plt.axis("off")
plt.show()
wordcloud1 = WordCloud(
                          background_color='black',
                          width=8000,
                          height=5000,
                          relative_scaling = 1.0
                         ).generate(" ".join(df['title']))
plt.imshow(wordcloud1)
plt.axis('off')
plt.show()