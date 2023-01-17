import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import datetime as dt
from sklearn.feature_extraction.text import TfidfVectorizer
from yellowbrick.text import TSNEVisualizer
from yellowbrick.datasets import load_hobbies
from yellowbrick.text import FreqDistVisualizer
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
nltk.download('all')
data=pd.read_csv('../input/reddit-india-flair-detection/datafinal.csv')
data.shape
data.columns
data.drop('id',axis=1,inplace=True)
data.drop('Unnamed: 0', axis=1, inplace= True )
data.shape
data.head()
data.info()
data['flair'].value_counts()
# converting to data frame
df=pd.DataFrame(data)
y=data.columns
x=data.count()
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
features = y 
values = x
ax.bar(y,x)
plt.xticks(rotation=90)
plt.savefig('value.png', dpi=300, bbox_inches='tight')
plt.show()
temp = df['title'].str.len()
temp.hist(bins = np.arange(0,200,1))
plt.title("no of words in TITLE")
plt.xlabel("length of Title (in words)", fontsize=12)
plt.ylabel("Number of posts", fontsize=12)
plt.savefig('title.png', dpi=300, bbox_inches='tight')
plt.show()

temp = df['body'].str.len()
temp.hist(bins = np.arange(0,1000,10))
plt.title("no of words in BODY")
plt.xlabel("length of BODY (in words)")
plt.ylabel("Number of posts")
plt.savefig('lenth of words in body.png', dpi=300, bbox_inches='tight')
plt.show()
fig,ax = plt.subplots()
ax.grid()
df.groupby('flair').plot(x='comms_num', y='score', ax=ax, legend=False)
plt.savefig('comments and upvotes corresponding to the flairs', dpi=300, bbox_inches='tight')
plt.show()
df2 = df.groupby("flair").mean()[['score']]

df2.plot(kind='bar', legend=False, grid=True)
plt.title("Average score per flair")

plt.xlabel("Flair")
plt.ylabel("Average score per post")
plt.savefig('score per flair', dpi=300, bbox_inches='tight')
plt.show()

df3 = df.groupby("flair").mean()[['comms_num']]

df3.plot(kind='bar', legend=False, grid=True)
plt.title("Average no of coments per flair")

plt.xlabel("Flair")
plt.ylabel("Average no of comments per post")
plt.savefig('comms per flair', dpi=300, bbox_inches='tight')
plt.show()

df4=df.set_index('timestamp')
sns.set(rc={'figure.figsize':(20, 8)})
df4['comms_num'].plot(linewidth=1);
plt.title("Number of comments per post during the timestamp")
plt.xlabel("Timestamp")
plt.ylabel("Number of comments")
plt.savefig('no of comments per time stamp.png')
plt.show()
sns.set(rc={'figure.figsize':(20, 8)})
df4['score'].plot(linewidth=1);
plt.title("score per post during the timestamp")
plt.xlabel("Timestamp")
plt.ylabel("score")
plt.savefig('score per post during time stamp.png')
plt.show()
df1=df.set_index('timestamp')
df2= df1.groupby('flair')[['comms_num']]
df2.head()
sns.set(rc={'figure.figsize':(20, 8)})
df1.groupby('flair')[['comms_num']].plot(linewidth=1);
plt.title("Number of comments per post during the timestamp corresponding to the flairs")
plt.xlabel("Timestamp")
plt.ylabel("Number of comments")
plt.savefig('no of comments per time stamp corresponding to the flairs.png')
plt.show()
z=df['flair'].unique()
df['body'].fillna(" ",inplace=True)


tfidf = TfidfVectorizer()
x = tfidf.fit_transform(df['body'])
y = df['flair']
tsne = TSNEVisualizer(labels=z)
tsne.fit(x, y)
tsne.poof()
df['comments'].fillna(" ",inplace=True)

tfidf = TfidfVectorizer()
x = tfidf.fit_transform(df['comments'])
y = df['flair']
tsne = TSNEVisualizer(labels=z)
tsne.fit(x, y)
tsne.poof()

tfidf = TfidfVectorizer()
x = tfidf.fit_transform(df['title'])
y = df['flair']
tsne = TSNEVisualizer(labels=z)
tsne.fit(x, y)
tsne.poof()

df['combined_features'].fillna(" ",inplace=True)

tfidf = TfidfVectorizer()
x = tfidf.fit_transform(df['combined_features'])
y = df['flair']
tsne = TSNEVisualizer(labels=z)
tsne.fit(x, y)
tsne.poof()
stopwords = set(stopwords.words('english'))
vectorizer = CountVectorizer(stop_words=stopwords)
docs = vectorizer.fit_transform(df['body'])
features = vectorizer.get_feature_names()
visualizer = FreqDistVisualizer(features=features)
visualizer.fit(docs)
visualizer.poof()
vectorizer = CountVectorizer(stop_words=stopwords)
docs = vectorizer.fit_transform(df['title'])
features = vectorizer.get_feature_names()
visualizer = FreqDistVisualizer(features=features)
visualizer.fit(docs)
visualizer.poof()
vectorizer = CountVectorizer(stop_words=stopwords)
docs = vectorizer.fit_transform(df['comments'])
features = vectorizer.get_feature_names()
visualizer = FreqDistVisualizer(features=features)
visualizer.fit(docs)
visualizer.poof()
vectorizer = CountVectorizer(stop_words=stopwords)
docs = vectorizer.fit_transform(df['combined_features'])
features = vectorizer.get_feature_names()
visualizer = FreqDistVisualizer(features=features)
visualizer.fit(docs)
visualizer.poof()
