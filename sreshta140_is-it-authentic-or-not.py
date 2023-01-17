# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import plotly.offline as pyoff
import plotly.graph_objs as go
import plotly.graph_objs as go
fake = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/Fake.csv')
real = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/True.csv')
fake['Target']=1
real['Target']=0
frames = [fake, real]

df = pd.concat(frames)
df
patternDel = "http"
filter1 = df['date'].str.contains(patternDel)
df = df[~filter1]
pattern = "Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec"
filter2 = df['date'].str.contains(pattern)
df=df[filter2]
df['date'] = pd.to_datetime(df['date'])
df_sub=df.groupby(['subject', 'Target'])['text'].count()
df_sub
df_sub = df_sub.unstack().fillna(0)
df_sub
# Visualize this data in bar plot
ax = (df_sub).plot(
kind='bar',
figsize=(10, 7),
grid=True
)
ax.set_ylabel('Count')
plt.show()
df_sub['Count']=df_sub[0]+df_sub[1]
import plotly.graph_objects as go

labels = df_sub.index
values = df_sub['Count']

fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
fig.show()
df_=df.copy()
df_=df_.sort_values(by=['date'])
df_=df_.reset_index(drop=True)
df_
df_1=df_[df_['Target']==1]
df_1=df_1.groupby(['date'])['Target'].count()
df_1=pd.DataFrame(df_1)
df_1['Target']
df_0=df_[df_['Target']==0]
df_0=df_0.groupby(['date'])['Target'].count()
df_0=pd.DataFrame(df_0)
plot_data = [
    go.Scatter(
        x=df_0.index,
        y=df_0['Target'],
        name='True',
        #x_axis="OTI",
        #y_axis="time",
    ),
    go.Scatter(
        x=df_1.index,
        y=df_1['Target'],
        name='Fake'
    )
    
]
plot_layout = go.Layout(
        title='Day-wise',
        yaxis_title='Count',
        xaxis_title='Time',
        plot_bgcolor='rgba(0,0,0,0)'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)
from wordcloud import WordCloud, STOPWORDS 
comment_words = '' 
stopwords = set(STOPWORDS) 
  
# iterate through the csv file 
for val in df[df['Target']==1]['text']: 
      
    # typecaste each val to string 
    val = str(val) 
  

    tokens = val.split() 
      
    # Converts each token into lowercase 
    for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower() 
      
    comment_words += " ".join(tokens)+" "
  
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='black', 
                stopwords = stopwords, 
                min_font_size = 10).generate(comment_words) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show()
comment_words = '' 
stopwords = set(STOPWORDS) 
  
# iterate through the csv file 
for val in df[df['Target']==0]['text']: 
      
    # typecaste each val to string 
    val = str(val) 
  

    tokens = val.split() 
      
    # Converts each token into lowercase 
    for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower() 
      
    comment_words += " ".join(tokens)+" "
  
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='black', 
                stopwords = stopwords, 
                min_font_size = 10).generate(comment_words) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show()
df_['news']=df_['subject']+' '+df_['title']+' '+df_['text']
df_['news'] = df_.apply(lambda x: x['news'].lower(),axis=1)
df_["news"] = df_['news'].str.replace('[^\w\s]','')
all_news=pd.DataFrame(pd.Series(' '.join(df_['news']).split()).value_counts())
allnews1=all_news.head(30)
plot_data = [
    go.Bar(
        x=allnews1.index,
        y=allnews1[0],
        name='True',
        #x_axis="OTI",
        #y_axis="time",
        marker = dict(
            color = allnews1[0]
        )
    )
]
plot_layout = go.Layout(
        title='Top 30 words',
        yaxis_title='Count',
        xaxis_title='Word',
        plot_bgcolor='rgba(0,0,0,0)'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)
fake_news=pd.DataFrame(pd.Series(' '.join(df_[df_['Target']==1]['news']).split()).value_counts())
fake_news30=fake_news.head(30)
plot_data = [
    go.Bar(
        x=fake_news30.index,
        y=fake_news30[0],
        #name='True',
        #x_axis="OTI",
        #y_axis="time",
        marker = dict(
            color = fake_news30[0]
        )
    )
]
plot_layout = go.Layout(
        title='Top 30 words from Fake news',
        yaxis_title='Count',
        xaxis_title='Word',
        plot_bgcolor='rgba(0,0,0,0)'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)
real_news=pd.DataFrame(pd.Series(' '.join(df_[df_['Target']==0]['news']).split()).value_counts())
real_news30=real_news.head(30)
plot_data = [
    go.Bar(
        x=real_news30.index,
        y=real_news30[0],
        #name='True',
        #x_axis="OTI",
        #y_axis="time",
        marker = dict(
            color = real_news30[0]
        )
    )
]
plot_layout = go.Layout(
        title='Top 30 words from Real news',
        yaxis_title='Count',
        xaxis_title='Word',
        plot_bgcolor='rgba(0,0,0,0)'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)
import nltk, re, string, collections
from nltk.util import ngrams
import re
import unicodedata
import nltk
from nltk.corpus import stopwords
def basic_clean(text):
  """
  A simple function to clean up the data. All the words that
  are not designated as a stop word is then lemmatized after
  encoding and basic regex parsing are performed.
  """
  wnl = nltk.stem.WordNetLemmatizer()
  stopwords = nltk.corpus.stopwords.words('english')
  text = (unicodedata.normalize('NFKD', text)
    .encode('ascii', 'ignore')
    .decode('utf-8', 'ignore')
    .lower())
  words = re.sub(r'[^\w\s]', '', text).split()
  return [wnl.lemmatize(word) for word in words if word not in stopwords]
words = basic_clean(''.join(str(df_['news'].tolist())))
bigram_all=(pd.Series(nltk.ngrams(words, 2)).value_counts())[:30]
bigram_all=pd.DataFrame(bigram_all)
bigram_all
bg_a=bigram_all.copy()
bg_a['in']=bg_a.index
bg_a['in'] = bg_a.apply(lambda x: '('+x['in'][0]+', '+x['in'][1]+')',axis=1)
plot_data = [
    go.Bar(
        x=bg_a['in'],
        y=bg_a[0],
        #name='True',
        #x_axis="OTI",
        #y_axis="time",
        marker = dict(
            color = 'blue'
        )
    )
]
plot_layout = go.Layout(
        title='Top 30 bigrams from News',
        yaxis_title='Count',
        xaxis_title='Word',
        plot_bgcolor='rgba(0,0,0,0)'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)
trigram_all=(pd.Series(nltk.ngrams(words, 3)).value_counts())[:30]
trigram_all=pd.DataFrame(trigram_all)
trigram_all
trigram_all['in']=trigram_all.index
trigram_all['in'] = trigram_all.apply(lambda x: '('+x['in'][0]+', '+x['in'][1]+', '+x['in'][2]+')',axis=1)
trigram_all
plot_data = [
    go.Bar(
        x=trigram_all['in'],
        y=trigram_all[0],
        #name='True',
        #x_axis="OTI",
        #y_axis="time",
        marker = dict(
            color = 'blue'
        )
    )
]
plot_layout = go.Layout(
        title='Top 30 trigrams from News',
        yaxis_title='Count',
        xaxis_title='tri-gram',
        plot_bgcolor='rgba(0,0,0,0)'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)
fake_words = basic_clean(''.join(str(df_[df_['Target']==1]['news'].tolist())))
bigram_fake=(pd.Series(nltk.ngrams(fake_words, 2)).value_counts())[:30]
bigram_fake=pd.DataFrame(bigram_fake)
bigram_fake
bigram_fake['in']=bigram_fake.index
bigram_fake['in'] = bigram_fake.apply(lambda x: '('+x['in'][0]+', '+x['in'][1]+')',axis=1)
plot_data = [
    go.Bar(
        x=bigram_fake['in'],
        y=bigram_fake[0],
        #name='True',
        #x_axis="OTI",
        #y_axis="time",
        marker = dict(
            color = 'Red'
        )
    )
]
plot_layout = go.Layout(
        title='Top 30 bi-grams from Fake News',
        yaxis_title='Count',
        xaxis_title='bi-gram',
        plot_bgcolor='rgba(0,0,0,0)'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)
trigram_fake=(pd.Series(nltk.ngrams(fake_words, 3)).value_counts())[:30]
trigram_fake=pd.DataFrame(trigram_fake)
trigram_fake
trigram_fake['in']=trigram_fake.index
trigram_fake['in'] = trigram_fake.apply(lambda x: '('+x['in'][0]+', '+x['in'][1]+', '+x['in'][2]+')',axis=1)
plot_data = [
    go.Bar(
        x=trigram_fake['in'],
        y=trigram_fake[0],
        #name='True',
        #x_axis="OTI",
        #y_axis="time",
        marker = dict(
            color = 'Red'
        )
    )
]
plot_layout = go.Layout(
        title='Top 30 tri-grams from Fake News',
        yaxis_title='Count',
        xaxis_title='tri-gram',
        plot_bgcolor='rgba(0,0,0,0)'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)
true_words = basic_clean(''.join(str(df_[df_['Target']==0]['news'].tolist())))
bigram_true=(pd.Series(nltk.ngrams(true_words, 2)).value_counts())[:30]
bigram_true=pd.DataFrame(bigram_true)
bigram_true
bigram_true['in']=bigram_true.index
bigram_true['in'] = bigram_true.apply(lambda x: '('+x['in'][0]+', '+x['in'][1]+')',axis=1)
plot_data = [
    go.Bar(
        x=bigram_true['in'],
        y=bigram_true[0],
        #name='True',
        #x_axis="OTI",
        #y_axis="time",
        marker = dict(
            color = 'Green'
        )
    )
]
plot_layout = go.Layout(
        title='Top 30 bi-grams from True News',
        yaxis_title='Count',
        xaxis_title='bi-gram',
        plot_bgcolor='rgba(0,0,0,0)'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)
trigram_true=(pd.Series(nltk.ngrams(true_words, 3)).value_counts())[:30]
trigram_true=pd.DataFrame(trigram_true)
trigram_true
trigram_true['in']=trigram_true.index
trigram_true['in'] = trigram_true.apply(lambda x: '('+x['in'][0]+', '+x['in'][1]+', '+x['in'][2]+')',axis=1)
plot_data = [
    go.Bar(
        x=trigram_true['in'],
        y=trigram_true[0],
        #name='True',
        #x_axis="OTI",
        #y_axis="time",
        marker = dict(
            color = 'Green'
        )
    )
]
plot_layout = go.Layout(
        title='Top 30 tri-grams from True News',
        yaxis_title='Count',
        xaxis_title='tri-gram',
        plot_bgcolor='rgba(0,0,0,0)'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)
y = df_['Target'].values
#Converting X to format acceptable by gensim, removing annd punctuation stopwords in the process
X = []
stop_words = set(nltk.corpus.stopwords.words("english"))
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
for par in df_["news"].values:
    tmp = []
    sentences = nltk.sent_tokenize(par)
    for sent in sentences:
        sent = sent.lower()
        tokens = tokenizer.tokenize(sent)
        filtered_words = [w.strip() for w in tokens if w not in stop_words and len(w) > 1]
        tmp.extend(filtered_words)
    X.append(tmp)

#del data
import gensim
#Dimension of vectors we are generating
EMBEDDING_DIM = 100

#Creating Word Vectors by Word2Vec Method (takes time...)
w2v_model = gensim.models.Word2Vec(sentences=X, size=EMBEDDING_DIM, window=5, min_count=1)
len(w2v_model.wv.vocab)
w2v_model["trump"]
w2v_model.wv.most_similar("trump")
w2v_model.wv.most_similar("obama")
w2v_model.wv.most_similar("news")
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(df_['news'], 
                                                    df_['Target'], 
                                                    random_state=0)
from sklearn.feature_extraction.text import CountVectorizer

# Fit the CountVectorizer to the training data
vect = CountVectorizer().fit(X_train)
X_train_vectorized = vect.transform(X_train)

X_train_vectorized
from sklearn.linear_model import LogisticRegression

# Train the model
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)
from sklearn.metrics import roc_auc_score

# Predict the transformed test documents
predictions = model.predict(vect.transform(X_test))

print('AUC: ', roc_auc_score(y_test, predictions))
from sklearn import metrics
print(metrics.confusion_matrix(y_test, predictions, labels=[0, 1]))
# Printing the precision and recall, among other metrics
print(metrics.classification_report(y_test, predictions, labels=[0, 1]))
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predictions , labels=[0, 1])
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax= ax)
ax.set_ylabel('y_test')
ax.set_xlabel('predictions')
# get the feature names as numpy array
feature_names = np.array(vect.get_feature_names())

# Sort the coefficients from the model
sorted_coef_index = model.coef_[0].argsort()

# Find the 10 smallest and 10 largest coefficients
# The 10 largest coefficients are being indexed using [:-11:-1] 
# so the list returned is in order of largest to smallest
print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))
from sklearn.naive_bayes import MultinomialNB 
vectorizer = CountVectorizer() 
X_train_transformed = vectorizer.fit_transform(X_train) 
X_test_transformed = vectorizer.transform(X_test)

clf = MultinomialNB(alpha=0.1) 
clf.fit(X_train_transformed, y_train)

y_predicted = clf.predict(X_test_transformed)
print('AUC: ', roc_auc_score(y_test, y_predicted))
from sklearn import metrics
print(metrics.confusion_matrix(y_test, y_predicted, labels=[0, 1]))
# Printing the precision and recall, among other metrics
print(metrics.classification_report(y_test, y_predicted, labels=[0, 1]))
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax= ax)
ax.set_ylabel('y_test')
ax.set_xlabel('predictions')
from sklearn.feature_extraction.text import TfidfVectorizer

# Fit the TfidfVectorizer to the training data specifiying a minimum document frequency of 5
vect = TfidfVectorizer(min_df=5).fit(X_train)
len(vect.get_feature_names())
X_train_vectorized = vect.transform(X_train)

model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

predictions = model.predict(vect.transform(X_test))

print('AUC: ', roc_auc_score(y_test, predictions))
from sklearn import metrics
print(metrics.confusion_matrix(y_test, predictions, labels=[0, 1]))
# Printing the precision and recall, among other metrics
print(metrics.classification_report(y_test, predictions, labels=[0, 1]))
cm = confusion_matrix(y_test, predictions , labels=[0, 1])
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax= ax)
ax.set_ylabel('y_test')
ax.set_xlabel('predictions')
feature_names = np.array(vect.get_feature_names())

sorted_tfidf_index = X_train_vectorized.max(0).toarray()[0].argsort()

print('Smallest tfidf:\n{}\n'.format(feature_names[sorted_tfidf_index[:10]]))
print('Largest tfidf: \n{}'.format(feature_names[sorted_tfidf_index[:-11:-1]]))

sorted_coef_index = model.coef_[0].argsort()

print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))
vectorizer = TfidfVectorizer(min_df=3)
X_train_transformed = vectorizer.fit_transform(X_train)
X_test_transformed = vectorizer.transform(X_test)

clf = MultinomialNB(alpha=0.1)
clf.fit(X_train_transformed, y_train)

# y_predicted_prob = clf.predict_proba(X_test_transformed)[:, 1]
y_predicted = clf.predict(X_test_transformed)
print('AUC: ', roc_auc_score(y_test, y_predicted))
print(metrics.confusion_matrix(y_test, y_predicted, labels=[0, 1]))
# Printing the precision and recall, among other metrics
print(metrics.classification_report(y_test, y_predicted, labels=[0, 1]))
cm = confusion_matrix(y_test, y_predicted , labels=[0, 1])
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax= ax)
ax.set_ylabel('y_test')
ax.set_xlabel('predictions')
#ngrams
# Fit the CountVectorizer to the training data specifiying a minimum 
# document frequency of 5 and extracting 1-grams and 2-grams
vect = CountVectorizer(min_df=5, ngram_range=(1,2)).fit(X_train)

X_train_vectorized = vect.transform(X_train)

len(vect.get_feature_names())

model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

predictions = model.predict(vect.transform(X_test))

print('AUC: ', roc_auc_score(y_test, predictions))
print(metrics.confusion_matrix(y_test, predictions, labels=[0, 1]))
# Printing the precision and recall, among other metrics
print(metrics.classification_report(y_test, predictions, labels=[0, 1]))
cm = confusion_matrix(y_test, predictions , labels=[0, 1])
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax= ax)
ax.set_ylabel('y_test')
ax.set_xlabel('predictions')
feature_names = np.array(vect.get_feature_names())

sorted_coef_index = model.coef_[0].argsort()

print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))