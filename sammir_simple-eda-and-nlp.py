import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotly import tools
import plotly.offline as py
import plotly.graph_objs as go
from nltk.corpus import stopwords 
from nltk import word_tokenize, sent_tokenize, pos_tag, ne_chunk, FreqDist
from textblob import TextBlob
import collections
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import normalize
from wordcloud import WordCloud
%matplotlib inline
py.init_notebook_mode(connected=True)
train_data=pd.read_csv("../input/train.csv")
test_data=pd.read_csv("../input/test.csv")
submission_data=pd.read_csv("../input/sample_submission.csv")
train_data.shape
test_data.shape
submission_data.shape
train_data.head()
train_data.columns
train_data['question_text'].isnull().value_counts()
train_data['target'].isnull().value_counts()
train_data['question_text'].drop_duplicates().count()
# Unique target
train_data['target'].unique()
insicere_quiz=train_data[train_data['target']==1]
sincere_quiz=train_data[train_data['target']==0]
insicere_quiz['target'].count()/train_data['target'].count() * 100
sincere_quiz['target'].count()/train_data['target'].count() * 100
target_counts = train_data['target'].value_counts()
target_counts
pie_labels = (np.array(target_counts.index))
pie_sizes = (np.array((target_counts / target_counts.sum())*100))

trace = go.Pie(labels=pie_labels, values=pie_sizes)
pie_layout = go.Layout(title='Target distribution',font=dict(size=16),width=500,height=500)
fig = go.Figure(data=[trace], layout=pie_layout)
py.iplot(fig, filename="file_name")
bar_graph = go.Bar(
        x=target_counts.index,
        y=target_counts.values,
        marker=dict(
        color=target_counts.values,
        colorscale = 'Picnic',
        reversescale = True
    ),
)

bar_layout = go.Layout(title='Target Distrinution',font=dict(size=20))
fig = go.Figure(data=[bar_graph], layout=bar_layout)
py.iplot(fig, filename="file_name")
stop_words = stopwords.words('english')

tokens=[]
for i in train_data[0:10]['question_text']:
    for j in word_tokenize(i):
        tokens.append(j)

filtered_text = [token for token in tokens if not token in stop_words]  

filtered_text = [] 
  
for i in tokens: 
    if i in stop_words: 
        filtered_text.append(i) 
        
print(np.array(filtered_text))
len(np.array(filtered_text))
token=[]
for i in train_data[0:2]['question_text']:
    token.append(pos_tag(word_tokenize(i)))

print (token)
for i in train_data[0:10]['question_text']:
    print(TextBlob(i).noun_phrases)
# lng=[]
# for i in train_data[0:5]['question_text']:
#     lng.append(TextBlob(i).detect_language())
    
# set(lng)
for i in train_data[0:5]['question_text']:
    print(TextBlob(i).word_counts["quebec"])
for i in train_data[0:10]['question_text']:
    print(i," => ",len(word_tokenize(i)))
for i in train_data[0:10]['question_text']:
    print(i," => ",len(sent_tokenize(i)))
tokens=[]
for i in train_data[0:10]['question_text']:
    for j in word_tokenize(i):
        tokens.append(j)


frequency_distribution=FreqDist(tokens).most_common()
print(frequency_distribution)
tokens=[]
for i in train_data[0:10]['question_text']:
    for j in word_tokenize(i):
        tokens.append(j)


frequency_distribution=FreqDist(tokens)
word_c={}
for i in frequency_distribution:
    word_c[i]=token
    word_c[i]=frequency_distribution[i]

sorted(word_c.items(), key=lambda x: x[1], reverse=True)
text=str(train_data[0:1000]['question_text'])
wordcloud = WordCloud(width=1600, height=800).generate(text)
plt.figure( figsize=(20,10), facecolor='k')
plt.imshow(wordcloud)
plt.axis("off")
plt.figure( figsize=(40,30) )
plt.show()
# Calculating Sentment Analysis with TextBlob
for i in train_data[0:5]['question_text']:
    print(i," => ",TextBlob(i).sentiment)
# Extracting the sentiment polarity of a text
for i in train_data[0:5]['question_text']:
    print(TextBlob(i).sentiment.polarity)
# Extracting the sentiment subjectivity of a text
for i in train_data[0:5]['question_text']:
    print(TextBlob(i).sentiment.subjectivity)
for i in train_data[0:2]['question_text']:
    print(TextBlob(i).ngrams(n=3))
corpus=[]
for i in train_data[0:5]['question_text']:
    corpus.append(i)

cvect = CountVectorizer(ngram_range=(1,1))
counts = cvect.fit_transform(corpus)
normalized_counts = normalize(counts, norm='l1', axis=1)

tfidf = TfidfVectorizer(ngram_range=(1,1), smooth_idf=False)
tfs = tfidf.fit_transform(corpus)
new_tfs = normalized_counts.multiply(tfidf.idf_)

feature_names = tfidf.get_feature_names()
corpus_index = [n for n in corpus]
df = pd.DataFrame(new_tfs.T.todense(), index=feature_names, columns=corpus_index)

print(df)
#Bow with collection
token=[]
for i in train_data[0:5]['question_text']:
    token.append(i)

bow = [collections.Counter(words.split(" ")) for words in token]
total_bow=sum(bow,collections.Counter())
print(total_bow)