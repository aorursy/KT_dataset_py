!ls ../input/sarcasm/
# some necessary imports
import os
import string
import numpy as np
import pandas as pd
from sklearn import preprocessing, naive_bayes,linear_model, metrics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
color = sns.color_palette()

from matplotlib import pyplot as plt
from plotly import tools
import plotly.graph_objs as go
import plotly.offline as py
py.init_notebook_mode(connected=True)
train_df = pd.read_csv('../input/sarcasm/train-balanced-sarcasm.csv')
train_df.head()
train_df.info()
train_df.dropna(subset=['comment'], inplace=True)
train_df.info()
train_df['label'].value_counts()
train_texts, valid_texts, y_train, y_valid = \
        train_test_split(train_df['comment'], train_df['label'], random_state=17, train_size=.7)
#cm = plt.cm.get_cmap('RdYlBu_r')

#n, bins, patches = plt.hist(train_df['label'], density = True)
#    then normalize
#col = (n - n.min())/(n.max() - n.min())
#print(col)
#for c, p in zip(col, patches):
 #   plt.setp(p, 'facecolor', cm(c))
trgt_count = train_df['label'].value_counts()

labels = '0', '1'
sizes = np.array(trgt_count / trgt_count.sum() * 100)
#_, axes = plt.subplots(1, 2, sharey = True, figsize=(12, 8))    
sns.countplot(x ='label', data = train_df)

   ## target distribution ##
plt.pie(sizes, labels = labels)

#target count#
trgt_counts = train_df['label'].value_counts()
trace = go.Bar(
    x=trgt_counts.index, 
    y = trgt_counts.values,
    marker=dict(
        color=trgt_counts.values,
        colorscale='Picnic',
        reversescale=True
    ),
)
layout = go.Layout(
    title='Target Count',
    font=dict(size=18),
    width = 400, 
    height =500,
)
data=[trace]
fig=go.Figure(data=data,layout=layout)
py.iplot(fig,filename='TargetCount')

#target distribution#

labels = np.array(trgt_counts.index)
sizes = np.array(trgt_counts /trgt_counts.sum() * 100)

trace = go.Pie(labels=labels,values=sizes)
layout = go.Layout(
    title='Target Distribution',
    font=dict(size=18),
    width=600,
    height=600,
)
data=[trace]
fig=go.Figure(data=data,layout=layout)
py.iplot(fig,filename='shit')
from wordcloud import WordCloud, STOPWORDS

# Thanks : https://www.kaggle.com/aashita/word-clouds-of-various-shapes ##
def plot_wordcloud(text, mask=None, max_words=200, max_font_size=100, figure_size=(24.0,16.0), 
                   title = None, title_size=40, image_color=False):
    stopwords = set(STOPWORDS)
    more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}
    stopwords = stopwords.union(more_stopwords)

    wordcloud = WordCloud(background_color='black',
                    stopwords = stopwords,
                    max_words = max_words,
                    max_font_size = max_font_size, 
                    random_state = 42,
                    width=800, 
                    height=400,
                    mask = mask)
    wordcloud.generate(str(text))
    
    plt.figure(figsize=figure_size)
    if image_color:
        image_colors = ImageColorGenerator(mask);
        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");
        plt.title(title, fontdict={'size': title_size,  
                                  'verticalalignment': 'bottom'})
    else:
        plt.imshow(wordcloud);
        plt.title(title, fontdict={'size': title_size, 'color': 'black', 
                                  'verticalalignment': 'bottom'})
    plt.axis('off');
    plt.tight_layout()  
    
plot_wordcloud(train_texts, title='Word Cloud of Comments')
from collections import defaultdict
train1_df = train_texts[y_train ==1]
train0_df = train_texts[y_train ==0]

## let's generate some ngrams ##
def generate_ngrams(text, n_gram=1):
    token = [token for token in text.lower().split(' ') if token != '' if token not in STOPWORDS]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    return [' '.join(ngram) for ngram in ngrams]

# custom function for horizontal bar chart ##
def horizontal_bar_chart(df, color):
    trace = go.Bar(
        y = df['word'].values[::-1],
        x=df['wordcount'].values[::-1],
        showlegend=False,
        orientation='h',
        marker=dict(color=color),
    )
    return trace

#Get the bar chart from non-sarcasm comments ##
freq_dict =defaultdict(int)
for sent in train0_df:
    for word in generate_ngrams(sent):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(),key=lambda x:x[1])[::-1])
fd_sorted.columns=['word','wordcount']
trace0=horizontal_bar_chart(fd_sorted.head(50),'blue')

#Get the bar chart from sarcasm comments ##
freq_dict =defaultdict(int)
for sent in train1_df:
    for word in generate_ngrams(sent):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(),key=lambda x:x[1])[::-1])
fd_sorted.columns=['word','wordcount']
trace1=horizontal_bar_chart(fd_sorted.head(50),'red')

#create two subplots
fig = tools.make_subplots(rows=1,cols=2,vertical_spacing=0.04,
                         subplot_titles=['Frequent words of non-sarcasm comments',
                                        'Frequent words of sarcasm comments'])
fig.append_trace(trace0,1,1)
fig.append_trace(trace1,1,2)
fig['layout'].update(height=1200, width=900,paper_bgcolor='rgb(233,233,233)',title='word count sarcasm plots')
py.iplot(fig, filename='word_count_plots')
freq_dict =defaultdict(int)
for sent in train0_df:
    for word in generate_ngrams(sent, 2):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(),key=lambda x:x[1])[::-1])
fd_sorted.columns=['word','wordcount']
trace0=horizontal_bar_chart(fd_sorted.head(50),'yellow')

#Get the bar chart from sarcasm comments ##
freq_dict =defaultdict(int)
for sent in train1_df:
    for word in generate_ngrams(sent, 2):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(),key=lambda x:x[1])[::-1])
fd_sorted.columns=['word','wordcount']
trace1=horizontal_bar_chart(fd_sorted.head(50),'orange')

#create two subplots
fig = tools.make_subplots(rows=1,cols=2,vertical_spacing=0.04,
                         subplot_titles=['Frequent bigrams of non-sarcasm comments',
                                        'Frequent bigrams of sarcasm comments'])
fig.append_trace(trace0,1,1)
fig.append_trace(trace1,1,2)
fig['layout'].update(height=1200, width=900,paper_bgcolor='rgb(233,233,233)',title='Bigrams sarcasm plots')
py.iplot(fig, filename='word_count_plots')
freq_dict =defaultdict(int)
for sent in train0_df:
    for word in generate_ngrams(sent, 3):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(),key=lambda x:x[1])[::-1])
fd_sorted.columns=['word','wordcount']
trace0=horizontal_bar_chart(fd_sorted.head(50),'green')

#Get the bar chart from sarcasm comments ##
freq_dict =defaultdict(int)
for sent in train1_df:
    for word in generate_ngrams(sent, 3):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(),key=lambda x:x[1])[::-1])
fd_sorted.columns=['word','wordcount']
trace1=horizontal_bar_chart(fd_sorted.head(50),'blue')

#create two subplots
fig = tools.make_subplots(rows=1,cols=2,vertical_spacing=0.04,
                         subplot_titles=['Frequent trigrams of non-sarcasm comments',
                                        'Frequent trigrams of sarcasm comments'])
fig.append_trace(trace0,1,1)
fig.append_trace(trace1,1,2)
fig['layout'].update(height=1200, width=1500,paper_bgcolor='rgb(233,233,233)',title='Trigram sarcasm plots')
py.iplot(fig, filename='word_count_plots')
type(train_texts)
train_texts = train_texts.to_frame('comment')

valid_texts=valid_texts.to_frame('comment')
train_texts['label']= y_train
valid_texts['label']= y_valid
train_texts['num_words'] = train_texts['comment'].apply(lambda x: len(str(x).split()))
valid_texts['num_words'] = valid_texts['comment'].apply(lambda x: len(str(x).split()))

train_texts['num_unique_words']=train_texts['comment'].apply(lambda x: len(set(str(x).split())))
valid_texts['num_unique_words']=valid_texts['comment'].apply(lambda x: len(set(str(x).split())))
#train_df['num_unique_words']/train_df['num_words']

train_texts['num_chars'] = train_texts['comment'].apply(lambda x: len(str(x)))
valid_texts['num_chars'] = valid_texts['comment'].apply(lambda x: len(str(x)))

train_texts['num_stopwords'] = train_texts['comment'].apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))
valid_texts['num_stopwords'] = train_texts.apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))

train_texts['num_punctuations'] = train_texts['comment'].apply(lambda x: len([p for p in str(x) if p in string.punctuation]))
valid_texts['num_punctuations'] = valid_texts['comment'].apply(lambda x: len([p for p in str(x) if p in string.punctuation]))

train_texts['num_words_upper'] = train_texts['comment'].apply(lambda x: len([u for u in str(x) if u.isupper()]))
valid_texts['num_words_upper'] = valid_texts['comment'].apply(lambda x: len([u for u in str(x) if u.isupper()]))

train_texts['num_words_title']=train_texts['comment'].apply(lambda x: len([t for t in str(x) if t.istitle()]))
valid_texts['num_words_title']=valid_texts['comment'].apply(lambda x: len([t for t in str(x) if t.istitle()]))

train_texts['mean_word_len'] = train_texts['comment'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
valid_texts['mean_word_len'] = valid_texts['comment'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
#Truncate some extreme values for better visuals ##
train_texts['num_words'].loc[train_texts['num_words']>60]=60
train_texts['num_punctuations'].loc[train_texts['num_punctuations']>10] = 10
train_texts['num_chars'].loc[train_texts['num_chars']>350]=350

f, axes =plt.subplots(3, 1, figsize=(10,20))
sns.boxplot(x='label', y='num_words', data=train_texts,ax=axes[0])
axes[0].set_xlabel('Label', fontsize=12)
axes[0].set_title('Number of words in each class', fontsize=15)

sns.boxplot(x='label', y='num_chars', data=train_texts,ax=axes[1])
axes[1].set_xlabel('Label', fontsize=12)
axes[1].set_title('Number of characters in each class', fontsize=15)

sns.boxplot(x='label', y='num_punctuations', data=train_texts,ax=axes[2])
axes[2].set_xlabel('Label', fontsize=12)
axes[2].set_title('Number of punctuations in each class', fontsize=15)
#get the tfidf vectors #
tfidf_vec=TfidfVectorizer(stop_words ='english', ngram_range=(1,3))
tfidf_vec.fit_transform(train_texts['comment'].values.tolist() + valid_texts['comment'].values.tolist())
train_tfidf = tfidf_vec.transform(train_texts['comment'].values.tolist())
test_tfidf = tfidf_vec.transform(valid_texts['comment'].values.tolist())
train_y = train_texts['label'].values

def runModel(train_X, train_y, test_X, test_y, test_X2):
    logreg = linear_model.LogisticRegression(C=5, solver = 'sag')
    logreg.fit(train_X, train_y)
    pred_test_y=logreg.predict_proba(test_X)[:,1]
    pred_test_y2=logreg.predict_proba(test_X2)[:,1]
    return pred_test_y, pred_test_y2, logreg

cv_scores =[]
pred_full_test=0
pred_train=np.zeros([train_df.shape[0]])
kf=KFold(n_splits=5,shuffle=True,random_state=2017)
for dev_index,val_index in kf.split(train_texts):
    dev_X, val_X = train_tfidf[dev_index],train_tfidf[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    pred_val_y, pred_test_y, model = runModel(dev_X, dev_y, val_X, val_y, test_tfidf)
    pred_full_test=pred_full_test+pred_test_y
    pred_train[val_index]=pred_val_y
    cv_scores.append(metrics.log_loss(val_y, pred_val_y))
    break
for thresh in np.arange(0.3, 0.401, 0.01):
    thresh = np.round(thresh, 2)
    print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(val_y, (pred_val_y>thresh).astype(int))))
import eli5
eli5.show_weights(model, vec=tfidf_vec, top=100, feature_filter=lambda x: x != '<BIAS>')