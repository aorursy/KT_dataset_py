from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.tokenize import word_tokenize
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF

from collections import Counter
import re

%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
tokenizer = Tokenizer(num_words=100, oov_token='OOV')
tokenizer.fit_on_texts(df.head()['text'])
word_index = tokenizer.word_index
print(word_index)
sequences = tokenizer.texts_to_sequences(df.head()['text'])
print(sequences)
padded = pad_sequences(sequences, padding='post')
print(padded)
def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'', text)


def remove_emoji(text):
    emoji_pattern = re.compile(
        '['
        u'\U0001F600-\U0001F64F' 
        u'\U0001F300-\U0001F5FF' 
        u'\U0001F680-\U0001F6FF' 
        u'\U0001F1E0-\U0001F1FF' 
        u'\U00002702-\U000027B0'
        u'\U000024C2-\U0001F251'
        ']+',
        flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def remove_html(text):
    html = re.compile(r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    return re.sub(html, '', text)


def remove_punct(text):
    table = str.maketrans('', '', string.punctuation)
    return text.translate(table)

def remove_num(text):
    text = ''.join([i for i in text if not i.isdigit()])
    return text


df['text_clean'] = df['text'].apply(lambda x: remove_URL(x))
df['text_clean'] = df['text_clean'].apply(lambda x: remove_emoji(x))
df['text_clean'] = df['text_clean'].apply(lambda x: remove_html(x))
df['text_clean'] = df['text_clean'].apply(lambda x: remove_punct(x))
df['text_clean'] = df['text_clean'].apply(lambda x: remove_num(x))

test['text_clean'] = test['text'].apply(lambda x: remove_URL(x))
test['text_clean'] = test['text_clean'].apply(lambda x: remove_emoji(x))
test['text_clean'] = test['text_clean'].apply(lambda x: remove_html(x))
test['text_clean'] = test['text_clean'].apply(lambda x: remove_punct(x))
test['text_clean'] = test['text_clean'].apply(lambda x: remove_num(x))
df['tokenized'] = df['text_clean'].apply(word_tokenize)
test['tokenized'] = test['text_clean'].apply(word_tokenize)
df['lower'] = df['tokenized'].apply(lambda x: [word.lower() for word in x])
test['lower'] = test['tokenized'].apply(lambda x: [word.lower() for word in x])
stop = set(stopwords.words('english'))
df['stopwords_removed'] = df['lower'].apply(lambda x: [word for word in x if word not in stop])
test['stopwords_removed'] = test['lower'].apply(lambda x: [word for word in x if word not in stop])
df['pos_tags'] = df['stopwords_removed'].apply(nltk.tag.pos_tag)
test['pos_tags'] = test['stopwords_removed'].apply(nltk.tag.pos_tag)
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


df['wordnet_pos'] = df['pos_tags'].apply(lambda x: [(word, get_wordnet_pos(pos_tag)) for (word, pos_tag) in x])
test['wordnet_pos'] = test['pos_tags'].apply(lambda x: [(word, get_wordnet_pos(pos_tag)) for (word, pos_tag) in x])
wnl = WordNetLemmatizer()
df['lemmatized'] = df['wordnet_pos'].apply(
    lambda x: [wnl.lemmatize(word, tag) for word, tag in x])
df['lemmatized'] = df['lemmatized'].apply(
    lambda x: [word for word in x if word not in stop])
df['lemma_str'] = [' '.join(map(str, l)) for l in df['lemmatized']]

test['lemmatized'] = test['wordnet_pos'].apply(
    lambda x: [wnl.lemmatize(word, tag) for word, tag in x])
test['lemmatized'] = test['lemmatized'].apply(
    lambda x: [word for word in x if word not in stop])
test['lemma_str'] = [' '.join(map(str, l)) for l in test['lemmatized']]
trace = go.Pie(labels = ['Disaster : no', 'Disaster : yes'], values = df['target'].value_counts(), 
               textfont=dict(size=15), opacity = 0.8,
               marker=dict(colors=['lightblue','gold'], 
                           line=dict(color='#000000', width=1.5)))


layout = dict(title =  'Real or not?',
                        autosize = False,
                        height  = 500,
                        width   = 800)
           
fig = dict(data = [trace], layout=layout)
iplot(fig)
df['Character_Count'] = df['text_clean'].apply(lambda x: len(str(x).split()))
def kdeplot(feature):
    plt.figure(figsize=(9, 4))
    plt.title("KDE for {}".format(feature))
    ax0 = sns.kdeplot(df[df['target'] == 1][feature].dropna(), color= 'navy', label= 'Dissaster: Yes')
    ax1 = sns.kdeplot(df[df['target'] == 0][feature].dropna(), color= 'orange', label= 'Disaster: No')
kdeplot('Character_Count')
results = Counter()
df['lemma_str'].str.lower().str.split().apply(results.update)
text = df['lemma_str']
plt.subplots(figsize=(16,12))
wordcloud = WordCloud(
                          background_color='white',
                          width=800,
                          height=600
                         ).generate(" ".join(text))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
disaster = df[df['target'] == 1]
non_disaster = df[df['target'] == 0]
text = disaster['lemma_str']
plt.subplots(figsize=(16,12))
wordcloud = WordCloud(
                          background_color='white',
                          width=800,
                          height=600
                         ).generate(" ".join(text))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
count = Counter(df['lemma_str'])
most_common = count.most_common(10)
lis = [
    df[df['target'] == 0]['lemma_str'],
    df[df['target'] == 1]['lemma_str']
]
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
axes = axes.flatten()

for i, j in zip(lis, axes):

    new = i.str.split()
    new = new.tolist()
    corpus = [word for i in new for word in i]

    counter = Counter(corpus)
    most = counter.most_common()
    x, y = [], []
    for word, count in most[:30]:
        if (word not in stop):
            x.append(word)
            y.append(count)

    sns.barplot(x=y, y=x, palette='plasma', ax=j)
axes[0].set_title('Non Disaster Tweets')

axes[1].set_title('Disaster Tweets')
axes[0].set_xlabel('Count')
axes[0].set_ylabel('Word')
axes[1].set_xlabel('Count')
axes[1].set_ylabel('Word')

fig.suptitle('Most Common Unigrams', fontsize=24, va='baseline')
plt.tight_layout()
def ngrams(n, title):
    """A Function to plot most common ngrams"""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    axes = axes.flatten()
    for i, j in zip(lis, axes):

        new = i.str.split()
        new = new.values.tolist()
        corpus = [word for i in new for word in i]

        def _get_top_ngram(corpus, n=None):
            #getting top ngrams
            vec = CountVectorizer(ngram_range=(n, n),
                                  max_df=0.9,
                                  stop_words='english').fit(corpus)
            bag_of_words = vec.transform(corpus)
            sum_words = bag_of_words.sum(axis=0)
            words_freq = [(word, sum_words[0, idx])
                          for word, idx in vec.vocabulary_.items()]
            words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
            return words_freq[:15]

        top_n_bigrams = _get_top_ngram(i, n)[:15]
        x, y = map(list, zip(*top_n_bigrams))
        sns.barplot(x=y, y=x, palette='plasma', ax=j)
        
        axes[0].set_title('Non Disaster Tweets')
        axes[1].set_title('Disaster Tweets')
        axes[0].set_xlabel('Count')
        axes[0].set_ylabel('Words')
        axes[1].set_xlabel('Count')
        axes[1].set_ylabel('Words')
        fig.suptitle(title, fontsize=24, va='baseline')
        plt.tight_layout()
ngrams(3, 'The trigram')
from sklearn import feature_extraction, linear_model, model_selection
count_vectorizer = feature_extraction.text.CountVectorizer()
train_vectors = count_vectorizer.fit_transform(df["lemma_str"])
clf = linear_model.RidgeClassifier()
scores = model_selection.cross_val_score(clf, train_vectors, df["target"], cv=3, scoring="f1")

scores
clf.fit(train_vectors, df["target"])
test_vectors = count_vectorizer.transform(test["lemma_str"])
submission = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
submission['target'] = clf.predict(test_vectors)
submission.to_csv("submission.csv", index=False)