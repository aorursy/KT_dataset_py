# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

#!pip install spacy
#!python3 -m spacy download en_core_web_sm
#!pip install wordcloud
#!pip install cufflinks
import os
import re
import string
from collections import Counter, defaultdict
import itertools

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk
nltk.download('stopwords')#Error loading
nltk.download('punkt')#Error loading
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk import word_tokenize, pos_tag
import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)
from sklearn.feature_extraction.text import CountVectorizer

import gensim
from gensim.models import phrases, word2vec

import spacy
from spacy import displacy
import en_core_web_sm
from PIL import Image
import requests
from bs4 import BeautifulSoup

from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt
import seaborn as sns
from bokeh.plotting import figure, output_file, show
from bokeh.models import Label
from bokeh.io import output_notebook
output_notebook()

import plotly.express as px

#!pip install textblob  #Error installing

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
for dirname, _, filenames in os.walk('/kaggle/input/dreams/'): #'/home/C00219805/Learning/dreams/'):#
    for filename in filenames:
        print(filename)
        df = pd.read_csv(os.path.join(dirname, filename), header = 0)
        df.columns = ['id', 'text']
        print(df.head(3))
        print("Total records in dataset: {}".format(len(df)))
        
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
print("Nulls in Datasets: ")
print(df.isnull().sum())
df = df[df['text'].notna()]
# word_count
df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))

# unique_word_count
df['unique_word_count'] = df['text'].apply(lambda x: len(set(str(x).split())))

# stop_word_count
stop_words = set(stopwords.words('english'))
df['stop_word_count'] = df['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in stop_words]))

# mean_word_length
df['mean_word_length'] = df['text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

# char_count
df['char_count'] = df['text'].apply(lambda x: len(str(x)))

# punctuation_count
df['punctuation_count'] = df['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))

print(df.head(3))
df = df[df['mean_word_length'].notna()]
METAFEATURES = ['word_count', 'unique_word_count', 'stop_word_count',  'mean_word_length', 'char_count', 'punctuation_count']
fig, axes = plt.subplots(ncols=1, nrows=len(METAFEATURES), figsize=(6, 10), dpi=100)

for i, feature in enumerate(METAFEATURES):
    sns.distplot(df[feature],ax = axes[i], color='green')    
    axes[i].set_xlabel('')
    axes[i].tick_params(axis='x', labelsize=6)
    axes[i].tick_params(axis='y', labelsize=6)
    axes[i].legend()    
    axes[i].set_title(f'{feature}', fontsize=8)

plt.show()
# Utility Functions for Text Cleaning
# Import spaCy's language model
en_model = en_core_web_sm.load()
#en_model = spacy.load('en', disable=['parser', 'ner'])

# function to lemmatize text
def lemmatization(texts):
    '''Get lemmatized tokens.'''
    output = []
    for i in texts:
        s = [token.lemma_ for token in en_model(i)]
        output.append(' '.join(s))
    return output
def clean_text(text):
    '''Text cleaning including punctation and numbers removal.'''
    if not is_nan(text):
        text = text.lower()
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('<.*?>+', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\n', '', text)
        text = re.sub('\w*\d\w*', '', text)
        return text
def is_nan(x):
    '''Checks if an entity is a null value.'''
    return (x != x)

# Applying the cleaning function to text and remove records with nulls in text
df['cleaned_text'] = df['text'].apply(lambda x: clean_text(x))
print(df['cleaned_text'].head(3))
def get_word_cloud_with_image(data,is_freq, title= None, image = None ):
    stopwords = set(stop_words)
    stopwords.add(".It")
    if is_freq:
        wc = WordCloud( max_words=2000, mask=image, width = 800, height = 800,
                    background_color ='white', 
                    stopwords = stopwords, 
                    min_font_size = 10).generate_from_frequencies(data)
    else:
        wc = WordCloud( max_words=2000, mask=image,
                    background_color ='white', 
                    stopwords = stopwords, 
                    min_font_size = 10).generate(data)

    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.title('Word Cloud of {}'.format(title), size =20)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.show()
#Contacenate all text rows into single string
text = df['text'].str.cat(sep=' ')
tokens = word_tokenize(text)
result = [i for i in tokens if not i.lower() in stop_words]
#lemmatized_result = lemmatization(result)
all_dreams = " ".join(result)

get_word_cloud_with_image(all_dreams, False , title = 'Dreams data')
#Utility functions for n-gram generation
def get_top_n_grams(corpus, n=None, ngram = 1):
    vec = CountVectorizer(ngram_range=(ngram, ngram), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

def plot_n_grams(n, ngram= 1):
    common_words = get_top_n_grams(df['cleaned_text'], n=n, ngram = ngram)
    df4 = pd.DataFrame(common_words, columns = ['ReviewText' , 'count'])
    df4.groupby('ReviewText').sum()['count'].sort_values(ascending=False).iplot(
        kind='bar', yTitle='Count', linecolor='black', title='Top {} {}-grams in dreams'.format(n, ngram))
N = 20
from sklearn.feature_extraction.text import CountVectorizer
plot_n_grams(N, ngram =2)
plot_n_grams(N, ngram =3)
plot_n_grams(N, ngram =4)
def prep_text(in_text):
    return in_text.lower().translate(str.maketrans("", "", string.punctuation)).split()
df['clean_token_text'] = df.apply(lambda row: prep_text(row['cleaned_text']), axis=1)
sentences = df.clean_token_text.values
from gensim.models.phrases import Phrases, Phraser

phrases = Phrases(sentences, min_count=1, threshold=1)
bigram = Phraser(phrases)
print(bigram[df['cleaned_text'][9].split()])

df['cleaned_text'][9]
bigram_counter = Counter()
for key in list(itertools.chain.from_iterable(bigram[sentences])):
    if key not in stopwords.words("english") :
        if len(key.split("_")) > 1 \
        and (key.split("_")[0] not in stopwords.words("english") \
             and key.split("_")[1] not in stopwords.words("english")):
            bigram_counter[key] += 1
for key, counts in bigram_counter.most_common(20):
    print('{0: <20} {1}'.format(key, counts))
#print(bigram_counter['in_cambodia'])
model = word2vec.Word2Vec(bigram[sentences], size=50, min_count=3, iter=20)
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=11)
vocab = list(model.wv.vocab)
#vocab = bigram_counter.keys()
clf = tsne.fit_transform(model[vocab])

tmp = pd.DataFrame(clf, index=vocab, columns=['x', 'y'])

tmp.head(3)
tmp = tmp.reset_index()
print(len(tmp))
print(tmp.columns)
tmp.columns = ['words', 'x', 'y']
tmp['count'] = tmp['words'].map(bigram_counter)
tmp= tmp.fillna(0)
print(len(tmp[tmp['count'] == 0]))
#This step will eliminate unigrams, If you want to
tmp = tmp[tmp['count'] != 0]
print(len(tmp[tmp['count'] == 0]))
print(len(tmp))
tmp1 = tmp.sample(150)
fig = px.scatter(tmp1, x='x', y='y', hover_name='words', text='words', color='words', size = 'count', size_max=45
                 , template='plotly_white', title='Bigram similarity and frequency'
                 , color_continuous_scale=px.colors.sequential.Sunsetdark)
fig.update_traces(marker=dict(line=dict(width=1, color='Gray')))
fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False)
fig.show()
#Pos Tags per dream
pos_tags = df['cleaned_text'].apply(lambda x: pos_tag(word_tokenize(x)))
df['pos_tags'] = pos_tags
print(pos_tags)
#Collecting all POS tags together in a list
all_tags = []
for sent in pos_tags:
    for word, tag in enumerate(sent):
        all_tags.append((word, tag))
pos_counts= Counter([ j[1] for i,j in all_tags])
print(pos_counts)
    
pos_sorted_types = sorted(pos_counts, key=pos_counts.__getitem__, reverse=True)
pos_sorted_counts = sorted(pos_counts.values(), reverse=True)

fig, ax = plt.subplots(figsize=(14,4))
ax.bar(range(len(pos_counts)), pos_sorted_counts);
ax.set_xticks(range(len(pos_counts)));
ax.set_xticklabels(pos_sorted_types);
ax.set_title('Part-of-Speech Tags of Dreams');
ax.set_xlabel('POS Type');
noun_counts= Counter([ j[0] for i,j in all_tags if (j[1] == 'NN') or (j[1] == 'NNP')])
print(len(noun_counts))
#Deleting 'dream' and 'i' which are highest frequency in nouns of the dreams dataset
noun_counts.pop('i', None)
noun_counts.pop('dream', None)

get_word_cloud_with_image(noun_counts,True, title= 'Nouns', image = None )
adj_counts= Counter([ j[0] for i,j in all_tags if (j[1] == 'JJ') or (j[1] == 'JJR') or (j[1] == 'JJS')])
#print(adj_counts)
#Deleting 'dream' and 'i' which are highest frequency in nouns of the dreams dataset
adj_counts.pop('i', None)
adj_counts.pop('dream', None)
adj_counts.pop('other', None)

get_word_cloud_with_image(adj_counts,True, title= 'Adjectives', image = None )
#Loading the required package from Spacy
nlp = en_core_web_sm.load()
displacy.render(nlp(str(df['text'].values[122])), jupyter=True, style='ent')
named_entities = []
entities = []
for sent in df['text'].values:
    article = nlp(sent)
    named_entities.append([(X.text, X.label_) for X in article.ents])
    entities.append(article.ents)
print(named_entities)
named_entities2 = list(itertools.chain.from_iterable(named_entities))
labels = [y for x,y in named_entities2]
named_entities_counter = Counter(labels)
print(Counter(labels))
#Plot 
pos_sorted_types = sorted(named_entities_counter, key=named_entities_counter.__getitem__, reverse=True)
pos_sorted_counts = sorted(named_entities_counter.values(), reverse=True)

fig, ax = plt.subplots(figsize=(14,4))
ax.bar(range(len(named_entities_counter)), pos_sorted_counts);
ax.set_xticks(range(len(named_entities_counter)));
ax.set_xticklabels(pos_sorted_types, rotation = 'vertical', size = 10);
ax.set_title('Named Entities in Dreams');
ax.set_xlabel('Entity Type');
all_entities_dict = {}
for entity_type in named_entities_counter.keys():
    words_entity_dict = {}
    for word, entity in named_entities2:
        if entity == entity_type:
            if word in words_entity_dict:
                words_entity_dict[word] += 1
            else:
                words_entity_dict[word] = 1
    all_entities_dict[entity_type] = words_entity_dict
    
#print(all_entities_dict)
person_mask = np.array(Image.open("/kaggle/input/images/trump.png"))
get_word_cloud_with_image(all_entities_dict['PERSON'], True, 'PERSON', image = person_mask)

gpe_mask = np.array(Image.open("/kaggle/input/images3/world.png"))
get_word_cloud_with_image(all_entities_dict['GPE'], True, 'GPE', image = gpe_mask)

org_mask = np.array(Image.open("/kaggle/input/images2/tree.png"))
get_word_cloud_with_image(all_entities_dict['ORG'], True, 'ORGANIZATION', image = org_mask)

# initialize afinn sentiment analyzer
!pip install afinn
from afinn import Afinn
af = Afinn()

# compute sentiment scores (polarity) and labels
sentiment_scores = [af.score(dream) for dream in df['cleaned_text'].values]
sentiment_category = ['positive' if score > 0 
                          else 'negative' if score < 0 
                              else 'neutral' 
                                  for score in sentiment_scores]
    
print(df.columns)    
# sentiment statistics per news category
df_senti = pd.DataFrame([list(df['text']), list(df['cleaned_text']), sentiment_scores, sentiment_category]).T
df_senti.columns = ['text', 'cleaned_text', 'sentiment_score', 'sentiment_category']
df_senti['sentiment_score'] = df_senti.sentiment_score.astype('float')
df_senti.head(5)
print(df_senti[['sentiment_score','sentiment_category']].iloc[227])
print(df_senti['text'].values[227])
print('\n')
print(df_senti[['sentiment_score','sentiment_category']].iloc[4])
print(df_senti['text'].values[4])
print('\n')
print(df_senti[['sentiment_score','sentiment_category']].iloc[18])
print(df_senti['text'].values[18])
grouped = df_senti.groupby([ 'sentiment_category']).describe().reset_index()
grouped
#!pip install plotly
df_g =pd.DataFrame({'sentiment_category':grouped['sentiment_category'], 
                    'counts': grouped['sentiment_score']['count']})
fig = px.treemap(df_g,  path=['sentiment_category'], values = 'counts',
                   title = 'Sentiment Polarity Distribution', 
                color_discrete_sequence = px.colors.qualitative.Dark2)
fig.data[0].textinfo = 'label+text+value'
fig.show()
fig = px.histogram(df, x="sentiment_score", title = 'Sentiment Score Distribution',  nbins=100)
fig.show()
print(df[df['sentiment_score'] == -62]['text'].values[0])
print(df[df['sentiment_score'] == 95]['text'].values[0])
def generate_bar_plots_sentiment(sentiment):
    #print(df.columns)
    blob = list(itertools.chain.from_iterable(df[df['sentiment_category'] == sentiment]['pos_tags'].values))
    pos_df = pd.DataFrame(blob, columns = ['word' , 'pos'])
    common_words = get_top_n_grams(pos_df[(pos_df['pos'] == 'JJ') | 
                                          (pos_df['pos'] == 'JJR') | 
                                          (pos_df['pos'] == 'JJS')]['word'], n = 20, ngram = 1)
    df1 = pd.DataFrame(common_words, columns = ['words' , 'count'])
    fig = df1.groupby('words').sum()['count'].sort_values(ascending=False).iplot(
        kind='bar', yTitle='Count', linecolor='black', title='Top 20 words in {} sentiment'.format(sentiment))
    #fig.update_layout(width=1200, height=500)
    #fig.show()

#from textblob import TextBlob
generate_bar_plots_sentiment('positive')
generate_bar_plots_sentiment('negative')
generate_bar_plots_sentiment('neutral')
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from textblob import TextBlob
import scipy.stats as stats

from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE

from bokeh.plotting import figure, output_file, show
from bokeh.models import Label
#from bokeh.io import output_notebook
#count_vectorizer = CountVectorizer(stop_words='english', max_features=40000)
count_vectorizer = TfidfVectorizer(stop_words='english', max_features=40000)
text_sample = df['cleaned_text'].values

print('Dreams before vectorization: {}'.format(text_sample[1]))

document_term_matrix = count_vectorizer.fit_transform(text_sample)

print('Dreams after vectorization: \n{}'.format(document_term_matrix[1]))
print(document_term_matrix.shape[1])
# Define helper functions
def get_keys(topic_matrix):
    '''
    returns an integer list of predicted topic 
    categories for a given topic matrix
    '''
    keys = topic_matrix.argmax(axis=1).tolist()
    return keys

def keys_to_counts(keys):
    '''
    returns a tuple of topic categories and their 
    accompanying magnitudes for a given list of keys
    '''
    count_pairs = Counter(keys).items()
    categories = [pair[0] for pair in count_pairs]
    counts = [pair[1] for pair in count_pairs]
    return (categories, counts)

# Define helper functions
def get_mean_topic_vectors(keys, two_dim_vectors):
    '''
    returns a list of centroid vectors from each predicted topic category
    '''
    mean_topic_vectors = []
    for t in range(n_topics):
        articles_in_that_topic = []
        for i in range(len(keys)):
            if keys[i] == t:
                articles_in_that_topic.append(two_dim_vectors[i])    
        #print(t, np.mean(articles_in_that_topic, axis = 0))
        #articles_in_that_topic = np.vstack(articles_in_that_topic)
        
        mean_article_in_that_topic = np.mean(articles_in_that_topic, axis=0)
        mean_topic_vectors.append(mean_article_in_that_topic)
    return mean_topic_vectors
n_topics = 20

colormap = np.array([
    "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
    "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
    "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f",
    "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5" ])
colormap = colormap[:n_topics]

#LDA Model Generation
lda_model = LatentDirichletAllocation(n_components=n_topics, learning_method='online', 
                                          random_state=0, verbose=0)
lda_topic_matrix = lda_model.fit_transform(document_term_matrix)
lda_keys = get_keys(lda_topic_matrix)
lda_categories, lda_counts = keys_to_counts(lda_keys)
terms = count_vectorizer.get_feature_names()
topic_names = []

for i, comp in enumerate(lda_model.components_):
    terms_comp = zip(terms, comp)
    #print(comp)
    sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:4]#Change this number to indicate the #words you want to see per topic
    print("Topic "+str(i)+": ")
    topic = []
    for t in sorted_terms:
        topic.append(t[0])
    print(" ".join(topic))
    topic_names .append(" ".join(topic))
print(topic_names)

tsne_lda_model = TSNE(n_components=2, perplexity=50, learning_rate=100, 
                        n_iter=2000, verbose=1, random_state=0, angle=0.75)
tsne_lda_vectors = tsne_lda_model.fit_transform(lda_topic_matrix)

lda_mean_topic_vectors = get_mean_topic_vectors(lda_keys, tsne_lda_vectors)

plot = figure(title="t-SNE Clustering of {} LDA Topics".format(n_topics), plot_width=800, plot_height=800)
plot.scatter(x=tsne_lda_vectors[:,0], y=tsne_lda_vectors[:,1], color=colormap[lda_keys])

for t in range(n_topics):
    label = Label(x=lda_mean_topic_vectors[t][0], y=lda_mean_topic_vectors[t][1], 
                  text=topic_names[t], text_color=colormap[t])
    plot.add_layout(label)

show(plot)
print(len(df))
page = requests.get("http://sleepanddreamdatabase.org/dream/search?searchconstraint={}")
soup = BeautifulSoup(page.content, 'html.parser')
dreams = soup.find_all('div', class_='searchhittext')
all_dreams = [dream.get_text().replace('\n\t\t\t\t\t\t','') for dream in dreams]
all_dreams = [dream.replace('\n\t\t\t\t','') for dream in all_dreams]
all_dreams.extend(df['dream'].values)
all_dreams = list(dict.fromkeys(all_dreams))
df = pd.DataFrame({'dreams_text' :all_dreams})
df.to_csv('dreams.csv')
