from ipywidgets import Image
f = open("/kaggle/input/image-covid/sars_imp.PNG", "rb")
image = f.read()
Image(value=image)
f = open("/kaggle/input/coronavirus-image/SARS-CoV-2_without_background.png", "rb")
image = f.read()
Image(value=image)
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import json
import scipy
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import tqdm
!ls /kaggle/input/CORD-19-research-challenge/
root_path = '/kaggle/input/CORD-19-research-challenge/'
metadata_path = f'{root_path}/metadata.csv'
meta_df = pd.read_csv(metadata_path, dtype={
    'pubmed_id': str,
    'Microsoft Academic Paper ID': str, 
    'doi': str
})
meta_df.head()
meta_df['source_x'].value_counts(normalize=True).plot(kind='bar',figsize=(5,5))
meta_df.apply(lambda x:sum(x.isna()))
meta_df['year']=pd.to_datetime(meta_df.publish_time).dt.year
meta_df['month']=pd.to_datetime(meta_df.publish_time).dt.month

meta_19_20=meta_df.loc[(meta_df.year.isin([2019,2020]))]

def get_data(dataframe):
    
    if dataframe['year']==2019:
        dataframe=dataframe.loc[dataframe['month'].isin([11,12])]
    return dataframe

meta_19_20=meta_19_20.sort_values(by=['year','month'])
meta_19=meta_19_20.iloc[np.hstack(np.argwhere((meta_19_20['year']==2019) & meta_19_20['month'].isin([11,12]))),:]
meta_20=meta_19_20.loc[meta_19_20['year']==2020]

meta_df=pd.concat([meta_19,meta_20],ignore_index=True)
meta_df.shape
meta_df.apply(lambda x:sum(x.isna()))
all_json = glob.glob(f'{root_path}/**/*.json', recursive=True)
len(all_json)
class FileReader:
    def __init__(self, file_path):
        with open(file_path) as file:
            content = json.load(file)
            self.paper_id = content['paper_id']
            self.abstract = []
            self.body_text = []
            # Abstract
            try:
                if content['abstract']:
                
                    for entry in content['abstract']:
                        self.abstract.append(entry['text'])
            except:
                self.abstract.append('NA')
                
            
            # Body text
            for entry in content['body_text']:
                self.body_text.append(entry['text'])
            self.abstract = '\n'.join(self.abstract)
            self.body_text = '\n'.join(self.body_text)
    def __repr__(self):
        return f'{self.paper_id}: {self.abstract[:200]}... {self.body_text[:200]}...'
first_row = FileReader(all_json[10000])
print(first_row)
def get_breaks(content, length):
    data = ""
    words = content.split(' ')
    total_chars = 0

    # add break every length characters
    for i in range(len(words)):
        total_chars += len(words[i])
        if total_chars > length:
            data = data + "<br>" + words[i]
            total_chars = 0
        else:
            data = data + " " + words[i]
    return data
dict_ = {'paper_id': [], 'abstract': [], 'body_text': [], 'authors': [], 'title': [], 'journal': [], 'abstract_summary': []}
for idx, entry in tqdm.tqdm(enumerate(all_json)):
    
    content = FileReader(entry)
    
    # get metadata information
    meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]
    # no metadata, skip this paper
    if len(meta_data) == 0:
        continue
    
    dict_['paper_id'].append(content.paper_id)
    dict_['abstract'].append(content.abstract)
    dict_['body_text'].append(content.body_text)
    
    # also create a column for the summary of abstract to be used in a plot
    if len(content.abstract) == 'NA': 
        # no abstract provided
        dict_['abstract_summary'].append("Not provided.")
    elif len(content.abstract.split(' ')) > 100:
        # abstract provided is too long for plot, take first 300 words append with ...
        info = content.abstract.split(' ')[:100]
        summary = get_breaks(' '.join(info), 40)
        dict_['abstract_summary'].append(summary + "...")
    else:
        # abstract is short enough
        summary = get_breaks(content.abstract, 40)
        dict_['abstract_summary'].append(summary)
        
    # get metadata information
    meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]
    
    try:
        # if more than one author
        authors = meta_data['authors'].values[0].split(';')
        if len(authors) > 2:
            # more than 2 authors, may be problem when plotting, so take first 2 append with ...
            dict_['authors'].append(". ".join(authors[:2]) + "...")
        else:
            # authors will fit in plot
            dict_['authors'].append(". ".join(authors))
    except Exception as e:
        # if only one author - or Null valie
        dict_['authors'].append(meta_data['authors'].values[0])
    
    # add the title information, add breaks when needed
    try:
        title = get_breaks(meta_data['title'].values[0], 40)
        dict_['title'].append(title)
    # if title was not provided
    except Exception as e:
        dict_['title'].append(meta_data['title'].values[0])
    
    # add the journal information
    dict_['journal'].append(meta_data['journal'].values[0])
    
df_covid = pd.DataFrame(dict_, columns=['paper_id', 'abstract', 'body_text', 'authors', 'title', 'journal', 'abstract_summary'])
df_covid.head()
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
import os
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from gensim.models.phrases import Phrases, Phraser
from gensim.models.word2vec import LineSentence
stop = stopwords.words('english')

def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma
    

def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)

def text_cleaner(Series):
    
    Series=Series.dropna()
    
    Series=Series.apply(lambda x: " ".join(word.lower() for word in str(x).split()))
    Series=Series.apply(lambda x: " ".join(word for word in str(x).split() if word not in stop))
    Series=Series.str.replace('[^\w\s]','')
    Series=Series.apply(lambda x: " ".join(get_lemma(word) for word in str(x).split()))
    Series=Series.apply(lambda x: " ".join(word for word in str(x).split() if len(word)>3))
    Series=Series.apply(lambda x:" ".join(word for word in str(x).split() if word.isalpha()))
    
    months=['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']
    
    Series=Series.apply(lambda x:" ".join(word for word in str(x).split() if word not in months))
    
    
    unigrams = Series.apply(word_tokenize)
    bigram_phrases = Phrases(unigrams)
        
    bigram_phrases = Phraser(bigram_phrases)
    
    sentences_bigrams_filepath = os.path.join(os.getcwd(), str(Series.name)+'_sentence_bigram_phrases_all.txt')
    
    with open(sentences_bigrams_filepath, 'w') as f:
        
        for sentence_unigrams in tqdm.tqdm(unigrams):
            
            sentence_bigrams = ' '.join(bigram_phrases[sentence_unigrams])
            
            f.write(sentence_bigrams + '\n')
    sentences_bigrams = LineSentence(sentences_bigrams_filepath)
    
    for sentence_bigrams in tqdm.tqdm(it.islice(sentences_bigrams, 60, 70)):
        print(' '.join(sentence_bigrams))
        print('')
    
    trigram_phrases = Phrases(sentences_bigrams)
    
    # Turn the finished Phrases model into a "Phraser" object,
    # which is optimized for speed and memory use
    trigram_phrases = Phraser(trigram_phrases)
    
    sentences_trigrams_filepath = os.path.join(os.getcwd(),str(Series.name)+ '_sentence_trigram_phrases_all.txt')
    with open(sentences_trigrams_filepath, 'w') as f:
        
        for sentence_bigrams in tqdm.tqdm(sentences_bigrams):
            
            sentence_trigrams = ' '.join(trigram_phrases[sentence_bigrams])
            
            f.write(sentence_trigrams + '\n')
            
    sentences_trigrams = LineSentence(sentences_trigrams_filepath)
    
    for sentence_trigrams in tqdm.tqdm(it.islice(sentences_trigrams, 60, 70)):
        print(' '.join(sentence_trigrams))
        print('')
        
    return sentences_trigrams_filepath,Series

    

from collections import Counter


import gensim
from gensim.corpora import Dictionary, MmCorpus
from gensim.models.ldamulticore import LdaMulticore
from gensim import corpora, models

import itertools as it

import pyLDAvis
import pyLDAvis.gensim
import warnings
import pickle

from pprint import pprint



def bow_generator(filepath,dictionary_trigrams):
    """
    generator function to read reviews from a file
    and yield a bag-of-words representation
    """
    
    for review in LineSentence(filepath):
        yield dictionary_trigrams.doc2bow(review)

def topic_modeling(Series_path):
    series_list=LineSentence(Series_path)

    lists=[]
    for item in series_list:
        lists.append(item)
        
    tokens=word_tokenize(' '.join(word for item in lists for word in item))
    
    print(Counter(tokens).most_common(50))
    
    articles_trigrams = LineSentence(Series_path)

    # learn the dictionary by iterating over all of the reviews
    dictionary_trigrams = Dictionary(articles_trigrams)
    
    bow_corpus_filepath = os.path.join(os.getcwd(), 'bow_trigrams_corpus_all.mm')
    
    MmCorpus.serialize(
        bow_corpus_filepath,
        bow_generator(Series_path,dictionary_trigrams)
        )
    
    trigram_bow_corpus = MmCorpus(bow_corpus_filepath)
    
    
    tfidf = models.TfidfModel(trigram_bow_corpus)
    corpus_tfidf = tfidf[trigram_bow_corpus]
    
        
    lda_bow_model = gensim.models.LdaMulticore(trigram_bow_corpus, num_topics=30, id2word=dictionary_trigrams, passes=2, workers=2)
    
    print('LDA BoW MODEL')
    for idx, topic in lda_bow_model.print_topics(-1):
        print('Topic: {} \nWords: {}'.format(idx, topic))
        
    lda_tfidf_model = gensim.models.LdaMulticore(corpus_tfidf, num_topics=30, id2word=dictionary_trigrams, passes=2, workers=2)
    
    print('LDA TFIDF MODEL')
    for idx, topic in lda_tfidf_model.print_topics(-1):
        print('Topic: {} \nWords: {}'.format(idx, topic))
        
    return lda_bow_model,lda_tfidf_model,dictionary_trigrams,trigram_bow_corpus,corpus_tfidf
    
    
        
    
abstract_path,clean_abstract=text_cleaner(df_covid['body_text'])
lda_bow_model,lda_tfidf_model,dictionary_trigrams,trigram_bow_corpus,corpus_tfidf=topic_modeling(abstract_path)
f = open("/kaggle/input/topic-eg/topic_3.PNG", "rb")
image = f.read()
Image(value=image)
f = open("/kaggle/input/topic-eg/topic_4.PNG", "rb")
image = f.read()
Image(value=image)
f = open("/kaggle/input/topic-eg/topic_5.PNG", "rb")
image = f.read()
Image(value=image)
f = open("/kaggle/input/topics-used/topic_1.PNG", "rb")
image = f.read()
Image(value=image)
f = open("/kaggle/input/topics-used/topic_2.PNG", "rb")
image = f.read()
Image(value=image)
def explore_topic(model,topic_number, topn=25):
    """
    accept a user-supplied topic number and
    print out a formatted list of the top terms
    """
        
    print(f'{"term":20} {"frequency"}' + '\n')

    for term, frequency in model.show_topic(topic_number, topn=40):
        print(f'{term:20} {round(frequency, 3):.3f}')
explore_topic(lda_bow_model,10)
explore_topic(lda_tfidf_model,10)

from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
virus_mask = np.array(Image.open("/kaggle/input/coronavirus-image/SARS-CoV-2_without_background.png"))
stopwords = set(STOPWORDS)
text=" ".join(word for abstract in clean_abstract for word in abstract.split())
wc = WordCloud(background_color="white", max_words=1000, mask=virus_mask,
               stopwords=stopwords, max_font_size=50, random_state=42,contour_width=4, contour_color='firebrick')
# generate word cloud
wc.generate(text)

# create coloring from image
image_colors = ImageColorGenerator(virus_mask)

# show
fig, axes = plt.subplots(1,2,figsize=(50,50))
# recolor wordcloud and show
# we could also give color_func=image_colors directly in the constructor
axes[0].imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")
axes[1].imshow(virus_mask, cmap=plt.cm.gray, interpolation="bilinear")
for ax in axes:
    ax.set_axis_off()
plt.show()
from gensim.models import Word2Vec

sentences_trigrams = LineSentence(abstract_path)
word2vec_filepath = os.path.join(os.getcwd(), 'word2vec_model_all')

corona2vec = Word2Vec(
        sentences_trigrams,
        size=100,
        window=5,
        min_count=50,
        sg=1,
        workers=7,
        iter=20
        )

corona2vec.init_sims()

print(f'{corona2vec.epochs} training epochs so far.')

print(f'{len(corona2vec.wv.vocab):,} terms in the corona2vec vocabulary.')

# build a list of the terms, integer indices,
# and term counts from the food2vec model vocabulary
ordered_vocab = [
    (term, voc.index, voc.count)
    for term, voc in corona2vec.wv.vocab.items()
    ]

# sort by the term counts, so the most common terms appear first
ordered_vocab = sorted(ordered_vocab, key=lambda term_tuple: -term_tuple[2])

# unzip the terms, integer indices, and counts into separate lists
ordered_terms, term_indices, term_counts = zip(*ordered_vocab)

# create a DataFrame with the food2vec vectors as data,
# and the terms as row labels
word_vectors = pd.DataFrame(
    corona2vec.wv.vectors_norm[term_indices, :],
    index=ordered_terms
    )

word_vectors
from sklearn.manifold import TSNE

tsne_input = (
    word_vectors
    .head(5000)
    )

tsne_input.head()

tsne_filepath = os.path.join(os.getcwd(), 'tsne_model')

tsne_vectors_filepath = os.path.join(os.getcwd(), 'tsne_vectors.npy')

tsne = TSNE()
tsne_vectors = tsne.fit_transform(tsne_input.values)
    
with open(tsne_filepath, 'wb') as f:
    pickle.dump(tsne, f)
    
    
tsne_vectors = pd.DataFrame(
    tsne_vectors,
    index=pd.Index(tsne_input.index),
    columns=['x_coord', 'y_coord']
    )

tsne_vectors['word'] = tsne_vectors.index
tsne_vectors.head()
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import HoverTool, ColumnDataSource, value

output_notebook()

# add our DataFrame as a ColumnDataSource for Bokeh
plot_data = ColumnDataSource(tsne_vectors)

# create the plot and configure the
# title, dimensions, and tools
tsne_plot = figure(
    title='t-SNE Word Embeddings',
    plot_width=800,
    plot_height=800,
    tools=(
        'pan, wheel_zoom, box_zoom,'
        'box_select, reset'
        ),
    active_scroll='wheel_zoom'
    )

# add a hover tool to display words on roll-over
tsne_plot.add_tools(
    HoverTool(tooltips = '@word')
    )

# draw the words as circles on the plot
tsne_plot.circle(
    'x_coord',
    'y_coord',
    source=plot_data,
    color='blue',
    line_alpha=0.2,
    fill_alpha=0.1,
    size=10,
    hover_line_color='black'
    )

# configure visual elements of the plotc
tsne_plot.title.text_font_size = value('16pt')
tsne_plot.xaxis.visible = False
tsne_plot.yaxis.visible = False
tsne_plot.grid.grid_line_color = None
tsne_plot.outline_line_color = None

# engage!
show(tsne_plot);
from matplotlib import cm
def get_related_terms(token, topn=20):
    """
    look up the topn most similar terms to token
    and print them as a formatted list
    """
    
    words,similarities=[],[]
    
    for word, similarity in corona2vec.wv.most_similar(positive=[token], topn=topn):

        print(f'{word:20} {round(similarity, 3)}')
        
        words.append(word)
        
        similarities.append(similarity)
        
    plt.style.use('ggplot')
    
    
    pd.DataFrame(data=similarities,index=words).sort_values(by=0,ascending=True).plot(kind='barh',cmap=cm.get_cmap('Spectral'),figsize=(10,10))
    
    
    
get_related_terms('intervention')
get_related_terms('medical_care')
get_related_terms('vaccine')
get_related_terms('infection')
get_related_terms('steroid')
get_related_terms('virus')
get_related_terms('protein')
get_related_terms('research')
get_related_terms('public_health_intervention')
get_related_terms('treatment')
get_related_terms('medicine')
get_related_terms('care')
get_related_terms('symptom')
get_related_terms('disease')
get_related_terms('literature')
get_related_terms('covid')
get_related_terms('sars')
get_related_terms('cure')
get_related_terms('medical')
LDAvis_prepared = pyLDAvis.gensim.prepare(
        lda_tfidf_model,
        trigram_bow_corpus,
        dictionary_trigrams
        )
pyLDAvis.display(LDAvis_prepared)
from sklearn.feature_extraction.text import CountVectorizer

norm_corpus=text_cleaner(df_covid1['abstract'])
cv = CountVectorizer(min_df=0., max_df=1.)
cv_matrix = cv.fit_transform(norm_corpus)
cv_matrix = cv_matrix.toarray()
cv_matrix