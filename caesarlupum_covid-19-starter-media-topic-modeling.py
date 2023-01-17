import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from wordcloud import WordCloud, STOPWORDS

from textblob import TextBlob

import scipy.stats as stats

import spacy 



from tqdm import tqdm

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.decomposition import LatentDirichletAllocation,  PCA, NMF

import random 



import gensim

from gensim import corpora, models, similarities



import logging

import tempfile

from nltk.corpus import stopwords

from string import punctuation

from collections import OrderedDict

import seaborn as sns

import pyLDAvis.gensim

import pyLDAvis.sklearn

import matplotlib.pyplot as plt

%matplotlib inline



# Suppress warnings 

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings("ignore", category=UserWarning)

warnings.filterwarnings("ignore", category=FutureWarning)

from IPython.display import HTML



# Dimension reduction

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

from sklearn.manifold import Isomap

from sklearn.decomposition import FactorAnalysis
df = pd.read_csv('/kaggle/input/covid19-public-media-dataset/covid19_articles_20200512.csv',index_col='Unnamed: 0')

df2 = pd.read_csv('/kaggle/input/covid19-public-media-dataset/covid19_articles_20200526.csv',index_col='Unnamed: 0')

df3 = pd.read_csv('/kaggle/input/covid19-public-media-dataset/covid19_articles_20200504.csv',index_col='Unnamed: 0')

# concatenating sources 

df = pd.concat([df,df2,df3],ignore_index=True)
# test for 1000 rows

df = df.head(500)

df.head()


text_example = ['novel coronavirus strain infect least people kill patient disease emerge china december last year coronavirus initial symptom include dry cough fever pneumonia kidney failure death leave untreated chinese health official confirm monday january virus jump people fuel fear global epidemic brewing big threat hand call asymptomatic carrier people infect coronavirus show symptom ill dr jeremy farrar director non profit charity wellcome warn true number infect different official count major concern range severity symptom virus cause clear people affect infectious experience mild symptom experience symptom asymptomatic read more warn coronavirus expect spread china may mask true number infected extent person person transmission matter urgency work novel coronavirus 2019-ncov belong family pathogens responsible sars pandemic sars severe acute respiratory syndrome kill least people infect more human contract sars cov virus civet cat asia coronavirus family zoonotic mean spread animal human world health organization believe many undiscovered strain wild time coronavirus outbreak trace seafood market wuhan city hubei province prevent virus spread authority stop travel strong city chinese official suspend bus subway system airport train link wuhan dr farrar outbreak concern person person transmission confirm expect increase case number china more country health care worker infect may mask true number infect dr jeremy farrar wellcome world health organization role ensure global public health response outbreak rapid robust comprehensive geographic spread case call emergency committee consider declare international public health emergency part process coronavirus break china infection confirm japan south korea thailand us accord who virus contract traveller wuhan don’t misscoronavirus symptom coronavirus chinese coronavirus strike map]how dangerous china coronavirus analysis concern mount increase travel public celebration mark chinese year saturday january dr farrar speed virus identify testament change public health china sars strong global coordination who know more outbreak travel huge part approach chinese year right concern level high level scientist able devise vaccine virus understand coronavirus strain preventative medication official urge traveller maintain good standard hygiene avoid contact raw food infected people dr farrar world prepare identify patient take necessary public health clinical measure sars decade understand virus public health clinical impact urgent focus evidence base intervention prove treatment vaccine cepi coalition epidemic preparedness innovations wellcome support work global partner accelerate vaccine research virus today front back page download newspaper order issue historic daily express newspaper archive']

text_example2 = ["coronavirus sweep globe chaotic epidemic virus continue spread epicentre china country world flu like illness think begin wuhan transfer country include france germany us death toll skyrocket death remain china present bid reduce further spread disease protect traveller uk airline british airways virgin atlantic amend booking policy flight mainland china weekend british airways introduce flexible booking policy ticket destine affected region january february airline allow passenger opportunity rebook alternative flight later date cancel booking penalty term condition lay initial booking read more coronavirus cruise line cancel sailing death toll rise update ba spokesperson express.co.uk understand customer may want change travel plan time order flexible possible await further advice government health organisation offer customer travel china february ability receive refund rebook flight country continue monitor situation virgin atlantic offer same flexible policy passenger book hong kong shanghai codeshare destination china passenger rebook cancel receive full refund book term condition don't missflight attendant reveal sad thing passenger insider]best bad cruise line reveal analysis]plane passenger ban bring onboard insight virgin atlantic spokesperson be monitor situation regard coronavirus follow guidance set relevant authority urge customer visit foreign commonwealth office travel web page more information travel affect area customer book travel china include hong kong like discuss travel plan invite contact customer care team sms messaging system +44 team happy assist enquiry time write foreign commonwealth office advise travel hubei province statement fco website urge be area able leave january wuhan authority close transport hub include airport railway bus station shop amenity closed public event cancel public health uk enforce monitoring flight wuhan uk statement public health england state enhanced monitoring package include number measure help provide advice traveller feel unwell travel wuhan include port health team meet direct flight aircraft provide advice support feel unwell team include principal port medical inspector port health doctor administrative support team leader check symptom coronavirus provide information passenger symptom become ill fco update travel information visit country surround china include thailand mongolia sri lanka marshall islands multiple major cruise line globe cancel journey mainland port royal caribbean msc passenger expect sail affected cruise line offer full refund port change royal caribbean cancel january sailing cruise ship spectrum seas schedule depart shanghai decision coordinate disease prevention ensure health safety passenger crew royal caribbean statement msc cancel january departure splendida shanghai schedule sail night guest book cruise option receive full refund cruise ticket port charge book alternative sailing equivalent price receive additional onboard credit embarkation date end year spokesperson msc time writing msc splendida plan remain port duration cruise january february today front back page download newspaper order issue historic daily express newspaper archive"]



# Load the small English model

nlp = spacy.load('en_core_web_sm')

# Process a text

doc = nlp(text_example[0])

# Iterate over the tokens

for token in doc:

    # Print the text and the predicted part-of-speech tag

    print(token.text, token.pos_)
## Predicting Named Entities

# Iterate over the predicted entities

for ent in doc.ents:

    # Print the entity text and its label

    print(ent.text, ent.label_)
## Predicting Syntactic Dependencies

for token in doc:

    print(token.text, token.pos_, token.dep_, token.head.text)
## Document Similarity



# Compare two documents

doc1 = nlp(text_example[0])

doc2 = nlp(text_example2[0])

print(doc1.similarity(doc2))
# Data Preprocessing 

def preprocess(txt):

  '''

  Take text pass through spacy's pipeline 

  Normalize text using remove stopwords from CUSTOM_STOPWORDS, take words which are RELEVENT_POS_TAGS and

  take lemma and use that in smaller version of alphabet

  '''

  doc = nlp(txt)

  rel_tokens = " ".join([tok.lemma_.lower() for tok in doc if tok.pos_ in RELEVANT_POS_TAGS and tok.lemma_.lower() not in CUSTOM_STOPWORDS])

  return rel_tokens
nlp = spacy.load('en_core_web_sm',disable=['parser','ner','tokenizer'])



# from https://www.kaggle.com/jannalipenkova/covid-19-media-overview

RELEVANT_POS_TAGS = ["PROPN", "VERB", "NOUN", "ADJ"]



CUSTOM_STOPWORDS = ["say", "%", "will", "new", "would", "could", "other", 

                    "tell", "see", "make", "-", "go", "come", "can", "do", 

                    "such", "give", "should", "must", "use"]



tqdm.pandas()

processed_content = df["content"].progress_apply(preprocess)

df["processed_content"] = processed_content

df.to_csv("processed_csv", index=False)  # execute the 1° time and save the file.  
# df = pd.read_csv('./processed_csv.csv',index_col='Unnamed: 0')  # more fast, just read the processed file 

# df.head()
reindexed_data = df['processed_content']
# Define helper functions

def get_top_n_words(n_top_words, count_vectorizer, text_data):

    '''

    returns a tuple of the top n words in a sample and their 

    accompanying counts, given a CountVectorizer object and text sample

    '''

    vectorized_headlines = count_vectorizer.fit_transform(text_data.values)

    vectorized_total = np.sum(vectorized_headlines, axis=0)

    word_indices = np.flip(np.argsort(vectorized_total)[0,:], 1)

    word_values = np.flip(np.sort(vectorized_total)[0,:],1)

    

    word_vectors = np.zeros((n_top_words, vectorized_headlines.shape[1]))

    for i in range(n_top_words):

        word_vectors[i,word_indices[0,i]] = 1



    words = [word[0].encode('ascii').decode('utf-8') for 

             word in count_vectorizer.inverse_transform(word_vectors)]



    return (words, word_values[0,:n_top_words].tolist()[0])
count_vectorizer = CountVectorizer(stop_words='english')

words, word_values = get_top_n_words(n_top_words=15,

                                     count_vectorizer=count_vectorizer, 

                                     text_data=reindexed_data)



fig, ax = plt.subplots(figsize=(16,8))

ax.bar(range(len(words)), word_values);

ax.set_xticks(range(len(words)));

ax.set_xticklabels(words, rotation='vertical');

ax.set_title('Top words in the first 15 content articles (excluding stop words)');

ax.set_xlabel('Word');

ax.set_ylabel('Number of occurences');

plt.show()
order_by = df['topic_area'].value_counts().index

sns.catplot(kind='count',x='topic_area',aspect=2,data=df,order=order_by)

plt.show()

# Preparing a corpus for analysis and checking the first 2 entries

corpus=[]

corpus = df['processed_content'].to_list()
corpus = list(set(corpus))

corpus[:2]
print('Corpus lenght, ',len(df['processed_content'].to_list()),' There is '+ str(len(corpus))+' unique content')
# Generating the wordcloud with the values under the category dataframe

corpus_graph_1 = list(set( df['processed_content'].to_list() ))

corpus_graph_2 = list(set( df['processed_content'].to_list() ))

corpus_graph_3 = list(set( df['processed_content'].to_list() ))

corpus_graph_4 = list(set( df['processed_content'].to_list() ))

corpus_graph_5 = list(set( df['processed_content'].to_list() ))
firstcloud = WordCloud(

                          stopwords=STOPWORDS,

                          background_color='black',

                          width=4500,

                          height=2800

                         ).generate(" ".join(corpus_graph_1))

plt.imshow(firstcloud)

plt.axis('off')

plt.show()
seccloud = WordCloud(

                          stopwords=STOPWORDS,

                          background_color='black',

                          width=4500,

                          height=2800

                         ).generate(" ".join(corpus_graph_2))

plt.imshow(seccloud)

plt.axis('off')

plt.show()
tcloud = WordCloud(

                          stopwords=STOPWORDS,

                          background_color='black',

                          width=4500,

                          height=2800

                         ).generate(" ".join(corpus_graph_3))

plt.imshow(tcloud)

plt.axis('off')

plt.show()
fourcloud = WordCloud(

                          stopwords=STOPWORDS,

                          background_color='black',

                          width=4500,

                          height=2800

                         ).generate(" ".join(corpus_graph_4))

plt.imshow(fourcloud)

plt.axis('off')

plt.show()
fivecloud = WordCloud(

                          stopwords=STOPWORDS,

                          background_color='black',

                          width=4500,

                          height=2800

                         ).generate(" ".join(corpus_graph_5))

plt.imshow(fivecloud)

plt.axis('off')

plt.show()
TEMP_FOLDER = tempfile.gettempdir()

print('Folder "{}" will be used to save temporary dictionary and corpus.'.format(TEMP_FOLDER))

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# removing common words and tokenizing

stoplist = stopwords.words('english') + list(punctuation) + list("([)]?") + [")?"]

texts = [[word for word in str(document).lower().split() if word not in stoplist] for document in corpus]

dictionary = corpora.Dictionary(texts)

dictionary.save(os.path.join(TEMP_FOLDER, 'content_file.dict'))  # store the dictionary
corpus = [dictionary.doc2bow(text) for text in texts]

corpora.MmCorpus.serialize(os.path.join(TEMP_FOLDER, 'content_file.mm'), corpus) 
tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model
corpus_tfidf = tfidf[corpus]  # step 2 -- use the model to transform vectors
#I will try 15 topics

total_topics = 15



lda = models.LdaModel(corpus, id2word=dictionary, num_topics=total_topics)

corpus_lda = lda[corpus_tfidf] # create a double wrapper over the original corpus: bow->tf
data_lda = {i: OrderedDict(lda.show_topic(i,25)) for i in range(total_topics)}
df_lda = pd.DataFrame(data_lda)

df_lda = df_lda.fillna(0).T

print(df_lda.shape)
HTML('<iframe width="900" height="687" src="https://www.youtube.com/embed/SF50IK5XgKA" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>')
pyLDAvis.enable_notebook()

panel = pyLDAvis.gensim.prepare(lda, corpus_lda, dictionary, mds='mmds')

panel
pyLDAvis.enable_notebook()

panel = pyLDAvis.gensim.prepare(lda, corpus_lda, dictionary, mds='tsne')

panel
corpus_topic_modeling_2example = df['processed_content'].to_list()

corpus_topic_modeling_2example = list(set(corpus_topic_modeling_2example))
## YOUR CODE HERE



max_features = 500

# Create a CountVectorizer object

tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,

                                max_features=max_features,

                                stop_words='english')

# Fit and transform this object to the processed reviews

tf = tf_vectorizer.fit_transform(corpus_topic_modeling_2example)

print("ready")

n_topics = 15

lda_model = LatentDirichletAllocation(n_components=n_topics, max_iter=5,

                                      learning_method='online',

                                      learning_offset=50.,

                                      random_state=0)

lda_model.fit(tf)

lda_transformed = lda_model.transform(tf)
def make_plot(lda_mat, n_components, alg):

    

    fig = plt.figure(figsize=(10,10), facecolor='white')

    ax = fig.add_subplot(111)

    if alg== 'TSNE':

        tsne = TSNE(n_components=n_components, perplexity=10, init='pca')

        projected = tsne.fit_transform(lda_mat)

    if alg== 'Isomap':    

        # Create instance

        iso = Isomap(n_components=n_components)

        # Fitting

        iso.fit(lda_mat)

        projected = iso.transform(lda_mat)        

    if alg== 'PCA':    

        pca = PCA(n_components=n_components)

        projected = pca.fit_transform(lda_mat)

    if alg== 'FactorAnalysis':    

        projected = FactorAnalysis(n_components = n_components).fit_transform(lda_mat)

    for class_num in np.arange(n_topics):

        topic_inds = np.where(lda_mat[:, class_num] > 0.5)[0]

        ax.scatter(projected[topic_inds, 0],

                   projected[topic_inds, 1], 

                   edgecolor='none', marker='.', alpha=0.7, label=str(class_num))

    plt.title('{} Components'.format(alg))

    ax.set_xlabel('component 1')

    ax.set_ylabel('component 2')

    ax.legend()

make_plot(lda_transformed, 2,'FactorAnalysis')
make_plot(lda_transformed, 2,'PCA')
make_plot(lda_transformed, 2, 'Isomap')
make_plot(lda_transformed, 2, 'TSNE')