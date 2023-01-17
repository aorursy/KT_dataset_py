import re
import csv
import codecs
import numpy as np
import pandas as pd
import operator
import string
import time
import matplotlib.pyplot as plt

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation
from collections import defaultdict
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
nltk.download('stopwords')
nltk.download('punkt')
eng_stopwords = set(stopwords.words("english"))
import sys
import io
data = pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv')
print("Data shape of metadata.csv: ", data.shape)
# remove articles that were published before November 2019
meta_data = data.loc[data["publish_time"] >= "2019-11-01"]
# only keep title, abstracts, doi and url
meta_data = meta_data[["cord_uid", "title", "abstract", "doi", "url"]]
meta_data = meta_data.reset_index(drop=True)
print("New data shape: ", meta_data.shape)
# clean abstract
def clean_abstract(text):

  text = text.lower()
  word_len = len("abstract")
  if text[:word_len] == "abstract":
    text = text[word_len:]
  if "risk factor" in text:
    text = text.replace("risk factor", "riskfactor")
  elif "risk factors" in text:
    text = text.replace("risk factors", "riskfactor")
  return text
#check na
meta_data.isna().sum()
# remove rows with NAs in title and abstract
complete_cases = meta_data.dropna(subset=['title', 'abstract'])
print(complete_cases.isna().sum())
print("Data shape: ", complete_cases.shape)
# clean abstracts
abstracts = complete_cases['abstract']
cleaned_abstract = [clean_abstract(text) for text in abstracts]
complete_cases.abstract = cleaned_abstract
def tokenize(text):
    '''
    Convert the text corpus to lower case, remove all punctuations and numbers which lead to
    a final cleaned corpus with only tokens where all characters in the string are alphabets.
    '''
    # convert the text to lower case and replace all new line characters by an empty string
    lower_text = text.lower().replace('\n', ' ')
    table = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    punct_text = lower_text.translate(table)
    # use NLTK's word tokenization to tokenize the text 
    # remove numbers and empty tokens, only keep tokens with all characters in the string are alphabets
    tokens = [word for word in word_tokenize(punct_text) if word.isalpha()]
    return tokens
def remove_stopwords(word_list, sw=stopwords.words('english')):
    """ 
    Filter out all stop words from the text corpus.
    """
    # It is important to keep words like no and not. Since the meaning of the text will change oppositely
    # if they are removed.
    if 'not' in sw:
        sw.remove('not')
    if 'no' in sw:
        sw.remove('no')
    
    cleaned = []
    for word in word_list:
        if word not in sw:
            cleaned.append(word)
    return cleaned
def stem_words(word_list):
    stemmer = SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in word_list]
    text = " ".join(stemmed_words)
    return stemmed_words, text
def preprocess(text):
    """
    Combine all preprocess steps together.
    Clean each text into tokenized stemmed word list and also return concatenated string
    """
    tokenized = tokenize(text)
    stopword_removed = remove_stopwords(tokenized)
    tokenized_str, cleaned_str = stem_words(stopword_removed)
    return stopword_removed, tokenized_str, cleaned_str
# clean abstracts
abstracts = complete_cases["abstract"].values.tolist()
tokenized_abstracts = []
str_abstracts = []
# create a word dictionary to store all words and their stemmed results
word_dict_abstract = {}
for abstract in abstracts:
  result = preprocess(abstract)
  tokenized = result[0]
  stemmed = result[1]
  for i in range(0,len(stemmed)):
    if stemmed[i] not in word_dict_abstract:
      word_dict_abstract[stemmed[i]] = tokenized[i]
  tokenized_abstracts.append(stemmed)
  str_abstracts.append(result[2])
print("Number of Unique Words in Abstracts:", len(word_dict_abstract))
complete_cases["tokenized_abstract"] = tokenized_abstracts
complete_cases["cleaned_abstracts"] = str_abstracts
# cleaned dataset
complete_cases.head()
import gensim
import itertools
import random
from gensim.models import LdaModel
from gensim import models, corpora, similarities
from scipy.stats import entropy
# 20 topics, on abstracts
def train_model_lda(data):
    
    num_topics = 20
    chunksize = 300
    dictionary = corpora.Dictionary(data['tokenized_abstract'])
    corpus = [dictionary.doc2bow(doc) for doc in data['tokenized_abstract']]

    lda = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary,
                   alpha=1e-3, eta=0.5e-3, chunksize=chunksize, minimum_probability=0.0, passes=2)
    return dictionary, corpus, lda
np.random.seed(2020)
dictionary, corpus, lda = train_model_lda(complete_cases)
for i in range(0,20):
    print("--------Topic: ", i, "--------")
    topic = lda.show_topic(topicid=i, topn=50)
    # use word dictionary to translate stemmed words back to original words
    print([word_dict_abstract[word[0]] for word in topic])
    print()
    print()
# read in task description, the task description was edited in order to cover more keywords
task_description = '''
Smoking, chronic and pre-existing pulmonary disease
Co-infections and co-morbidities
Neonates, pregnant women, transmission during pregnancy 
Age and sex/gender difference, women, men
Socio-economic, psychological, behavioral and environmental factors
Transmission dynamics
Severity of disease, fatality, mortality and high-risk patient groups
Susceptibility
Public health mitigation measures
'''
task_complete_w, tokenized_list, cleaned_desc = preprocess(task_description)
bow = dictionary.doc2bow(tokenized_list)
doc_distribution = np.array([tup[1] for tup in lda.get_document_topics(bow=bow)])
# bar plot of topic distribution of this document
fig, ax = plt.subplots(figsize=(12,6));
# the histogram of the data
patches = ax.bar(np.arange(len(doc_distribution)), doc_distribution)
ax.set_xlabel('Topic ID', fontsize=15)
ax.set_ylabel('Topic Distribution', fontsize=15)
ax.set_title("Topic Distribution of Task Description", fontsize=20)
ax.set_xticks(np.linspace(0,19,20))
fig.tight_layout()
plt.show()
for i in doc_distribution.argsort()[-5:][::-1]:
    print(i, [word_dict_abstract[item[0]] for item in lda.show_topic(topicid=i, topn=50)], "\n")
def assign_article_topics(text):
    """
    Find the top 6 topics that are most relevant to each abstract in data.
    """
    bow = dictionary.doc2bow(text)
    doc_distribution = np.array([tup[1] for tup in lda.get_document_topics(bow=bow)])
    top6_topics = doc_distribution.argsort()[-6:][::-1]
    return list(top6_topics)
top6_topics = []
for abstract in list(complete_cases["tokenized_abstract"]):
    top6_topics.append(assign_article_topics(abstract))
complete_cases['top6_topics'] = top6_topics
complete_cases.head()
# topic distribution of all abstracts in data
from collections import Counter

topics_total = list(itertools.chain.from_iterable(complete_cases['top6_topics']))
count = Counter(topics_total)
df = pd.DataFrame.from_dict(count, orient='index')
ax = df.sort_index().plot(kind='bar', legend=None, title="Topic Distribution Frequency Plot")
ax.set_xlabel("Topic ID")
ax.set_ylabel("Frequency Count")
# Most popular 5 topics in data
sorted_count = {k: v for k, v in sorted(count.items(), key=lambda item: item[1], reverse = True)}
for i in list(sorted_count.keys())[:5]:
    print(i, [word_dict_abstract[item[0]] for item in lda.show_topic(topicid=i, topn=20)], "\n")
target = list(doc_distribution.argsort()[-5:][::-1])
target = sorted(target)
print(target)
match_4 = complete_cases[[len(set(item).intersection(set(target)))>=4 for item in complete_cases['top6_topics'].tolist()]]
match_4_title_abs = match_4.drop(["top6_topics"], axis = 1)

# filter out articles that are not related to COVID-19 or general infectious diseases
cov19_names = ["ncov", "covid-19", "coronavirus", "sars-cov-2"]
related = []
for i in range(0, match_4_title_abs.shape[0]):
  r = any([(keyword in match_4_title_abs["title"].tolist()[i].lower()) or (keyword in match_4_title_abs["abstract"].tolist()[i].lower()) for keyword in cov19_names])
  related.append(r)

# relevant target articles
target_data = match_4_title_abs[related]
# evaluate results
sample = target_data.sample(5, random_state=2)
for i in range(0,sample.shape[0]):
  print("----------Article", i, "----------")
  print("Title: ", "\n", sample['title'].tolist()[i])
  print("Abstracts: ", "\n", sample['abstract'].tolist()[i]) 
  print()
  print()
keyword = ["smok", "preexisting", "pre-existing", "chronic", "underlying", # smoking, pre-existing pulmonary disease
           "co-infection", "coinfection", "co-morbidities", "comorbidities", # co-infections
           "neonat", "pregnancy", "pregnant", "newborn", "uterine", "infant", # neorates and pregnant women
           "socioeconomic", "socio-economic", # socio-economic
           "susceptibility", "environmental", "psychological", "stress", "mental", "frustration", # mental
           "mitigation measures", #government mitigation measures
           "riskfactor", "riskfactors"
]
found = []
complete_cases = complete_cases.reset_index(drop=True)
for i in range(0, complete_cases.shape[0]):
  r = any([(word in complete_cases["title"][i].lower()) or (word in complete_cases["abstract"][i].lower()) for word in keyword])
  found.append(r)
# keyword search results
keyword_search_output = complete_cases[found].reset_index(drop=True)
# filter out articles that are not related to covid-19 from keyword search results
related_2 = []
for i in range(0, keyword_search_output.shape[0]):
  r = any([(keyword in keyword_search_output["title"][i].lower()) or (keyword in keyword_search_output["abstract"][i].lower()) for keyword in cov19_names])
  related_2.append(r)
keyword_found = keyword_search_output[related_2].reset_index(drop=True)

# evaluate results
sample = keyword_found.sample(5, random_state=0)
for i in range(0,sample.shape[0]):
  print("----------Article", i, "----------")
  print("Title: ", "\n", sample['title'].tolist()[i])
  print("Abstracts: ", "\n", sample['abstract'].tolist()[i]) 
  print()
  print()
keyword_search_ids = set(keyword_found['cord_uid'].tolist())
lda_search_ids = set(target_data['cord_uid'].tolist())
all_found = keyword_search_ids.union(lda_search_ids)
final_articles = complete_cases[[uid in list(all_found) for uid in complete_cases['cord_uid']]]
#!pip install numpy==1.16.1
import numpy as np
import pandas as pd
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from keras.models import load_model
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
# clone the skip-thoughts github repository
# !git clone https://github.com/ryankiros/skip-thoughts
# pre-trained models and word embeddings (wikipedia data)
#!mkdir ../input/skipthoughtspackage/skip-thoughts-master/models
#!wget -P ../input/skipthoughtspackage/skip-thoughts-master/models http://www.cs.toronto.edu/~rkiros/models/dictionary.txt
#!wget -P ../input/skipthoughtspackage/skip-thoughts-master/models http://www.cs.toronto.edu/~rkiros/models/utable.npy
#!wget -P ../input/skipthoughtspackage/skip-thoughts-master/models http://www.cs.toronto.edu/~rkiros/models/btable.npy
#!wget -P ../input/skipthoughtspackage/skip-thoughts-master/models http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz
#!wget -P ../input/skipthoughtspackage/skip-thoughts-master/models http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz.pkl
#!wget -P ../input/skipthoughtspackage/skip-thoughts-master/models http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz
#!wget -P ../input/skipthoughtspackage/skip-thoughts-master/models http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz.pkl
# import os
# os.chdir('../../')
# !pwd
# import skipthoughts
def skipthought_encode(text):
    """
    Sentences are encoded into fixed-dimensional vector representation
    """
    
    enc_text = [None]*len(text)
    cum_sum_sentences = [0]
    sent_count = 0
    for txt in text:
        sent_count += len(txt)
        cum_sum_sentences.append(sent_count)
        all_sentences = [sent for txt in text for sent in txt]
    
    print('Loading pre-trained models...')
    model = skipthoughts.load_model()
    encoder = skipthoughts.Encoder(model)
    print('Encoding sentences...')
    enc_sentences = encoder.encode(all_sentences, verbose=False)

    for i in range(len(text)):
        begin = cum_sum_sentences[i]
        end = cum_sum_sentences[i+1]
        enc_text[i] = enc_sentences[begin:end]
    return enc_text
def MakeSummary(texts):
    """
    Produce final summary of each text.
    """
    n_topics = len(texts)
    summary = [None]*n_topics
    print('Starting to encode...')
    # sentence encoding...
    enc_texts= skipthought_encode(texts)
    print('Encoding Finished')
    # perform K-Means clustering
    for i in range(n_topics):
        enc_txt = enc_texts[i]
        # number of clusters
        n_clusters = int(np.ceil(len(enc_txt)**0.5))
        kmeans = KMeans(n_clusters=n_clusters, random_state=1)
        kmeans = kmeans.fit(enc_txt)
        avg = []
        closest = []
        for j in range(n_clusters):
            idx = np.where(kmeans.labels_ == j)[0]
            avg.append(np.mean(idx))
        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_,\
                                                   enc_txt)
        ordering = sorted(range(n_clusters), key=lambda k: avg[k])
        summary[i] = ' '.join([texts[i][closest[idx]] for idx in ordering])
    print('Clustering Finished')

    return summary
all_abstracts = final_articles['abstract'].tolist()
tokenized_abstracts = [sent_tokenize(abstract) for abstract in all_abstracts]
# like 2-gram words, a contiguous sequence of 2 words
# here I created a list of 2-gram sentences in order to preserve sentence completeness

# Chronic diseases, especially cardiovescular diseases were found to be the most popular underlying conditions that high-risk patients have.
# We inferred that it might be a risk factor for 2019-ncov.

twogram_abstracts = []
for abstract in tokenized_abstracts:
    abs_list = []
    if len(abstract) == 1:
        abs_list = abstract
    else:
        for i in range(len(abstract)-1):
            abs_list.append(abstract[i]+abstract[i+1])
        twogram_abstracts.append(abs_list)

#abstract_summary = MakeSummary(twogram_abstracts)
import pickle
with open("/kaggle/input/skipthoughts-results/abstract_summary.pkl", 'rb') as handle:
    abstract_summary = pickle.load(handle)
twogram_abstracts[999]
sent_tokenize(abstract_summary[999])
# build a large sentence corpus
# Dump all the sentences in all abstract summaries into the large list
large_list = []
from nltk.tokenize import sent_tokenize
nltk.download('stopwords')
nltk.download('punkt')

for abstract in abstract_summary:
  large_list += sent_tokenize(abstract)
len(large_list)
import re
import csv
import codecs
import numpy as np
import pandas as pd
import operator
import string
import time
import matplotlib.pyplot as plt

import nltk
nltk.download('stopwords')
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation
from collections import defaultdict
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
eng_stopwords = set(stopwords.words("english"))
import sys
def tokenize(text):
    '''
    Convert the text corpus to lower case, remove all punctuations and numbers which lead to
    a final cleaned corpus with only tokens where all characters in the string are alphabets.
    '''
    # convert the text to lower case and replace all new line characters by an empty string
    lower_text = text.lower().replace('\n', ' ')
    table = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    punct_text = lower_text.translate(table)
    # use NLTK's word tokenization to tokenize the text 
    # remove numbers and empty tokens, only keep tokens with all characters in the string are alphabets
    tokens = [word for word in word_tokenize(punct_text) if word.isalpha()]
    return tokens

    
def remove_stopwords(word_list, sw=stopwords.words('english')):
    """ 
    Filter out all stop words from the text corpus.
    """
    # It is important to keep words like no and not. Since the meaning of the text will change oppositely
    # if they are removed.

    rm_words = ['covid', 'cov', 'sars', 'ncov', 'coronavirus', 'coronaviruses', 'mers', 'corona', 'virus', 'disease', 'diseases', 'viral',
                'jan', 'january', 'feb', 'february', 'march', 'wuhan', 'china', 'hubei', 'december', 'chinese', 'province', 'article', 'protection',
                'copyright', 'abstract', 'background', 'conclusion', 'summary']
    sw += rm_words
    if 'not' in sw:
        sw.remove('not')
    if 'no' in sw:
        sw.remove('no')
    
    cleaned = []
    for word in word_list:
        if word not in sw:
            cleaned.append(word)
    return cleaned

    
def stem_words(word_list):
    stemmer = SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in word_list]
    text = " ".join(stemmed_words)
    return stemmed_words, text

    
def preprocess(text):
    """
    Combine all preprocess steps together.
    Clean each text into tokenized stemmed word list and also return concatenated string
    """
    tokenized = tokenize(text)
    stopword_removed = remove_stopwords(tokenized)
    tokenized_str, cleaned_str = stem_words(stopword_removed)
    return stopword_removed, tokenized_str, cleaned_str
# clean sentences
sentences = large_list
tokenized_sent = []
str_sent = []
word_dict = {}
for sent in sentences:
  result = preprocess(sent)
  tokenized = result[0]
  stemmed = result[1]
  for i in range(0,len(stemmed)):
    if stemmed[i] not in word_dict:
      word_dict[stemmed[i]] = tokenized[i]
  tokenized_sent.append(stemmed)
  str_sent.append(result[2])
new_data = pd.DataFrame(
    {"sentence": large_list,
     "tokenized_sent": tokenized_sent,
     "string_sent": str_sent}
)
new_data.head()
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
tfidf_vectorizer = TfidfVectorizer(min_df=5, norm="l2", sublinear_tf=True, strip_accents='unicode', 
                                                  analyzer='word', token_pattern=r'\w{1,}', stop_words='english', 
                                                  ngram_range=(1,3))
%time tfidf_matrix = tfidf_vectorizer.fit_transform(new_data["string_sent"])
print(tfidf_matrix.shape)
terms_title = tfidf_vectorizer.get_feature_names()
num_clusters = 5
km = KMeans(n_clusters = num_clusters, random_state=1)

%time km.fit(tfidf_matrix)
clusters = km.labels_.tolist()
print("Top Vocabularies of each cluster:")

order_centroids = km.cluster_centers_.argsort()[:, ::-1]

for i in range(num_clusters):
  print("Cluster %d vocabs" % i, end = "\n")

  for ind in order_centroids[i, :50]:
    top_word = ""
    for word in terms_title[ind].split(" "):
      top_word += word_dict[word] + " "

    print(" %s" % top_word.encode("utf-8", "ignore"), end = "\n")
  print("")
  print("")
# remove cluster 3,4
new_data['cluster_id'] = clusters
data_sub = new_data[(new_data['cluster_id'] != 3) & (new_data['cluster_id'] != 4)]
from nltk import FreqDist
import gensim
from gensim.models import LdaModel
from gensim import models, corpora, similarities
from scipy.stats import entropy

def train_model_lda(data):
    
    num_topics = 20
    chunksize = 300
    dictionary = corpora.Dictionary(new_data['tokenized_sent'])
    corpus = [dictionary.doc2bow(doc) for doc in data['tokenized_sent']]

    lda = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary,
                   alpha=0.005, eta=0.001, chunksize=chunksize, minimum_probability=0.0, passes=2)
    return dictionary, corpus, lda
np.random.seed(1)
dictionary, corpus, lda = train_model_lda(data_sub)
for i in range(0,20):
  print("--------Topic: ", i, "--------")
  topic = lda.show_topic(topicid=i, topn=50)
  print([word_dict[word[0]] for word in topic])
  print()
  print()
def assign_topic_importance(text, topic):
  bow = dictionary.doc2bow(text)
  doc_distribution = np.array([tup[1] for tup in lda.get_document_topics(bow=bow)])
  return doc_distribution[topic]
for i in [1,2,18,19]:
  print("--------Topic: ", i, "--------")
  topic = lda.show_topic(topicid=i, topn=50)
  print([word_dict[word[0]] for word in topic])
  print()
  print()
topic_importance = []
for topic in [1,2,18,19]:
    temp = []
    for sent in list(new_data['tokenized_sent']):
        temp.append(assign_topic_importance(sent, topic))
    topic_importance.append(temp)
# calculate topic coverage scores for each sentence
new_data['pregnancy and neonates'] = topic_importance[0]
new_data['comorbidities and pre-existing disease'] = topic_importance[1]
new_data['gender and age'] = topic_importance[2]
new_data['psychological and mitigation measures'] = topic_importance[3]
# clean
filter_out = []
sentences = new_data['sentence'].tolist()
del_keywords = ['http', 'copy', 'right', 'copyrights', 'reserved', 'www']
for sent in sentences:
    if any([word in sent for word in del_keywords]):
        filter_out.append(False)
    else:
        filter_out.append(True)
new_data = new_data[filter_out]
new_data = new_data.drop_duplicates(subset="sentence")
print(new_data.shape)
new_data[['pregnancy and neonates', 
          'comorbidities and pre-existing disease', 
          'gender and age', 
          'psychological and mitigation measures']].describe()
pd.qcut(new_data['pregnancy and neonates'], 20)
# Pregnancy and Neonates
preg_neo = new_data.sort_values(by=['pregnancy and neonates'], ascending=False)
# high importance
preg_neo = preg_neo[preg_neo['pregnancy and neonates']>=0.214]
keyword_check = ['pregnan', 'newborn', 'neonat', 'babies', 'baby', 'birth', 'mother', 'delivery',
                'breastmilk', 'perinatal', 'placental', 'fetal', 'maternal', 'breast', 'uterine']
background_check = ['background', 'aim', 'in this', 'would like to', 'objective', 'introduction', 'this paper',
                   'review', 'emerge', 'cite', 'record', 'setting', 'recent', 'january', 'february', 'march',
                    'method', 'approach', 'unpublished']
preg_check = []
for sent in preg_neo['sentence']:
    if len(sent.split(' ')) < 30:
        preg_check.append(False)
    else:
        key = any([word in sent for word in keyword_check])
        drop = not any([word in sent for word in background_check])
        filtr = key & drop
        preg_check.append(filtr) 

preg_neo = preg_neo[preg_check]
print("Number of sentences found: ", preg_neo.shape[0])
print()
print("Top 30 Answers:")
# show the most relevant 30 sentences
for i in range(30):
    print(preg_neo['sentence'].tolist()[i])
    print(round(preg_neo['pregnancy and neonates'].tolist()[i],5))
    print()
    print()

sw = stopwords.words('english') + ['conclusions', 'conclusion', 'summary', 'data', 'sars', 'cov', 'of', 'and',
                                   'with', 'covid', 'therefore', 'many', 'data', 'also', 'but', 'however']
from wordcloud import WordCloud
# Generate a word cloud image
text =  ' '.join(preg_neo['sentence'].tolist())
#wordcloud = WordCloud(stopwords = stopwords.words('english')).generate(text)

import matplotlib.pyplot as plt

# lower max_font_size
wordcloud = WordCloud(width=800, height=400, background_color='black', stopwords = sw, max_font_size=60, random_state=4).generate(text)
plt.figure(figsize=(30,30), facecolor = 'k')
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
pd.qcut(new_data['comorbidities and pre-existing disease'], 20)
# Comorbidities and Pre-existing Diseases
comorbidity = new_data.sort_values(by=['comorbidities and pre-existing disease'], ascending=False)
# high importance
comorbidity = comorbidity[comorbidity['comorbidities and pre-existing disease']>=0.447]
keyword_check = ['diabete', 'comorbid', 'cancer', 'cardio', 'pulmonary',
                'smok', 'condition', 'chronic', 'underlying', 'hypertension', 'pneumonia',
                 'severe', 'preexisting', 'pre-existing', 'obesity', 'co-infection', 'coinfection',
                'kidney', 'liver', 'severity']
background_check = ['background', 'aim', 'in this', 'would like to', 'objective', 'introduction', 'this paper',
                   'review', 'emerge', 'cite', 'record', 'setting', 'recent', 'january', 'february', 'march',
                    'method', 'approach', 'unpublished', 'mers', 'middle east', 'investigat', 'explor', '2019',
                    '2020']
comorb_check = []
for sent in comorbidity['sentence']:
    if len(sent.split(' ')) < 30:
        comorb_check.append(False)
    else:
        key = any([word in sent for word in keyword_check])
        drop = not any([word in sent for word in background_check])
        filtr = key & drop
        comorb_check.append(filtr) 

comorbidity = comorbidity[comorb_check]
print("Number of sentences found: ", comorbidity.shape[0])
print()
print("Top 30 Answers:")
# show the most relevant 30 sentences
for i in range(30):
    print(comorbidity['sentence'].tolist()[i])
    print(round(comorbidity['comorbidities and pre-existing disease'].tolist()[i],5))
    print()
    print()

sw = stopwords.words('english') + ['conclusions', 'conclusion', 'summary', 'data', 'sars', 'cov', 'of', 'and',
                                   'with', 'covid', 'therefore', 'many', 'data', 'also', 'but', 'however', 'or ci',
                                   'ci and', 'ci to', 'including'
                                  ]

# Generate a word cloud image
text =  ' '.join(comorbidity['sentence'].tolist())

# lower max_font_size
wordcloud = WordCloud(width=800, height=400, background_color='black', stopwords = sw, max_font_size=60, random_state=4).generate(text)
plt.figure(figsize=(30,30), facecolor = 'k')
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
pd.qcut(new_data['gender and age'], 20)
# Gender and Age
gender_age = new_data.sort_values(by=['gender and age'], ascending=False)
# high importance
gender_age = gender_age[gender_age['gender and age']>=0.273]
keyword_check = ['gender', 'male', 'female', 'woman', 'women', 'man', 'men', 'newborn', 'infant', 'child', 
                 'adult', 'old', 'elder', 'young', 'sex', 'advanced age', 'age', 'years old']
background_check = ['background', 'aim', 'in this', 'would like to', 'objective', 'introduction', 'this paper',
                   'review', 'emerge', 'cite', 'record', 'setting', 'recent', 'january', 'february', 'march',
                    'method', 'approach', 'unpublished', 'mers', 'middle east', 'investigat', 'explor', '2019',
                    '2020', 'studies']
agegender_check = []
for sent in gender_age['sentence']:
    if len(sent.split(' ')) < 30:
        agegender_check.append(False)
    else:
        key = any([word in sent for word in keyword_check])
        drop = not any([word in sent for word in background_check])
        filtr = key & drop
        agegender_check.append(filtr) 

gender_age = gender_age[agegender_check]
print("Number of sentences found: ", gender_age.shape[0])
print()
print("Top 30 Answers:")
# show the most relevant 30 sentences
for i in range(30):
    print(gender_age['sentence'].tolist()[i])
    print(round(gender_age['gender and age'].tolist()[i],5))
    print()
    print()
pd.qcut(new_data['psychological and mitigation measures'], 20)
# Psychological and Mitigation Measures
psy_measures = new_data.sort_values(by=['psychological and mitigation measures'], ascending=False)

# high importance
psy_measures = psy_measures[psy_measures['psychological and mitigation measures']>=0.255]
keyword_check = ['mental', 'psycholog', 'environmental', 'anxiety', 'socioeconomic',
                'socio-economic', 'econom', 'measure', 'lockdown', 'quarantine', 'isolation',
                'mitigation', 'distancing', 'emotional', 'regulation', 'order', 'behav']
background_check = ['background', 'aim', 'in this', 'would like to', 'objective', 'introduction', 'this paper',
                   'review', 'emerge', 'cite', 'record', 'setting', 'recent', 'january', 'february', 'march',
                    'method', 'approach', 'capital', 'unpublished', 'mers', 'middle east', 'investigat', 
                    'explor', '2019',
                    '2020', 'studies']
psy_check = []
for sent in psy_measures['sentence']:
    if len(sent.split(' ')) < 30:
        psy_check.append(False)
    else:
        key = any([word in sent for word in keyword_check])
        drop = not any([word in sent for word in background_check])
        filtr = key & drop
        psy_check.append(filtr) 

psy_measures = psy_measures[psy_check]
print("Number of sentences found: ", psy_measures.shape[0])
print()
print("Top 30 Answers:")
# show the most relevant 30 sentences
for i in range(30):    
    print(psy_measures['sentence'].tolist()[i])
    print(round(psy_measures['psychological and mitigation measures'].tolist()[i],5))
    print()
    print()
sw = stopwords.words('english') + ['conclusions', 'conclusion', 'summary', 'data', 'sars', 'cov', 'of', 'and',
                                   'with', 'covid', 'therefore', 'many', 'data', 'also', 'but', 'however', 
                                   'including', 'among', 'due to', 'may', 'could'
                                  ]
# Generate a word cloud image
text =  ' '.join(psy_measures['sentence'].tolist())

# lower max_font_size
wordcloud = WordCloud(width=800, height=400, background_color='black', stopwords = sw, max_font_size=60, random_state=3).generate(text)
plt.figure(figsize=(30,30), facecolor = 'k')
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()