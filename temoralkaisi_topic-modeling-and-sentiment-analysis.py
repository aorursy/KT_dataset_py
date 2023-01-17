import pandas as pd
# To change date to datetime
from datetime import datetime
import re
# Gensim
import gensim
from gensim.utils import simple_preprocess
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from sklearn.feature_extraction.text import CountVectorizer
# Spacy for preprocessing
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
nlp = spacy.load('en_core_web_sm')
# To change date to datetime
import nltk
nltk.download('stopwords')
stop=set(stopwords.words('english'))
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)
from collections import defaultdict
# Code Snippet for Named Entity Barchart
import spacy
from collections import  Counter
import seaborn as sns
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# import data
news = pd.read_json('../input/news-category-dataset/News_Category_Dataset_v2.json', lines=True)
news.head(3)

news.rename(columns = {'category':'Section'}, inplace = True) 
# Counting how many headline each section have 
New_section_type = news.groupby('Section').count()['headline'].reset_index()
New_section_type
# Section distribution
ax = news.groupby("Section").count()["headline"].plot(kind="bar", 
                                                 figsize=(8, 5),
                                                 title="Headline based on each section")
plt.show()
news = news[ news["Section"].isin(['ENTERTAINMENT','POLITICS','WELLNESS']) ][["Section","headline"]]
# converting the column news headline to string column
news['headline'] = news['headline'].astype(str)
#showing the number of chracters appear in each news headline
news['headline'].str.len().hist(color='r')
def plot_word_number_histogram(text):
    text.str.split().\
        map(lambda x: len(x)).\
        hist(color='red')

plot_word_number_histogram(news['headline'])
news['headline'].str.split().\
   apply(lambda x : [len(i) for i in x]). \
   map(lambda x: np.mean(x)).hist(color='red')
corpus=[]
new= news['headline'].str.split()
new=new.values.tolist()
corpus=[word for i in new for word in i]
dic=defaultdict(int)
for word in corpus:
    if word in stop:
        dic[word]+=1
def plot_top_stopwords_barchart(text):
    stop=set(stopwords.words('english'))
    
    new= text.str.split()
    new=new.values.tolist()
    corpus=[word for i in new for word in i]
    from collections import defaultdict
    dic=defaultdict(int)
    for word in corpus:
        if word in stop:
            dic[word]+=1
            
    top=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10] 
    x,y=zip(*top)
    plt.bar(x,y)
plot_top_stopwords_barchart(news['headline'])
def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_words(news['headline'], 20)
for word, freq in common_words:
    print(word, freq)
df1 = pd.DataFrame(common_words, columns = ['headline' , 'count'])
df1.groupby('headline').sum()['count'].sort_values(ascending=False).iplot(
kind='bar', yTitle='Count', linecolor='black', title='Top 20 words in news_headline before removing stop words')
def get_top_n_bigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_bigram(news["headline"], 20)
df3 = pd.DataFrame(common_words, columns = ['bigram' , 'count'])

fig = go.Figure([go.Bar(x=df3['bigram'], y=df3['count'])])
fig.update_layout(title=go.layout.Title(text="Top 20 bigrams in the news headline text after removing stop words and lemmatization"))
fig.show()
def get_top_n_trigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(3, 3), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_trigram(news["headline"], 20)
df4 = pd.DataFrame(common_words, columns = ['trigram' , 'count'])

fig = go.Figure([go.Bar(x=df4['trigram'], y=df4['count'])])
fig.update_layout(title=go.layout.Title(text="Top 20 trigrams in the news headline"))
fig.show()
def plot_named_entity_barchart(text):
    nlp = spacy.load("en_core_web_sm")
    
    def _get_ner(text):
        doc=nlp(text)
        return [X.label_ for X in doc.ents]
    
    ent=text.apply(lambda x : _get_ner(x))
    ent=[x for sub in ent for x in sub]
    counter=Counter(ent)
    count=counter.most_common()
    
    x,y=map(list,zip(*count))
    sns.barplot(x=y,y=x)

plot_named_entity_barchart(news['headline'])
def ner(text,ent="PERSON"):
    doc=nlp(text)
    return [X.text for X in doc.ents if X.label_ == ent]

gpe=news['headline'].apply(lambda x: ner(x))
gpe=[i for x in gpe for i in x]
counter=Counter(gpe)

x,y=map(list,zip(*counter.most_common(10)))
sns.barplot(y,x)
def ner(text,ent="ORG"):
    doc=nlp(text)
    return [X.text for X in doc.ents if X.label_ == ent]

gpe=news['headline'].apply(lambda x: ner(x))
gpe=[i for x in gpe for i in x]
counter=Counter(gpe)

x,y=map(list,zip(*counter.most_common(10)))
sns.barplot(y,x)
news.isnull().sum()
# converting the column news headline to string column
news['headline'] = news['headline'].astype(str)
news.shape

# Select duplicate rows except first occurrence based on all columns
duplicateRowsDF = news[news.duplicated()]
print(duplicateRowsDF)
news.drop_duplicates(subset=['headline'], keep='first', inplace=True)
news.shape

news.head(1)
news['headline'] = news['headline'].str.lower()
news.head(1)
# Apply a second round of cleaning
def clean_text_round2(text):
    '''Get rid of some additional punctuation and non-sensical text that was missed the first time around.'''
    text = re.sub('[‘’“”…]', '', text)
    text = re.sub('\n', '', text)
    return text

round2 = lambda x: clean_text_round2(x)
news = pd.DataFrame(news.headline.apply(round2))
news.shape
# Remove the numbers from the news_headline
news['headline'] = news['headline'].apply(lambda x: re.sub(r'[^A-Za-z\s]', '', x))
news
news['length'] = news['headline'].apply(len)
news = news.drop(news[news['length']<3].index)

# Create an object of Vader Sentiment Analyzer
vader_analyzer = SentimentIntensityAnalyzer()
#setting up array to save the scores inside them before adding to the dataframe
negative = []
neutral = []
positive = []
compound = []
#using the vadar funtion to identify the sentiment then add the score for each inside the array .
def sentiment_scores(df, negative, neutral, positive, compound):
    for i in news['headline']:
        sentiment_dict = vader_analyzer.polarity_scores(i)
        negative.append(sentiment_dict['neg'])
        neutral.append(sentiment_dict['neu'])
        positive.append(sentiment_dict['pos'])
        compound.append(sentiment_dict['compound'])

# Function calling 
sentiment_scores(news, negative, neutral, positive, compound)
# Prepare columns to add the scores later to the dataframe 
news["negative"] = negative
news["neutral"] = neutral
news["positive"] = positive
news["compound"] = compound

# Fill the overall sentiment with encoding:
# (-1)Negative, (0)Neutral, (1)Positive
sentiment = []
for i in news['compound']:
    if i >= 0.05 : 
        sentiment.append(1)
  
    elif i <= - 0.05 : 
        sentiment.append(-1) 
        
    else : 
        sentiment.append(0)
news['sentiment'] = sentiment

neg_headline = news.sentiment.value_counts()[-1]
neu_headline = news.sentiment.value_counts()[0]
pos_headline = news.sentiment.value_counts()[1]

# Draw Plot
fig, ax = plt.subplots(figsize=(10, 6), subplot_kw=dict(aspect="equal"), dpi= 80)

data = [news.sentiment.value_counts()[-1], news.sentiment.value_counts()[0], news.sentiment.value_counts()[1]]
categories = ['Negative', 'Neutral', 'Positive']
explode = [0.05,0.05,0.05]

def func(pct, allvals):
    absolute = int(pct/100.*np.sum(allvals))
    return "{:.1f}% ({:d} )".format(pct, absolute)

wedges, texts, autotexts = ax.pie(data, 
                                  autopct=lambda pct: func(pct, data),
                                  textprops=dict(color="w"), 
                                  colors=['#c9658f', '#3c6382', '#cdc0b0'],
                                  startangle=140,
                                  explode=explode)

# Decoration
ax.legend(wedges, categories, title="Sentiment", loc="center left", bbox_to_anchor=(1, 0.2, 0.5, 1))
plt.setp(autotexts, size=10, weight=700)
ax.set_title("Number of headlines by Sentiment", fontsize=12, fontweight="bold")
plt.show()





Positive_headline = news[news["sentiment"]== 1].sort_values(by=["positive"], ascending=False)
negative_headline = news[news["sentiment"]== -1].sort_values(by=["positive"], ascending=False)
import collections
ner = spacy.load("en_core_web_lg")
lst_tag_type = ["ORG","PERSON","NORP","GPE","LOC","FAC","EVENT","PRODUCT","WORK_OF_ART"]
def utils_ner_text(txt, ner=None, lst_tag_filter=None, grams_join="_"):
    ## apply model
    ner = spacy.load("en_core_web_lg") if ner is None else ner
    entities = ner(txt).ents

    ## tag text
    tagged_txt = txt
    for tag in entities:
        if (lst_tag_filter is None) or (tag.label_ in lst_tag_filter):
            try:
                tagged_txt = re.sub(tag.text, grams_join.join(tag.text.split()), tagged_txt) #it breaks with wild characters like *+
            except Exception as e:
                continue

    ## extract tags list
    if lst_tag_filter is None:
        lst_tags = [(tag.text, tag.label_) for tag in entities]  #list(set([(word.text, word.label_) for word in ner(x).ents]))
    else: 
        lst_tags = [(word.text, word.label_) for word in entities if word.label_ in lst_tag_filter]

    return tagged_txt, lst_tags
def add_ner_spacy(dtf, column, ner=None, lst_tag_filter=None, grams_join="_", create_features=True):
    ner = spacy.load("en_core_web_lg") if ner is None else ner

    ## tag text and exctract tags
    print("--- tagging ---")
    dtf[[column+"_tagged", "tags"]] = dtf[[column]].apply(lambda x: utils_ner_text(x[0], ner, lst_tag_filter, grams_join), 
                                                          axis=1, result_type='expand')

    ## put all tags in a column
    print("--- counting tags ---")
    dtf["tags"] = dtf["tags"].apply(lambda x: utils_lst_count(x, top=None))
    
    ## extract features
    if create_features == True:
        print("--- creating features ---")
        ### features set
        tags_set = []
        for lst in dtf["tags"].tolist():
            for dic in lst:
                for k in dic.keys():
                    tags_set.append(k[1])
        tags_set = list(set(tags_set))
        ### create columns
        for feature in tags_set:
            dtf["tags_"+feature] = dtf["tags"].apply(lambda x: utils_ner_features(x, feature))
    return dtf
def utils_lst_count(lst, top=None):
    dic_counter = collections.Counter()
    for x in lst:
        dic_counter[x] += 1
    dic_counter = collections.OrderedDict(sorted(dic_counter.items(), key=lambda x: x[1], reverse=True))
    lst_top = [ {key:value} for key,value in dic_counter.items() ]
    if top is not None:
        lst_top = lst_top[:top]
    return lst_top
def utils_ner_features(lst_dics_tuples, tag):
    if len(lst_dics_tuples) > 0:
        tag_type = []
        for dic_tuples in lst_dics_tuples:
            for tuple in dic_tuples:
                type, n = tuple[1], dic_tuples[tuple]
                tag_type = tag_type + [type]*n
                dic_counter = collections.Counter()
                for x in tag_type:
                    dic_counter[x] += 1
        return dic_counter[tag]   #pd.DataFrame([dic_counter])
    else:
        return 0
# this takes a while
dtf_positive = add_ner_spacy(Positive_headline, "headline", nlp, lst_tag_type, grams_join="_", create_features=True)
dtf_negative = add_ner_spacy(negative_headline, "headline", nlp, lst_tag_type, grams_join="_", create_features=True)

%%time
import gensim
from gensim.utils import simple_preprocess
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations
data_positive = dtf_positive.headline_tagged.values.tolist()
data_negative = dtf_negative.headline_tagged.values.tolist()
data_words_positive = list(sent_to_words(data_positive))
data_words_negative = list(sent_to_words(data_negative))
print(data_words_positive[:3])
# Build the bigram and trigram models
bigram_positive = gensim.models.Phrases(data_words_positive, min_count=5, threshold=100) # higher threshold fewer phrases.
bigram_negative = gensim.models.Phrases(data_words_negative, min_count=5, threshold=100) # higher threshold fewer phrases
trigram_positive = gensim.models.Phrases(bigram_positive[data_words_positive], threshold=100)
trigram_negative = gensim.models.Phrases(bigram_negative[data_words_negative], threshold=100)
# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram_positive)
bigram_mod = gensim.models.phrases.Phraser(bigram_negative)
trigram_mod = gensim.models.phrases.Phraser(trigram_positive)
trigram_mod = gensim.models.phrases.Phraser(trigram_negative)
# NLTK Stop words
# import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]
def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

# Remove Stop Words
data_words_nostops = remove_stopwords(data_words_positive)
# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)
# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
nlp = spacy.load("en_core_web_sm")
# Do lemmatization keeping only noun, adj
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ'])
print(data_lemmatized[:2])

# Remove Stop Words
data_words_nostops = remove_stopwords(data_words_negative)
# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)
# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
nlp = spacy.load("en_core_web_sm")
# Do lemmatization keeping only noun, adj
data_lemmatized_negative = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ'])
print(data_lemmatized_negative[:1])
data_lemmatized
%%time
from gensim import corpora
# Create Dictionary for positive news headline
id2wordn_positive = corpora.Dictionary(data_lemmatized)
# Create Corpus: Term Document Frequency
corpusn_positive = [id2wordn_positive.doc2bow(news) for news in data_lemmatized ]

# Create Dictionary for negative news headline
id2wordn_negative = corpora.Dictionary(data_lemmatized_negative)
# Create Corpus: Term Document Frequency
corpusn_negative = [id2wordn_negative.doc2bow(news) for news in data_lemmatized_negative ]
%%time
from gensim import matutils, models
# Let's start with 4 topics
ldan10 = models.LdaModel(corpus=corpusn_positive, num_topics=10, id2word=id2wordn_positive, passes=10)
ldan10.print_topics()
%%time
from gensim import matutils, models
# Let's start with 4 topics
ldan10 = models.LdaModel(corpus=corpusn_negative, num_topics=10, id2word=id2wordn_negative, passes=10)
ldan10.print_topics()
%%time
# Build LDA model
lda_model_positive = gensim.models.LdaMulticore(corpus=corpusn_positive,
                                       id2word=id2wordn_positive,
                                       num_topics=10, 
                                       random_state=100,
                                       chunksize=100,
                                       passes=10,
                                       per_word_topics=True)
from pprint import pprint
# Print the Keyword in the 10 topics
pprint(lda_model_positive.print_topics())

%%time
# Build LDA model
lda_model_negative = gensim.models.LdaMulticore(corpus=corpusn_negative,
                                       id2word=id2wordn_negative,
                                       num_topics=10, 
                                       random_state=100,
                                       chunksize=100,
                                       passes=10,
                                       per_word_topics=True)
from pprint import pprint
# Print the Keyword in the 10 topics
pprint(lda_model_negative.print_topics())

import matplotlib.pyplot as plt
import math
from matplotlib import gridspec
k =10
def plot_top_words(lda=lda_model_positive , nb_topics=10, nb_words=10):
    top_words = [[word for word,_ in lda.show_topic(topic_id, topn=50)] for topic_id in range(lda.num_topics)]
    top_betas = [[beta for _,beta in lda.show_topic(topic_id, topn=50)] for topic_id in range(lda.num_topics)]

    gs  = gridspec.GridSpec(round(math.sqrt(k))+1,round(math.sqrt(k))+1)
    gs.update(wspace=0.5, hspace=0.5)
    plt.figure(figsize=(20,15))
    for i in range(nb_topics):
        ax = plt.subplot(gs[i])
        plt.barh(range(nb_words), top_betas[i][:nb_words], align='center',color='blue', ecolor='black')
        ax.invert_yaxis()
        ax.set_yticks(range(nb_words))
        ax.set_yticklabels(top_words[i][:nb_words])
        plt.title("Positive_Topic "+str(i))
        
  
plot_top_words()
import matplotlib.pyplot as plt
import math
from matplotlib import gridspec
k =10
def plot_top_words(lda=lda_model_negative , nb_topics=10, nb_words=10):
    top_words = [[word for word,_ in lda.show_topic(topic_id, topn=50)] for topic_id in range(lda.num_topics)]
    top_betas = [[beta for _,beta in lda.show_topic(topic_id, topn=50)] for topic_id in range(lda.num_topics)]

    gs  = gridspec.GridSpec(round(math.sqrt(k))+1,round(math.sqrt(k))+1)
    gs.update(wspace=0.5, hspace=0.5)
    plt.figure(figsize=(20,15))
    for i in range(nb_topics):
        ax = plt.subplot(gs[i])
        plt.barh(range(nb_words), top_betas[i][:nb_words], align='center',color='red', ecolor='black')
        ax.invert_yaxis()
        ax.set_yticks(range(nb_words))
        ax.set_yticklabels(top_words[i][:nb_words])
        plt.title("Negative_Topic "+str(i))
        
  
plot_top_words()
%%time

lda_model_multicore_positive_2 = gensim.models.LdaMulticore(corpus=corpusn_positive, num_topics=10 , id2word=id2wordn_positive, passes=10, alpha=0.1 , eta = 50 /10)    
 
lda_model_multicore_positive_2.show_topics()
import matplotlib.pyplot as plt
import math
from matplotlib import gridspec
k =10
def plot_top_words(lda=lda_model_multicore_positive_2 , nb_topics=10, nb_words=10):
    top_words = [[word for word,_ in lda.show_topic(topic_id, topn=50)] for topic_id in range(lda.num_topics)]
    top_betas = [[beta for _,beta in lda.show_topic(topic_id, topn=50)] for topic_id in range(lda.num_topics)]

    gs  = gridspec.GridSpec(round(math.sqrt(k))+1,round(math.sqrt(k))+1)
    gs.update(wspace=0.5, hspace=0.5)
    plt.figure(figsize=(20,15))
    for i in range(nb_topics):
        ax = plt.subplot(gs[i])
        plt.barh(range(nb_words), top_betas[i][:nb_words], align='center',color='blue', ecolor='black')
        ax.invert_yaxis()
        ax.set_yticks(range(nb_words))
        ax.set_yticklabels(top_words[i][:nb_words])
        plt.title("Positive_Topic "+str(i))
        
  
plot_top_words()
%%time
lda_model_multicore_negativ_2 = gensim.models.LdaMulticore(corpus=corpusn_negative, num_topics=10 , id2word=id2wordn_negative, passes=10, alpha=0.1 , eta = 50 /10)    
 
lda_model_multicore_negativ_2.show_topics()
import matplotlib.pyplot as plt
import math
from matplotlib import gridspec
k =10
def plot_top_words(lda=lda_model_multicore_negativ_2 , nb_topics=10, nb_words=10):
    top_words = [[word for word,_ in lda.show_topic(topic_id, topn=50)] for topic_id in range(lda.num_topics)]
    top_betas = [[beta for _,beta in lda.show_topic(topic_id, topn=50)] for topic_id in range(lda.num_topics)]

    gs  = gridspec.GridSpec(round(math.sqrt(k))+1,round(math.sqrt(k))+1)
    gs.update(wspace=0.5, hspace=0.5)
    plt.figure(figsize=(20,15))
    for i in range(nb_topics):
        ax = plt.subplot(gs[i])
        plt.barh(range(nb_words), top_betas[i][:nb_words], align='center',color='black', ecolor='black')
        ax.invert_yaxis()
        ax.set_yticks(range(nb_words))
        ax.set_yticklabels(top_words[i][:nb_words])
        plt.title("Negative_Topic "+str(i))
        
  
plot_top_words()