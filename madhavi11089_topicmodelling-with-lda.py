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
data=pd.read_csv('/kaggle/input/consumer-reviews-of-amazon-products/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv')
data.info()
# filtering only required column review.text
amazon_reviews=data[['reviews.text']]

print(amazon_reviews.info())
amazon_reviews.sample(10)
amazon_reviews.sample(10).values
# Doing some data cleaning
reviews_text=amazon_reviews['reviews.text'].values.tolist()
reviews_text[0]
# trying to remove all characters except alphabet letters and space
import re
reviews_text=[re.sub(r'[^A-Za-z\s]','',text) for text in reviews_text]   # extra punctuations,numbers are removed
from pprint import pprint
pprint(len(reviews_text[0]))

print(reviews_text[0:3])
import gensim
from gensim.utils import simple_preprocess
def sent_to_words(texts):
    for text in texts:
        yield(simple_preprocess(str(text),deacc=True))  # deacc=True removes the punctuation marks
        
data_words=list(sent_to_words(reviews_text))
print(data_words[:3])
from nltk.corpus import stopwords

stop_words=stopwords.words('english')
print(stop_words)
# remove stopwords
def remove_stopwords(texts):
    return [[word for word in text if word not in stop_words] for text in texts]

data_words_nostops=remove_stopwords(data_words)
    
print(data_words_nostops[:3])
import gensim
# Applying Bigrams and trigrams 
bigram=gensim.models.Phrases(data_words_nostops,min_count=5,threshold=100)
trigram=gensim.models.Phrases(bigram[data_words_nostops],threshold=100)

bigram_mod=gensim.models.phrases.Phraser(bigram)

def make_bigram(texts):
    return [bigram_mod[doc] for doc in texts]

data_words_nostops_bigrams=make_bigram(data_words_nostops)
print(data_words_nostops_bigrams[:4])
#lemmatize
import spacy
nlp=spacy.load('en',disable=['parser','ner'])

def lemmatize(texts,allowed_postags=['NOUN','ADJ','VERB','ADV']):
    texts_out=[]
    for sent in texts:
        doc=nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        
    return texts_out
data_lemmatized=lemmatize(data_words_nostops_bigrams,allowed_postags=['NOUN','ADJ','VERB','ADV'] )
print(data_lemmatized[:4])
import gensim.corpora as corpora
id2word=corpora.Dictionary(data_lemmatized)
texts=data_lemmatized  # list of list of tokens
corpus=[id2word.doc2bow(text) for text in texts]
print(corpus[:1])   #first document
id2word[0]   # dictionary of int,str
# token word and token id
print([[(id2word[id],freq) for (id,freq) in cp]for cp in corpus[:2]])
# building LDA Model
lda_model=gensim.models.ldamodel.LdaModel(corpus=corpus,id2word=id2word,num_topics=9,random_state=100,update_every=1,
                                         chunksize=100,passes=10,alpha='auto',per_word_topics=True)


pprint(lda_model.print_topics())

doc_lda=lda_model[corpus]

print(list(doc_lda)[0])
print(lda_model[corpus[0]])
print("*****************")
print("The Topics Distribution for first doc: ")
print(lda_model[corpus[0]][0])
#LDA Model Perfromance check
print("Perplexity: ",lda_model.log_perplexity(corpus))

#compute coherence score
from gensim.models import CoherenceModel
coherence_model_lda=CoherenceModel(model=lda_model,texts=data_lemmatized,dictionary=id2word,coherence='c_v')

coherence_lda=coherence_model_lda.get_coherence()
print('Coherence Score: ',coherence_lda)
import pyLDAvis.gensim
import pyLDAvis

import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

pyLDAvis.enable_notebook()
vis=pyLDAvis.gensim.prepare(lda_model,corpus,id2word)
vis
def compute_coherence_values(dictionary,corpus,texts,start,limit,step):
    coherence_vals=[]
    model_list=[]
    
    for num_topics in range(start,limit,step):
        # building LDA Model
        model=gensim.models.ldamodel.LdaModel(corpus=corpus,id2word=dictionary,
                                              num_topics=num_topics,random_state=100,
                                              chunksize=100,passes=10,alpha='auto',per_word_topics=True)

        model_list.append(model)
        
        coherencemodel=CoherenceModel(model=model,texts=texts,dictionary=dictionary,coherence='c_v')
        
        coherence_vals.append(coherencemodel.get_coherence())
    return model_list,coherence_vals

model_list,coherence_vals=compute_coherence_values(dictionary=id2word,
                                                   corpus=corpus,texts=data_lemmatized,
                                                   start=2,limit=20,step=4)
import matplotlib.pyplot as plt

# visualize the optimal LDA Model
limit=20
start=2
step=4
x=range(start,limit,step)

plt.plot(x,coherence_vals)
plt.xlabel('Num_topics')
plt.ylabel('Coherence score')
plt.legend(('coh'),loc='best')
plt.show()
for m, cv in zip(x,coherence_vals):
    print("num topics: ",m,'has coherence value of :',round(cv,4))
optimal_model=model_list[0]  # number of topics is 2
model_topics=optimal_model.show_topics(formatted=False)

pprint(optimal_model.print_topics(num_words=10))
model_topics=optimal_model.show_topics(formatted=False)  
print(model_topics)
model_topics1=optimal_model.show_topics(formatted=True)
print(model_topics1)
pyLDAvis.enable_notebook()
vis1=pyLDAvis.gensim.prepare(optimal_model,corpus,id2word)
vis1
def format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row=row[0]
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant highest weighted topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)

df_sent_topic_keywords=format_topics_sentences(ldamodel=optimal_model,corpus=corpus,texts=data_lemmatized)
df_dominant_topic=df_sent_topic_keywords.reset_index()
df_dominant_topic.columns=['DocumentNo','Dominant_Topic','Perc_Contribution','Topic_Keywords','texts']

df_dominant_topic.sample(10)
print(df_dominant_topic.groupby('Dominant_Topic').count())
# showing best relevant document under each topic
topic_sentences_df =pd.DataFrame()
df_topic_sents_grped=df_dominant_topic.groupby('Dominant_Topic')

for i,grp in df_topic_sents_grped:
    topic_sentences_df=pd.concat([topic_sentences_df,grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)], axis=0)
    
    
#reset index
topic_sentences_df.reset_index(drop=True,inplace=True)

#Format
topic_sentences_df.columns=['Document No','Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]

topic_sentences_df.head()
# Number of Documents for Each Topic
topic_counts = df_sent_topic_keywords['Dominant_Topic'].value_counts()
print(topic_counts)
# Percentage of Documents for Each Topic
topic_contribution = round(topic_counts/topic_counts.sum(), 4)
print(topic_contribution)



# Concatenate Column wise
df_dominant_topics = pd.concat([ topic_counts, topic_contribution], axis=1,)


# Show
df_dominant_topics.reset_index(inplace=True)


# Change Column names
df_dominant_topics.columns = ['Topic id', 'Num_Documents', 'Perc_Documents']

df_dominant_topics