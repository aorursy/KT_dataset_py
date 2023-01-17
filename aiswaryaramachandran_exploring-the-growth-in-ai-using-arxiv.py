import pandas as pd 

import numpy as np 

from datetime import datetime

import sys

import ast



import plotly_express as px



import nltk

from nltk.corpus import stopwords

import spacy



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

from sklearn.decomposition import TruncatedSVD

from sklearn.preprocessing import Normalizer

from sklearn.pipeline import make_pipeline

from sklearn.metrics import pairwise_distances

from sklearn.metrics.pairwise import cosine_similarity





import networkx

from networkx.algorithms.components.connected import connected_components



import json

import dask.bag as db





import utils
ai_category_list=['stat.ML','cs.LG','cs.AI']

records=db.read_text("/kaggle/input/arxiv/*.json").map(lambda x:json.loads(x))

ai_docs = (records.filter(lambda x:any(ele in x['categories'] for ele in ai_category_list)==True))

get_metadata = lambda x: {'id': x['id'],

                  'title': x['title'],

                  'category':x['categories'],

                  'abstract':x['abstract'],

                 'version':x['versions'][-1]['created'],

                         'doi':x["doi"],

                         'authors_parsed':x['authors_parsed']}



data=ai_docs.map(get_metadata).to_dataframe().compute()



data.to_excel("AI_ML_ArXiv_Papers.xlsx",index=False,encoding="utf-8")
print("Number of Papers Related to AI and ML is ",data.shape[0])
data.head()
data['DateTime']=pd.to_datetime(data['version'])

data.head()
data=utils.extractDateFeatures(data,"DateTime")

data.head()
data['num_authors']=data['authors_parsed'].apply(lambda x:len(x))
data['authors']=data['authors_parsed'].apply(lambda authors:[(" ".join(author)).strip() for author in authors])

data.head()
print("Number of Papers with No DOI ",data[pd.isnull(data['doi'])].shape[0])


papers_over_years=data.groupby(['Year']).size().reset_index().rename(columns={0:'Number Of Papers Published'})

px.line(x="Year",y="Number Of Papers Published",data_frame=papers_over_years,title="Growth of AI ML over the Years")
papers_published_over_days=data.groupby(['Date']).size().reset_index().rename(columns={0:'Papers Published By Date'})

px.line(x="Date",y="Papers Published By Date",data_frame=papers_published_over_days,title="Average Papers Published Over Each Day")
ai_authors=pd.DataFrame(utils.flattenList(data['authors'].tolist())).rename(columns={0:'authors'})

papers_by_authors=ai_authors.groupby(['authors']).size().reset_index().rename(columns={0:'Number of Papers Published'}).sort_values("Number of Papers Published",ascending=False).head(20)

px.bar(x="Number of Papers Published",y="authors",data_frame=papers_by_authors.sort_values("Number of Papers Published",ascending=True),title="Top 20 Popular Authors",orientation="h")
data['is_bengio_author']=data['authors'].apply(lambda x:1 if "Bengio Yoshua" in x else 0)

bengio_papers=data[data['is_bengio_author']==1]

bengio_papers=bengio_papers.reset_index(drop=True)



print("Number of Papers by Bengio Yoshua on Arxiv is ",bengio_papers.shape[0])
print("Bengio Yoshua Published His First Paper in ",min(bengio_papers['Date']))

print("Bengio Yoshua Published His Recent Paper in ",max(bengio_papers['Date']))

bengio_papers_by_year=bengio_papers.groupby(['Year']).size().reset_index().rename(columns={0:'Number of Papers Published'})



px.bar(x="Year",y="Number of Papers Published",title="Papers by Bengio Yoshua Over Years",data_frame=bengio_papers_by_year)
print("Average Papers Published in a Year By Bengio Yoshua ",np.median(bengio_papers_by_year['Number of Papers Published']))
titles=bengio_papers['title'].tolist()

stop_words = set(stopwords.words('english')) 

titles=[title.lower() for title in titles] ### Lower Casing the Title

titles=[utils.removeStopWords(title,stop_words) for title in titles]


bigrams_list=[" ".join(utils.generateNGram(title,2)) for title in titles]

topn=50

top_bigrams=utils.getMostCommon(bigrams_list,topn=topn)

top_bigrams_df=pd.DataFrame()

top_bigrams_df['words']=[val[0] for val in top_bigrams]

top_bigrams_df['Frequency']=[val[1] for val in top_bigrams]

px.bar(data_frame=top_bigrams_df.sort_values("Frequency",ascending=True),x="Frequency",y="words",orientation="h",title="Top "+str(topn)+" Bigrams in Papers by Bengio Yoshua")
trigrams_list=[" ".join(utils.generateNGram(title.replace(":",""),3)) for title in titles]

topn=50

top_trigrams=utils.getMostCommon(trigrams_list,topn=topn)

top_trigrams_df=pd.DataFrame()

top_trigrams_df['words']=[val[0] for val in top_trigrams]

top_trigrams_df['Frequency']=[val[1] for val in top_trigrams]

top_trigrams_df=top_trigrams_df[top_trigrams_df["words"]!=""]

px.bar(data_frame=top_trigrams_df.sort_values("Frequency",ascending=True),x="Frequency",y="words",orientation="h",title="Top "+str(topn)+" Trigrams in Papers by Bengio Yoshua")
from pprint import pprint



# Gensim

import gensim

import gensim.corpora as corpora

from gensim.utils import simple_preprocess

from gensim.models import CoherenceModel



# spacy for lemmatization

import spacy



# Plotting tools

import pyLDAvis

import pyLDAvis.gensim 
'''

The tokenise function will lowercase, and tokenise the sentences

'''



def tokenise(sentences):

    return [gensim.utils.simple_preprocess(sentence, deacc=True,max_len=50) for sentence in sentences]

tokenised_sentences=tokenise(bengio_papers['title'].tolist())

tokenised_sentences[0]
nlp = spacy.load('en')
def lemmatise(sentence,stop_words,allowed_postags=None):

    doc=nlp(sentence)

    #print(sentence)

    if allowed_postags!=None:

        tokens = [token.lemma_ for token in doc if (token.pos_ in allowed_postags) and (token.text not in stop_words)]

    if allowed_postags==None:

        tokens= [token.lemma_ for token in doc if (token.text not in stop_words)]

    return tokens
stop_words = spacy.lang.en.stop_words.STOP_WORDS
sentences=[" ".join(tokenised_sentence) for tokenised_sentence in tokenised_sentences]

lemmatised_sentences=[lemmatise(sentence,stop_words) for sentence in sentences]

lemmatised_sentences[0]
# Build the bigram and trigram models

bigram = gensim.models.Phrases(lemmatised_sentences,min_count=2) 

trigram = gensim.models.Phrases(bigram[lemmatised_sentences],min_count=2)  



bigram_mod = gensim.models.phrases.Phraser(bigram)

trigram_mod = gensim.models.phrases.Phraser(trigram)
bigrams_words=[bigram_mod[sentence] for sentence in lemmatised_sentences]



trigrams_words=[trigram_mod[sentence] for sentence in bigrams_words]

id2word = corpora.Dictionary(trigrams_words)

corpus = [id2word.doc2bow(text) for text in trigrams_words]

[(id2word[id], freq) for id, freq in corpus[0]] 
def compute_coherence_values(id2word, corpus, texts, limit, start=2, step=3):

    """

    Compute c_v coherence for various number of topics



    Parameters:

    ----------

    dictionary : Gensim dictionary

    corpus : Gensim corpus

    texts : List of input texts

    limit : Max num of topics



    Returns:

    -------

    model_list : List of LDA topic models

    coherence_values : Coherence values corresponding to the LDA model with respective number of topics

    """

    coherence_values = []

    model_list = []

    for num_topics in range(start, limit, step):

        model = gensim.models.ldamodel.LdaModel(corpus=corpus,

                                           id2word=id2word,

                                           num_topics=num_topics, 

                                           random_state=100,

                                           update_every=1,

                                           chunksize=20,

                                           passes=10,

                                           alpha='auto',

                                           per_word_topics=True)

        model_list.append(model)

        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=id2word, coherence='c_v')

        coherence_values.append(coherencemodel.get_coherence())



    return model_list, coherence_values
models,coherence=compute_coherence_values(id2word,corpus,trigrams_words,limit=20,start=2,step=2)

x = range(2, 20, 2)

plt.plot(x, coherence)

plt.xlabel("Num Topics")

plt.ylabel("Coherence score")

plt.legend(("coherence_values"), loc='best')

plt.show()
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,

                                           id2word=id2word,

                                           num_topics=6, 

                                           random_state=100,

                                           update_every=1,

                                           chunksize=20,

                                           passes=10,

                                           alpha='auto',

                                           per_word_topics=True)
pprint(lda_model.print_topics())

doc_lda = lda_model[corpus]
#pyLDAvis.enable_notebook()

#vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)

#vis
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.



# Compute Coherence Score

coherence_model_lda = CoherenceModel(model=lda_model, texts=trigrams_words, dictionary=id2word, coherence='c_v')

coherence_lda = coherence_model_lda.get_coherence()

print('\nCoherence Score: ', coherence_lda)
def format_topics_sentences(texts,ldamodel=lda_model, corpus=corpus):

    # Init output

    sent_topics_df = pd.DataFrame()



    # Get main topic in each document

    for i, row in enumerate(ldamodel[corpus]):

        row = sorted(row[0], key=lambda x: (x[1]), reverse=True)

        # Get the Dominant topic, Perc Contribution and Keywords for each document

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





df_topic_sents_keywords = format_topics_sentences(bengio_papers['title'].tolist(),ldamodel=lda_model, corpus=corpus)



# Format

df_dominant_topic = df_topic_sents_keywords.reset_index()

df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']



# Show

df_dominant_topic.head(10)
topic_counts=df_dominant_topic['Dominant_Topic'].value_counts().reset_index().rename(columns={'index':'Topic','Dominant_Topic':'Number of Documents'})

topic_counts['percentage_contribution']=(topic_counts['Number of Documents']/topic_counts['Number of Documents'].sum())*100

topic_counts
# Get topic weights and dominant topics ------------

from sklearn.manifold import TSNE





# Get topic weights

topic_weights = []

for i, row_list in enumerate(lda_model[corpus]):

    topic_weights.append([w for i, w in row_list[0]])



# Array of topic weights    

arr = pd.DataFrame(topic_weights).fillna(0).values





# Dominant topic number in each doc

topic_num = np.argmax(arr, axis=1)



# tSNE Dimension Reduction

tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')

tsne_lda = tsne_model.fit_transform(arr)



sent_topics_df=pd.DataFrame()

sent_topics_df['Text']=bengio_papers['title'].tolist()

sent_topics_df['tsne_x']=tsne_lda[:,0]

sent_topics_df['tsne_y']=tsne_lda[:,1]

sent_topics_df['Topic_No']=topic_num

sent_topics_df=pd.merge(sent_topics_df,df_dominant_topic,on="Text")

sent_topics_df.head()
px.scatter(x='tsne_x',y='tsne_y',data_frame=sent_topics_df,color="Topic_No",hover_data=["Topic_Perc_Contrib"])
bengio_papers=pd.merge(bengio_papers,df_dominant_topic.rename(columns={'Text':'title'}),on='title')



num_topics=bengio_papers['Dominant_Topic'].nunique()

authors_df_list=[]



for topic_no in range(num_topics):

    



    temp=bengio_papers[bengio_papers['Dominant_Topic']==topic_no]

    authors=pd.DataFrame(utils.flattenList(temp['authors'].tolist())).rename(columns={0:'authors'})

    authors=authors[authors['authors']!="Bengio Yoshua"]

    papers_authors=authors.groupby(['authors']).size().reset_index().rename(columns={0:'Number of Papers Published'}).sort_values("Number of Papers Published",ascending=False).head(10)

    papers_authors['Topic No']=topic_no

    authors_df_list.append(papers_authors)



co_occurring_authors=pd.concat(authors_df_list)

from plotly.subplots import make_subplots

import plotly.graph_objects as go
fig = make_subplots(rows=3, cols=2)

row=1

col=1

for topic_no in range(num_topics):

    

    wp = lda_model.show_topic(topic_no)

    topic_keywords = ", ".join([word for word, prop in wp])

    temp=co_occurring_authors.loc[co_occurring_authors['Topic No']==topic_no].sort_values("Number of Papers Published",ascending=True)



    fig.add_trace(

    go.Bar(

        x=temp['Number of Papers Published'],

        y=temp['authors'],

        orientation='h',

        name="Topic "+str(topic_no)

        #mode="markers+text",

        #text=["Text A", "Text B", "Text C"],

        #textposition="bottom center"

    ),

    row=row, col=col)

    if col%2==0:

        row=row+1

        col=1

    else:

        col=col+1

fig.update_layout(height=1000, width=1200, title_text="Top 10 Authors With Whom Bengio Worked Across Different Topics")



fig.show()


