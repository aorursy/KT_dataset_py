# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
!pip install wget
!pip install gensim
!pip install fse
!pip install nltk
!pip install spacy==2.1.0
!pip install fse
from gensim.models import KeyedVectors
import fse # fast sentence embeddings
from fse.models import uSIF
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import TweetTokenizer
import os
import requests
from spacy import util
import spacy
from IPython.core.display import display, HTML
from collections import namedtuple
import json
import tqdm
from gensim.models.fasttext import FastText
import json
def convert_html(jsonfile):    
    html_markup_list=[]
    for er in jsonfile['annotations']:
        title = '<h1>'+er['title']+'</h1>'
        authors = '<h6> '+er['authors']+'</h6>'
        paragraph = '<p>'+er['paragraph'] + '</p>'   
        document_score = '<i>'+str(er['document_score'])+'</i>'
        html_markup_list.append(title+ ' ('+ document_score +')' + '\n'+authors+'\n'+ paragraph)
    return html_markup_list
    
# query = "pre-existing pulmonary disease SARS-Cov2 Hypertension" 
# query = "What is the incubation days of SARS-CoV-2" 
# query = "incubation days coronavirus 2019-nCoV"#  COVID-19
# query = 'socio economic poverty behaviour'
# query = 'what is the comorbidities associated with death'
# query = 'public health mitigation measures that could be effective for control'
# query = 'socio-economic and behavioral factors to understand the economic impact of the SARS-CoV-2 virus and whether there were differences. '
# query = 'what are the risk factors for death in COVID-19'
# query = 'what is the basic reproductive number of SARS-CoV-2 in days'
# query = 'what is the serial interval days SARS-CoV-2'
# query = 'what do we know about the environmental factors influencing SARS-CoV-2'
# query = 'what do we know about medication COVID-19'
query = 'Transmission dynamics of the virus SARS-CoV-2'
# query ='risk of fatality among symptomatic hospitalized patients in SARS-CoV-2'
# query = 'Efforts targeted at a universal coronavirus vaccine'

query_result = requests.get("http://covido-api.volitionlabs.xyz:5200/covido_predictor?query_sentence="+query)
display(HTML('<hr>\n'.join(convert_html(query_result.json()))))
tknzr = TweetTokenizer()
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer() 


#remove stop words and lemmatize the query terms
def clean_my_text(full_text):
    stopset = set(stopwords.words('english')) #| set(string.punctuation)
    tokens = tknzr.tokenize(full_text)
    cleanup = [lemmatizer.lemmatize(token) for token in tokens if token not in stopset and len(token) > 3]
    return cleanup

#the highlighter for the query terms found in the text
def term_highlighter(text: str = None, query: str = None) -> str:
    color_poll=['#61F96C', '#F98A61', '#F3F967','#61DDF9', '#6F8AFA',
                '#F77FBB', '#69F5E9', '#CBD5D4','#21C758', '#5491BC',
               '#EEAE4D', '#D14DEE', '#EE4D4D', '#5E8A86', '#218F1B',
                '#CA5522', '#6458E0','#84B50B','#B50BA0','#0BB5A8']
    
    if not text or not query:
        raise ValueError('Either the no result was found for the query, or the query is empty.')
    
    terms=clean_my_text(query)
    colorindex=0
    
    for term in set(terms):
        if type(term) != str:
            continue
        if term.lower() in text.replace("'", "\'").lower():
            text = re.sub(r"(?i)\b%s" % term , '<span style="background-color:'+color_poll[colorindex]+'">' + term + '</span>', text)
            colorindex=colorindex+1
    return text
from gensim.summarization.summarizer import summarize
import requests
from nltk import wordpunct_tokenize
import re

#submit the query to the API
query = 'How did covid-19 influence global economy, job market and affect small businesses?'
url_ = 'http://covido-api.volitionlabs.xyz:8157/solr/covid_tweetdata/select?q=tweet%3A%20'+query+'&rows=100&wt=json'

r= requests.get(url_)
query_result = r.json()
tweet_result =[]
tweet_ids = []

for each_result in query_result['response']['docs']:
    tweet_result.append(each_result['tweet'])
    tweet_ids.append(each_result['tweetID'])
#for privacy reasons, we cannot display all the original tweets. The front-end will display the link to the tweet. 
#And for the notebook output, we summarize the tweet outputs
summary_result = summarize(' '.join(tweet_result), ratio=0.1)
new_text = term_highlighter(summary_result, query)
#visualise the output
from IPython.core.display import display, HTML
display(HTML(new_text))
Item = namedtuple('INFO', ('title', 'section', 'pdate', 'weburl', 'apiurl', 'text', 'ents', 'sentiment'))
Result = namedtuple('Result', ('sid', 'aid', 'pid', 'sent', 'para'))

def process_sent(sent, remove_punct=True):
    if remove_punct:
        tokens = [token.lower() for token in nltk.word_tokenize(sent.strip()) if len(token)>1 or token.isalnum()]
    else:
        tokens = [token.lower() for token in nltk.word_tokenize(sent.strip())]
    return tokens
    
def flat_corpus(dataset, para_level=False, remove_punct=True):
    corpus = []
    sid2art = {}
    for aid, an in enumerate(tqdm.tqdm(dataset)):
        title = an.title
        paragraphs = [title] + an.text.split("\n")
        if para_level:
            for pid, para in enumerate(paragraphs):
                start_idx = len(sid2art)
                sid2art[start_idx] = {'aid':aid, 'pid':pid}
                tokens = process_sent(para, remove_punct)
                corpus.append(tokens)           
        else:
            for pid, para in enumerate(paragraphs):
                sents = nltk.sent_tokenize(para)
                start_idx = len(sid2art)
                for i in range(len(sents)):
                    sid2art[start_idx+i] = {'aid':aid, 'pid':pid}
                    tokens = process_sent(sents[i], remove_punct)
                    corpus.append(tokens)
    return corpus, sid2art






# load NewsMedia dataset
with open('/kaggle/input/news-socio-economicdata/news-text-unique.json', 'r') as f:
    data = json.load(f)
    analysis = [Item._make(d) for d in data]
# generate tokenized corpus for fse model to use
line_corpus, sid2aid = flat_corpus(analysis, para_level=False, remove_punct=True)
# load fasttext model pre-trained on news-media corpus
ft_model = FastText.load('/kaggle/input/newsfasttext/news-fast.model')
from fse.models import SIF
from fse import IndexedList
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
import numpy as np
def train_fse(fasttext_model, flat_corpus):
    # train sentence embedding model
    model = SIF(fasttext_model)
    indexed_corpus = IndexedList(flat_corpus)
    model.train(indexed_corpus)
    return model, indexed_corpus

def retrieve_paragraph(mappings, dataset):
    # from sentence get its paragragh
    retrieved_paras = []
    for m in mappings:
        article = dataset[m['aid']]
        paras = [article.title] + article.text.split('\n')
        paragraph = paras[m['pid']]
        retrieved_paras.append(paragraph)
    return retrieved_paras

def train_tfidf_vectorizer(dataset):
    # train tfidf vectorizer from paragrapgh context
    tfidf_vectorizer = TfidfVectorizer(binary=True, sublinear_tf=True, norm='l2', ngram_range=(1,2))
    para_corpus, pid2aid = flat_corpus(dataset, para_level=True, remove_punct=True)
    tfidf_vectorizer.fit([' '.join(line) for line in para_corpus])
    return tfidf_vectorizer
    
def rerank(query, results, vectorizer):
    # rerank the retrieved sentences by incoporating paragrapgh context
    trans_para = vectorizer.transform([query]+[' '.join(process_sent(r.para)) for r in results])
    dense = trans_para.todense()
    sims = metrics.pairwise.cosine_similarity(dense[0], dense[1:])
    sorted_rank = np.argsort(-sims[0]).tolist()
    return [results[sr] for sr in sorted_rank]
    
def search(model, vectorizer, query, sent_corpus, dataset, sid_mapping):
    # search best sentences/articles to addresss the query
    query = ' '.join(process_sent(query, remove_punct=True))
    sents = model.sv.similar_by_sentence(nltk.word_tokenize(query), model=model)
    sids = [s[0] for s in sents]
    mapping = [sid_mapping[sid] for sid in sids]
    paragraphs = retrieve_paragraph(mapping, dataset)
    
    results = []
    exist = set()
    for i in range(len(sids)):
        sid = sids[i]
        aid = mapping[i]['aid']
        pid = mapping[i]['pid']
        sent = sent_corpus[i]
        para = paragraphs[i]
        if aid not in exist:
            results.append(Result(sid=sid, aid=aid, pid=pid, sent=sent, para=para))
            exist.add(aid)
    return rerank(query, results, vectorizer)
# build fse model and tfidf vectorizer for document retrieval
fse_model, indexed_corpus = train_fse(ft_model, line_corpus)
tfidf_vectorizer = train_tfidf_vectorizer(dataset=analysis)
# retrieve relevant sentence, paragraph and articles by a query
query = "human rights coronavirus"
search_results = search(model=fse_model, vectorizer=tfidf_vectorizer, 
                               query=query, sid_mapping=sid2aid,
                               sent_corpus=line_corpus, dataset=analysis)
# show the example search results
search_results[:2]
# show the titles of retrieved articles
news_titles = [analysis[r.aid].title for r in search_results]
print(f"query: {query}\n")
print(f"retrieved articles' titles:\n")
for i,t in enumerate(news_titles):
    print(i,t)
# show the examples of retrieved articles from datasetnews_
news_articles = [analysis[r.aid] for r in search_results]
print(news_articles[:2])
