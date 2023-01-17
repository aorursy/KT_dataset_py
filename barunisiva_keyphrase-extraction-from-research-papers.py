#Installing pke

!pip install git+https://github.com/boudinfl/pke.git
import numpy as np 
import pandas as pd 
import os
import seaborn as sns
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
import re
import pke
papers = pd.read_csv('../input/201812_CL_Github.csv')
papers.head()
papers.shape
#Total 106 papers given
extractor = pke.unsupervised.TextRank() 
extractor.load_document(papers['Abstract'][0])
#Keyphrase extraction(top 10) from abstracts using textrank algorithm

def extract_keyphrases(caption, n):
    extractor = pke.unsupervised.TextRank() 
    extractor.load_document(caption)
    extractor.candidate_selection()
    extractor.candidate_weighting()
    keyphrases = extractor.get_n_best(n=n, stemming=False)
    print(keyphrases,"\n")
    return(keyphrases)
    
papers['Text_Rank_Abstract_Keyphrases'] = papers.apply(lambda row: (extract_keyphrases(row['Abstract'],10)),axis=1)
papers.head()
#titles & keyphrases

papers.loc[:,['Title','Text_Rank_Abstract_Keyphrases']]
titles = papers['Title']
titles[1]
count_vectorizer = CountVectorizer()
counts = count_vectorizer.fit_transform(titles)
tfidf_vectorizer = TfidfTransformer().fit(counts)
tfidf_titles = tfidf_vectorizer.transform(counts)

tfidf_titles
#Affinity Propogation
X = tfidf_titles
clustering = AffinityPropagation().fit(X)
clustering 

content_affinity_clusters = list(clustering.labels_)
content_affinity_clusters
papers['title_cluster'] = content_affinity_clusters
#Let's check all papers in cluster 11

papers_cluster11 = papers.loc[papers['title_cluster']==11,['Title','Abstract_Keyphrases']]
papers_cluster11
dict(sorted(papers_cluster11.values.tolist())) 

!git clone https://github.com/csurfer/rake-nltk.git
!python rake-nltk/setup.py install
!pip install rake-nltk
from rake_nltk import Rake
r = Rake() # Uses stopwords for english from NLTK, and all puntuation characters.

def rake_extraction(x):
    r.extract_keywords_from_text(x)
   
    return r.get_ranked_phrases_with_scores() # To get keyword phrases ranked highest to lowest.
papers['rake_extraction']=papers['Abstract'].apply(rake_extraction)
# wordscores = r.calculate_word_scores(papers['Abstract'][0])
papers.head()
papers['Text_Rank_Abstract_Keyphrases'][0]
papers['rake_extraction'][0]
import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 
samp=papers['rake_extraction'][0]
x=[]
y=[]
rake_string=''
for i in samp:
    rake_string+=i[1]

    
stopwords = set(STOPWORDS) 
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(rake_string) 
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.title('RAKE')
plt.show() 
papers.head()
from wordcloud import WordCloud, STOPWORDS 
stopwords = set(STOPWORDS)
tr_res=papers['Text_Rank_Abstract_Keyphrases'][0]
string=''
for i in tr_res:
    string=string+i[0]
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(string) 
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.title('TextRanK ') 
plt.show() 
papers['Text_Rank_Abstract_Keyphrases'][0]
papers["rake_extraction"][0]
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import conlltags2tree, tree2conlltags
from pprint import pprint
def NET(sent):
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)
    pattern = 'NP: {<DT>?<JJ>*<NN>}'
    cp = nltk.RegexpParser(pattern)
    cs = cp.parse(sent)
    iob_tagged = tree2conlltags(cs)
    pprint(iob_tagged)
    #ne_tree = nltk.ne_chunk(pos_tag(word_tokenize(ex)))
    #print(ne_tree)
    return iob_tagged
papers['NET']=papers['Abstract'].apply(NET)
papers.head()
net=papers['NET'][0]
string=''
for i in net:
    string=string+''+i[0]
    
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(string) 
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.title('Named Entity ') 
plt.show() 
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()
doc = nlp('European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices')
pprint([(X.text, X.label_) for X in doc.ents])
pprint([(X, X.ent_iob_, X.ent_type_) for X in doc])

from bs4 import BeautifulSoup
import requests
import re
def url_to_string(url):
    res = requests.get(url)
    html = res.text
    soup = BeautifulSoup(html, 'html5lib')
    for script in soup(["script", "style", 'aside']):
        script.extract()
    return " ".join(re.split(r'[\n\t]+', soup.get_text()))
ny_bb = url_to_string('https://www.nytimes.com/2018/08/13/us/politics/peter-strzok-fired-fbi.html?hp&action=click&pgtype=Homepage&clickSource=story-heading&module=first-column-region&region=top-news&WT.nav=top-news')
article = nlp(ny_bb)
len(article.ents)
labels = [x.label_ for x in article.ents]
Counter(labels)
items = [x.text for x in article.ents]
Counter(items).most_common(3)
sentences = [x for x in article.sents]
print(sentences[20])
displacy.render(nlp(str(sentences[20])), jupyter=True, style='ent')
displacy.render(nlp(str(sentences[20])), style='dep', jupyter = True, options = {'distance': 120})
papers.head()
plot_data=papers['Text_Rank_Abstract_Keyphrases'][0]
x=[]
for i in plot_data:
    x.append(i[0])
y=[]
for i in plot_data:
    y.append(i[1])
x
plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
plt.bar(x,y)
plt.xticks(rotation=85)
plt.show()
plot_data1=papers['rake_extraction'][0]
x1=[]
y1=[]
for i in plot_data1:
    x1.append(i[1])
    y1.append(i[0])
    
plt.figure(num=None, figsize=(10,20), dpi=80, facecolor='w', edgecolor='k')
plt.bar(x1,y1)
plt.xticks(rotation=90)
plt.show()

import matplotlib.gridspec as gridspec

fig = plt.figure(tight_layout=True)
gs = gridspec.GridSpec(2, 2)
ax = fig.add_subplot(gs[0, :])
ax.plot(np.arange(0, 1e6, 1000))
ax.set_ylabel('YLabel0')
ax.set_xlabel('XLabel0')
for i in range(2):
    ax = fig.add_subplot(gs[1, i])
    ax.plot(np.arange(1., 0., -0.1) * 2000., np.arange(1., 0., -0.1))
    ax.set_ylabel('YLabel1 %d' % i)
    ax.set_xlabel('XLabel1 %d' % i)
    if i == 0:
        for tick in ax.get_xticklabels():
            tick.set_rotation(55)
fig.align_labels()  # same as fig.align_xlabels(); fig.align_ylabels()

plt.show()

from matplotlib import pyplot as plt
from datetime import datetime, timedelta

values = range(10)
dates = [datetime.now()-timedelta(days=_) for _ in range(10)]

fig,ax = plt.subplots()
plt.plot(dates, values)
plt.xticks(rotation=45)
plt.grid(True)

plt.show()

