import numpy as np, pandas as pd, os, json, glob, re, nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image
import spacy
from spacy import displacy
from textblob.sentiments import NaiveBayesAnalyzer

%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from textblob import TextBlob

import scattertext as st
import re, io
from pprint import pprint

from scipy.stats import rankdata, hmean, norm
import os, pkgutil, json, urllib
from urllib.request import urlopen
from IPython.display import IFrame
from IPython.core.display import display, HTML
from scattertext import CorpusFromPandas, produce_scattertext_explorer
display(HTML("&lt;style>.container { width:98% !important; }&lt;/style>"))
covid_19 = pd.read_csv("cleanCORD.csv")
covid_19.head()
corpus = []
stopwords = nltk.corpus.stopwords.words('english')

for i in tqdm(range(len(covid_19["Body_text"]))):
    body_text = re.sub('[^a-zA-Z]', ' ', covid_19['Body_text'][i])
    body_text = body_text.lower()
    body_text = body_text.split()
    body_text = [word for word in body_text if not word in set(stopwords)]
    body_text = ' '.join(body_text)
    corpus.append(body_text)

text = ""

for i in tqdm(corpus):
    text += i

wordcloud = WordCloud(max_words=100, background_color="white",contour_color='firebrick').generate(text)

plt.figure(figsize=(14,8))
plt.imshow(wordcloud, interpolation='nearest')
plt.axis("off")
plt.show()
plt.savefig("wc1.png", dpi=900)
blob = TextBlob(text)
sentiment = blob.sentiment.polarity
print(sentiment)
def sentiment_calc(text):
    try:
        return TextBlob(text).sentiment
    except:
        return None

covid_19['sentiment'] = covid_19['Body_text'].apply(sentiment_calc)
covid_19[['polarity', 'subjectivity']] = pd.DataFrame(covid_19['sentiment'].tolist(), index=covid_19.index) 
covid_19.sort_values(by=['polarity'], inplace=True, ascending=False)
covid_19.sort_values(by=['subjectivity'], inplace=True, ascending=False)
covid  = covid_19[:10]
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")
f, ax = plt.subplots(figsize=(15, 8))
fig = sns.barplot(x="Paper_id", y= "polarity", data=covid,
            label="Total", color="b")
fig.set_xticklabels(fig.get_xticklabels(),rotation=90)
fig1 = sns.barplot(x="Paper_id", y= "subjectivity", data=covid,
            label="Total", color="b")
fig1.set_xticklabels(fig1.get_xticklabels(),rotation=90) 
covid1  = covid_19[:50]
nlp = spacy.load('en_core_web_sm')
covid1['parsed'] = covid1.Body_text.apply(nlp)
print("Document Count")
print(covid1.groupby('Paper_id')['Body_text'].count(), "\t", )
print("Word Count")
covid1.groupby('Paper_id').apply(lambda x: x.Body_text.apply(lambda x: len(x.split())).sum())
corpus = st.CorpusFromParsedDocuments(covid1, category_col='Paper_id', parsed_col='parsed').build()
html = produce_scattertext_explorer(corpus,
                                    category='3e40cb7055fe5ef148eced8994ff059a0e98aef1',
                                    category_name='',
                                    not_category_name='',
                                    width_in_pixels=1000,
                                    minimum_term_frequency=5,
                                    transform=st.Scalers.scale,
                                    metadata=covid1['Paper_id'])
file_name = 'Covid2.html'
open(file_name, 'wb').write(html.encode('utf-8'))
IFrame(src=file_name, width = 1200, height=700)
html = st.produce_scattertext_explorer(corpus,
                                       category='3e40cb7055fe5ef148eced8994ff059a0e98aef1',
#                                        category_name='Democratic',
#                                        not_category_name='Republican',
                                       minimum_term_frequency=5,
                                       width_in_pixels=1000,
                                       transform=st.Scalers.log_scale_standardize)
file_name = 'LogCovid.html'
open(file_name, 'wb').write(html.encode('utf-8'))
IFrame(src=file_name, width = 1200, height=700)
corpus = st.CorpusFromPandas(covid, category_col='Paper_id', text_col='Body_text',nlp=nlp).build()
print(list(corpus.get_scaled_f_scores_vs_background().index[:20]))
feat_builder = st.FeatsFromOnlyEmpath()
empath_corpus = st.CorpusFromParsedDocuments(covid1,category_col='Paper_id',feats_from_spacy_doc=feat_builder,parsed_col='Body_text').build()
html = st.produce_scattertext_explorer(empath_corpus,category='8a6809df45d5f80a822d68d3c305f7640e10234a',
                                       category_name='e18773ecee762195cee18f1b3d83ef02f2db0dc9',
                                       not_category_name='e18773ecee762195cee18f1b3d83ef02f2db0dc9',
                                       width_in_pixels=1000,metadata=covid['Body_text'],use_non_text_features=True,
                                       use_full_doc=True,topic_model_term_lists=feat_builder.get_top_model_term_lists())
open("Convention-Visualization-Empath.html", 'wb').write(html.encode('utf-8'))
IFrame(src="Convention-Visualization-Empath.html", width = 1200, height=700)
html = st.produce_scattertext_explorer(empath_corpus,category='3e40cb7055fe5ef148eced8994ff059a0e98aef1',
                                       category_name='3e40cb7055fe5ef148eced8994ff059a0e98aef1',
                                       not_category_name='3e40cb7055fe5ef148eced8994ff059a0e98aef1',
                                       width_in_pixels=1000,metadata=covid1['Body_text'],use_non_text_features=True,
                                       use_full_doc=True,topic_model_term_lists=feat_builder.get_top_model_term_lists())
open("Convention-Visualization-Empath.html", 'wb').write(html.encode('utf-8'))
IFrame(src="Convention-Visualization-Empath.html", width = 1200, height=700)
import string

def clean_text(article):
    clean1 = re.sub(r'['+string.punctuation + '’—”'+']', "", article.lower())
    return re.sub(r'\W+', ' ', clean1)
covid_19['tokenized'] = covid_19['Body_text'].map(lambda x: clean_text(x))

covid_19['num_wds'] = covid_19['tokenized'].apply(lambda x: len(x.split()))
covid_19['num_wds'].mean()
def find_cc_wds(content, cc_wds=['covid19','vaccine', 'quarantine', 'corona virus'
                                 'pandemic', 'social distancing','SARS','MERS','genome','China','Wuhan','economy']
):
    found = False
    for w in cc_wds:
        if w in content:
            found = True
            break

    if not found:
        disj = re.compile(r'(vaccine\w+\W+(?:\w+\W+){1,5}?developed) | (coronavirus\W+(?:\w+\W+){1,5}?deaths)')
        if disj.match(content):
            found = True
    return found
covid_19['cc_wds'] = covid_19['tokenized'].apply(find_cc_wds)
valid_papers = covid_19[covid_19['cc_wds']==True]
valid_papers.reset_index(inplace = True, drop = True) 
valid_papers.head()
corpus = []
stopwords = nltk.corpus.stopwords.words('english')

for i in tqdm(range(len(valid_papers["Body_text"]))):
    body_text = re.sub('[^a-zA-Z]', ' ', valid_papers['Body_text'][i])
    body_text = body_text.lower()
    body_text = body_text.split()
    body_text = [word for word in body_text if not word in set(stopwords)]
    body_text = ' '.join(body_text)
    corpus.append(body_text)

text = ""

for i in tqdm(corpus):
    text += i

wordcloud = WordCloud(max_words=100, background_color="white",contour_color='firebrick').generate(text)

plt.figure(figsize=(14,8))
plt.imshow(wordcloud, interpolation='nearest')
plt.axis("off")
plt.show()
plt.savefig("wc1.png", dpi=900)
blob = TextBlob(text)
sentiment = blob.sentiment.polarity
print(sentiment)
