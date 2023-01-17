# library imports
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.style.use('ggplot')
import matplotlib.pyplot as plt
width = 0.75
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.axis('off')
from nltk.corpus import stopwords
from textblob import TextBlob
import scattertext as st
import spacy
import spacy_cld

from IPython.display import IFrame
from IPython.core.display import display, HTML
from collections import Counter
from tqdm import tqdm_notebook as tqdm  # cool progress bars
tqdm().pandas()  # Enable tracking of progress in dataframe `apply` calls
tweets = pd.read_csv('../input/twcs/twcs.csv',encoding='utf-8')
print(tweets.shape)
tweets.head()
first_inbound = tweets[pd.isnull(tweets.in_response_to_tweet_id) & tweets.inbound]

QnR = pd.merge(first_inbound, tweets, left_on='tweet_id', 
                                  right_on='in_response_to_tweet_id')

# Filter to only outbound replies (from companies)
QnR = QnR[QnR.inbound_y ^ True]
print(f'Data shape: {QnR.shape}')
QnR.head()
# removing anonymized screen names 
def sn_replace(match):
    _sn = match.group(2).lower()
    if not _sn.isnumeric():
        # This is a company screen name
        return match.group(1) + match.group(2)
    return ''

sn_re = re.compile('(\W@|^@)([a-zA-Z0-9_]+)')
print("Removing anonymized screen names in X...")
QnR["text_x"] = QnR.text_x.progress_apply(lambda txt: sn_re.sub(sn_replace, txt))
print("Removing anonymized screen names in Y...")
QnR["text_y"] = QnR.text_y.progress_apply(lambda txt: sn_re.sub(sn_replace, txt))
#making sure the dataframe contains only the needed columns
QnR = QnR[["author_id_x","created_at_x","text_x","author_id_y","created_at_y","text_y"]]
QnR.head(5)
count = QnR.groupby("author_id_y")["text_x"].count()
c = count[count>15000].plot(kind='barh',figsize=(10, 8), color='#619CFF', zorder=2, width=width,)
c.set_ylabel('')
plt.show()
amazonQnR = QnR[QnR["author_id_y"]=="AmazonHelp"]
amazonQnR.tail(10)["text_x"]
# amazonQnR["text_x"] = amazonQnR["text_x"].str.encode("utf-8")
# amazonQnR["text_x"] = amazonQnR["text_x"].apply(str)
nlp_cld = spacy.load('en',disable_pipes=["tagger","ner"])
language_detector = spacy_cld.LanguageDetector()
nlp_cld.add_pipe(language_detector)
doc = nlp_cld(amazonQnR.iloc[4]["text_x"])
print(doc)
print(doc._.languages)  
print(doc._.language_scores)
mask = []
try:
    for i,doc in tqdm(enumerate(nlp_cld.pipe(amazonQnR["text_x"], batch_size=512))):
            if 'en' not in doc._.languages or len(doc._.languages) != 1:
                mask.append(False)
            else:
                mask.append(True)
except Exception:
    print("excepted ")
amazonQnR = amazonQnR[mask]
# sample a random fraction to visually ensure that we have only English tweets
amazonQnR.sample(frac=0.0002)    
amazonQnR.tail(10)["text_x"]
nlp = spacy.load("en_core_web_lg",disable_pipes=["tagger"])

from spacymoji import Emoji
emoji = Emoji(nlp)
nlp.add_pipe(emoji, first=True)
print(nlp.pipe_names)
emojis = []
for doc in tqdm(nlp.pipe(amazonQnR["text_x"], batch_size=512)):
    if doc._.has_emoji:
        for e in doc._.emoji:
            emojis.extend(e[0])
eCount = Counter(emojis)
eCount.most_common(10)
response_emojis = []
for doc in tqdm(nlp.pipe(amazonQnR["text_y"], batch_size=512)):
    elist = []
    if doc._.has_emoji:
        for e in doc._.emoji:
            elist.append(e[0])
    response_emojis.append(elist)
Counter([item for sublist in response_emojis for item in sublist]).most_common(10)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
sent_analyser = SentimentIntensityAnalyzer()
positive_text = "love this phone! its the best one I've owned over the years"
negative_text = "what sort of company makes such products? this phone hangs up all the time and is totally useless"
print("positive_text sentiment : ",sent_analyser.polarity_scores(positive_text)["compound"])
print("negative_text sentiment : ",sent_analyser.polarity_scores(negative_text)["compound"])
def sentiment(text):
    return (sent_analyser.polarity_scores(text)["compound"] + TextBlob(text).sentiment.polarity)/2
amazonQnR["text_x_sentiment"] = amazonQnR["text_x"].apply(sentiment)
response_emojis_for_positive_queries = []
response_emojis_for_negative_queries = []
for i,sentiment in enumerate(amazonQnR["text_x_sentiment"]):
    if sentiment > 0.0:
        response_emojis_for_positive_queries.extend(response_emojis[i])
    elif sentiment < 0.0:
        response_emojis_for_negative_queries.extend(response_emojis[i])
amazonQnR[amazonQnR["text_x_sentiment"]>0].head()
Counter(response_emojis_for_positive_queries).most_common(10)
Counter(response_emojis_for_negative_queries).most_common(10)
count = QnR.groupby("author_id_y")["text_x"].count()
c = count[count>15000].plot(kind='barh',figsize=(10, 8), color='#619CFF', zorder=2, width=width,)
c.set_ylabel('')
plt.show()
airlinesQnR = QnR[(QnR["author_id_y"]=="AmericanAir")|(QnR["author_id_y"]=="British_Airways")]
airlinesQnR.head(4)
airlinesQnR["text_y"] = airlinesQnR["text_y"].str.lower()  
stop = stopwords.words('english')
big_regex = re.compile(' | '.join(stop))
airlinesQnR["text_y"].progress_apply(lambda x: big_regex.sub(" ",x))
import scattertext as st
nlp = spacy.load('en',disable_pipes=["tagger","ner"])
airlinesQnR['parsed'] = airlinesQnR.text_y.progress_apply(nlp)
corpus = st.CorpusFromParsedDocuments(airlinesQnR,
                             category_col='author_id_y',
                             parsed_col='parsed').build()
html = st.produce_scattertext_explorer(corpus,
          category='British_Airways',
          category_name='British Airways',
          not_category_name='American Airlines',
          width_in_pixels=600,
          minimum_term_frequency=10,
          term_significance = st.LogOddsRatioUninformativeDirichletPrior(),
          )
# uncomment this cell to load the interactive scattertext visualisation
# filename = "americanAir-vs-britishAirways.html"
# open(filename, 'wb').write(html.encode('utf-8'))
# IFrame(src=filename, width = 800, height=700)
feat_builder = st.FeatsFromOnlyEmpath()
empath_corpus = st.CorpusFromParsedDocuments(airlinesQnR,
                                              category_col='author_id_y',
                                              feats_from_spacy_doc=feat_builder,
                                              parsed_col='parsed').build()
html = st.produce_scattertext_explorer(empath_corpus,
                                        category='British_Airways',
                                        category_name='British Airways',
                                        not_category_name='American Airlines',
                                        width_in_pixels=700,
                                        metadata=airlinesQnR['author_id_y'],
                                        use_non_text_features=True,
                                        use_full_doc=True,
                                        topic_model_term_lists=feat_builder.get_top_model_term_lists())
# uncomment this cell to load the interactive scattertext visualisation
# filename = "empath-BA-vs-AA.html"
# open(filename, 'wb').write(html.encode('utf-8'))
# IFrame(src=filename, width = 900, height=700)
corpus = (st.CorpusFromParsedDocuments(airlinesQnR,
                             category_col='author_id_y',
                             parsed_col='parsed').build().get_stoplisted_unigram_corpus())
target_term = 'delay'
html = st.word_similarity_explorer(corpus,
                                   category='British_Airways',
                                   category_name='British Airways',
                                   not_category_name='American Airlines',
                                   target_term=target_term,
                                   minimum_term_frequency=5,
                                   width_in_pixels=800)
# file_name = 'similarity.html'
# open(file_name, 'wb').write(html.encode('utf-8'))
# IFrame(src=file_name, width = 1000, height=700)
html = st.produce_projection_explorer(corpus,
                                   category='British_Airways',
                                   category_name='British Airways',
                                   not_category_name='American Airlines',
                                   width_in_pixels=800)
# file_name = 'projection.html'
# open(file_name, 'wb').write(html.encode('utf-8'))
# IFrame(src=file_name, width = 1200, height=700)
