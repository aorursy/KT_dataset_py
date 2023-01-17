!python -m spacy download es_core_news_md

!python -m spacy link es_core_news_md es_md

!pip install spacy_spanish_lemmatizer stopwordsiso stop_words tweet-preprocessor

!python -m spacy_spanish_lemmatizer download wiki
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import spacy

from spacy_spanish_lemmatizer import SpacyCustomLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import nltk

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from stop_words import get_stop_words

import stopwordsiso as stopwordsiso

from wordcloud import WordCloud, ImageColorGenerator

from PIL import Image

import requests

from io import BytesIO

import matplotlib

import matplotlib.pyplot as plt

import preprocessor as p

import re

import csv
nlp = spacy.load('es_md', disable=['ner']) # disabling Named Entity Recognition for speed

lemmatizer = SpacyCustomLemmatizer() 

nlp.add_pipe(lemmatizer, name="lemmatizer", after="tagger")

#

p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.MENTION, p.OPT.RESERVED, p.OPT.SMILEY)

# Stopword removal

stop_words = set(stopwords.words('spanish'))

stop_words_en = set(stopwords.words('english'))

stop_words_iso = set(stopwordsiso.stopwords(["es", "en"]))

reserved_words = ["rt", "fav", "españa", "paraguay", "vía", "nofollow", "twitter", "true", "href", "rel"]

key_words = ['coronavirus', 'coronavirusoutbreak', 'coronavirusPandemic', 'covid19', 'covid_19', 'epitwitter', 'ihavecorona', 'StayHomeStaySafe', 'TestTraceIsolate'] # twitter search keys

stop_words_es = set(get_stop_words('es'))

stop_words_en_ = set(get_stop_words('en'))

stop_words.update(stop_words_es)

stop_words.update(stop_words_en)

stop_words.update(stop_words_en_)

stop_words.update(stop_words_iso)

stop_words.update(reserved_words)

stop_words.update(key_words)

#

file_name = 'covid19-tweets-early-late-april'
li = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        df = pd.read_csv(os.path.join(dirname, filename), index_col=None, header=0)

        df = df[df['lang']=='es']

        li.append(df)



tweets_es = pd.concat(li, axis=0, ignore_index=True)

del li # free memory
tweets_es.text.sample(10)
file_name += '_lang_es'

tweets_es.info()

tweets_es.to_csv(file_name+'_lang_es.csv',encoding='utf8', index=False)
tweets_es_ES = tweets_es[tweets_es['country_code']=='ES']

tweets_es_ES.info()

file_name += '_country'

tweets_es_ES.to_csv(file_name+'_ES.csv',encoding='utf8', index=False)
tweets_es_PY = tweets_es[tweets_es['country_code']=='PY']

tweets_es_PY.info()

tweets_es_PY.to_csv(file_name+'_PY.csv',encoding='utf8', index=False)
# Preprocessing

def remove_string_special_characters(s):

    stripped = str(s)



    # Python regex, keep alphanumeric but remove numeric

    stripped = re.sub(r'\b[0-9]+\b', '', stripped)



    # Change any white space to one space

    stripped = re.sub('\s+', ' ', stripped)



    # Remove urls

    stripped = re.sub(r"http\S+",'', stripped)

    

    # check again

    #stripped = p.clean(stripped) be careful... also deletes ñ and accents

    

    # # to ''

    stripped = stripped.replace('#','')



    # Remove start and end white spaces

    stripped = stripped.strip()

    if len(stripped) >= 3:#stripped != '':

        return stripped.lower()
# only lemmas of content words           

def lemmatizer(text):        

    sent = []

    doc = nlp(text)

    for word in doc:

        if (word.pos_ not in ['VERB','ADV','ADJ','NOUN','PROPN']):

            continue

        sent.append(word.lemma_)

    return " ".join(sent)
# wordcloud

maxWords=200

def show_wordcloud_mask(data, mask, stopwords, fileName="wordcloud.png", title = None, maxWords=maxWords):

    wordcloud = WordCloud(

        background_color = 'white',

        max_words = maxWords,

        max_font_size = 180,

        scale = 3,

        random_state = 42,

        stopwords = stopwords,

        mask=mask,

        collocations=False,

    ).generate_from_frequencies(frequencies=data)



    # create coloring from image

    image_colors = ImageColorGenerator(mask)

    fig = plt.figure(figsize=[14,14])

    plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear")

    plt.axis("off")

    if title:

        fig.suptitle(title, fontsize = 20)

        fig.subplots_adjust(top = 2.3)

    plt.savefig(fileName, facecolor='k', bbox_inches='tight')

    return
def unigrams_freq_wordcloud(data,retweets=False,country='ES'):

    

    print('country', country, "="*50)



    data = data[data['is_retweet']==retweets]

    data["text"]=data["text"].astype(str)

    data.info()

    print(data['text'].sample(5))

    

    print("remove_string_special_characters ...")

    data["text"]= data["text"].apply(remove_string_special_characters)

    data.dropna(subset = ["text"], inplace=True)

    print(data['text'].sample(5))



    print("remove_stopwords ...")

    data["text"] = data["text"].apply(lambda y: ' '.join([x for x in nltk.word_tokenize(y) if ( x not in set(stop_words) and len(x)>2 and "covid" not in x) ]))

    data = data[~data['text'].isnull()]

    data.dropna(subset = ["text"], inplace=True)

    print(data['text'].sample(5))

    

    print("lemmatizer ...")

    data["text_lemmatize"] = data.apply(lambda x: lemmatizer(x['text']), axis=1)

    data['text'] = data['text_lemmatize'].str.replace('-PRON-', '')

    data.dropna(subset = ["text"], inplace=True)

    print(data['text'].sample(5))



    print("n-grams ...")

    gram = 1

    vectorizer = CountVectorizer(ngram_range = (gram,gram),stop_words=stop_words).fit(data.text.values.astype('U'))

    bag_of_words = vectorizer.transform(data["text"].values.astype('U'))

    sum_words = bag_of_words.sum(axis=0)

    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items() if len(word)>2]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)



    print("plot ...")

    mask_url = "https://live.staticflickr.com/7102/7378035790_231462ff2e_b.jpg"

    if country == 'PY':

        mask_url = "https://live.staticflickr.com/8161/7383472952_4f08c69c6c_b.jpg"

    response = requests.get(mask_url)

    mask = np.array(Image.open(BytesIO(response.content)))

    show_wordcloud_mask(dict(words_freq), mask, stop_words, "retweets"+str(retweets)+"country"+country+"-1gram.png")

    



    print("to_file ...")

    with open("word_freq-"+str(gram)+"gram_"+country+".csv", "w") as f:

        w = csv.writer(f)

        w.writerows(words_freq)

    f.close()

    

    return
unigrams_freq_wordcloud(tweets_es_ES,retweets=False,country='ES')

unigrams_freq_wordcloud(tweets_es_PY,retweets=False,country='PY')