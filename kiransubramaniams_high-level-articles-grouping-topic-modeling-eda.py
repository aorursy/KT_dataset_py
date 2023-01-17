import pandas as pd

from tqdm import tqdm

df = pd.read_csv('../input/CORD-19-research-challenge/2020-03-13/all_sources_metadata_2020-03-13.csv')

# biorxiv_clean = pd.read_csv("../input/cord-19-eda-parse-json-and-generate-clean-csv/biorxiv_clean.csv")

# clean_comm_use = pd.read_csv("../input/cord-19-eda-parse-json-and-generate-clean-csv/clean_comm_use.csv")

# clean_noncomm_use = pd.read_csv("../input/cord-19-eda-parse-json-and-generate-clean-csv/clean_noncomm_use.csv")

# clean_pmc = pd.read_csv("../input/cord-19-eda-parse-json-and-generate-clean-csv/clean_pmc.csv")

# df = pd.concat([biorxiv_clean, clean_comm_use, clean_noncomm_use, clean_pmc]).reset_index(drop=True)
df.head(2)
df.shape
duplicate_paper = ~(df.title.isnull() | df.abstract.isnull()) & (df.duplicated(subset=['title', 'abstract']))

df = df[~duplicate_paper].reset_index(drop=True)
df.shape
import nltk

from nltk.corpus import stopwords

from nltk import ngrams

stop_words = set(stopwords.words('english'))
def pos_tag_text(actual_text, print_ = False):

    p_tg = (nltk.pos_tag(nltk.word_tokenize(actual_text)))

    if(print_):

        for e in p_tg:

            print(e[0]+' : '+e[1])

    return p_tg
reject = ['(', ')', 'IN', 'DT', ':', 'CC', ',', '.']

def prep_text(text_in):

    try:

        rfa = pos_tag_text(text_in)

        ret_tex = ''

        for ev in rfa:

            if(ev[1] not in reject):

                ret_tex = ret_tex + ' ' + ev[0]

    except:

        ret_tex = str(text_in)

    return(ret_tex.strip().lower())
title_without_sw = []

for et in tqdm(df['abstract']):

    title_without_sw.append(prep_text(et))
df['title_without_sw'] = title_without_sw
def get_n_grams(df_c, ngr):

    kwrds_list = []

    for et in df_c:

        split_up = str(et).lower().split()

        for ew in ngrams(split_up, ngr):

            bg = ''

            for w in ew:

                bg = bg + ' '+ w

            kwrds_list.append(bg.strip())

    return(kwrds_list)
key_w_l = []

key_w_l_bi = []

key_w_l_tri = []

for et in tqdm(df['title_without_sw']):

    split_up = []

    for evwd in str(et).lower().split():

        if(evwd not in stop_words):

            split_up.append(evwd)

    #code to create single keywords list

    tem_ = []

    for eww in split_up:

        if eww not in tem_:

            tem_.append(eww)

            key_w_l.append(eww)

        #bigrams

    for ew in ngrams(split_up, 2):

        bg = ''

        for w in ew:

            bg = bg + ' '+ w

        key_w_l_bi.append(bg.strip())

        #tri-grams

    for ew in ngrams(split_up, 3):

        bg = ''

        for w in ew:

            bg = bg + ' '+ w

        key_w_l_tri.append(bg.strip())

        
pd.Series(key_w_l).value_counts()[:20]
pd.Series(key_w_l_bi).value_counts()[:20]
pd.Series(key_w_l_tri).value_counts()[:25]
def is_in_top(text, top_key_ws_lst):

    #code to remove stopwords

    ret_v = ''

    for ewd in text.split():

        if ewd not in stop_words:

            ret_v = ret_v+' '+ ewd

    ret_v = ret_v.strip()

    fl = False

    for ek in top_key_ws_lst:

        for ev in ek.split():

            #print(ev)

            if(ev in text):

                if (len(ev)>3):

                    fl = True

                    for ef in ret_v.split():

                        if(ev in ef):

                            ret_v = ret_v.replace(ef,'').replace('  ',' ')

    return [ret_v,fl]
top_ct = 25

is_in_top_tri = []

top_text_tri = []

tri = pd.Series(key_w_l_tri).value_counts()[:top_ct].index.tolist()

for et in tqdm(df['title_without_sw']):

    

    te = is_in_top(et, tri)

    is_in_top_tri.append(te[1])

    top_text_tri.append(te[0])

    

df['is_in_top_tri'] = is_in_top_tri

df['top_text_tri'] = top_text_tri
df['is_in_top_tri'].value_counts()
df_tri = df[df['is_in_top_tri'] == True]

df_tri.reset_index(inplace = True) 

# for i, r in df_tri[:3].iterrows():

#     print('\n====================\n'+r['title']+'\n'+r['top_text_tri'])

#     pos_tag_text(r['top_text_tri'], True)
df_tri.head(10)
df_tri.shape
import numpy as np 

from sklearn.feature_extraction import text

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.decomposition import LatentDirichletAllocation



#import scispacy

#import spacy



from scipy.spatial.distance import jensenshannon



import joblib



import pyLDAvis

import pyLDAvis.sklearn

pyLDAvis.enable_notebook()
#import en_core_sci_sm

#nlp = en_core_sci_sm.load()
#def spacy_tokenizer(sentence):

#    return [word.lemma_ for word in nlp(sentence)]

def print_top_words(model, feature_names, n_top_words):

    for topic_idx, topic in enumerate(model.components_):

        message = "\nTopic #%d: " % topic_idx

        message += " ".join([feature_names[i]

                             for i in topic.argsort()[:-n_top_words - 1:-1]])

        print(message)

    print()
tf_vectorizer = CountVectorizer(strip_accents = 'unicode',

                                stop_words = stop_words,

                                lowercase = True

                               )
tf = tf_vectorizer.fit_transform(df_tri['top_text_tri'].str.replace('\n\n', ''))

tf.shape
lda_tf = LatentDirichletAllocation(n_components=4, random_state=0)

lda_tf.fit(tf)

tfidf_feature_names = tf_vectorizer.get_feature_names()

#print_top_words(lda_tf, tfidf_feature_names, 5)

#viz = pyLDAvis.sklearn.prepare(lda_tf, tf, tf_vectorizer)

#pyLDAvis.display(viz)
topic_dist = pd.DataFrame(lda_tf.transform(tf))

topic_dist.shape
def get_k_nearest_docs(doc_dist, k=5, use_jensenshannon=True):

    '''

    doc_dist: topic distribution (sums to 1) of one article

    

    Returns the index of the k nearest articles (as by Jensen–Shannon divergence/ Euclidean distance in topic space). 

    '''

    

    if use_jensenshannon:

            distances = topic_dist.apply(lambda x: jensenshannon(x, doc_dist), axis=1)

    else:

        diff_df = topic_dist.sub(doc_dist)

        distances = np.sqrt(np.square(diff_df).sum(axis=1)) # euclidean distance (faster)

        

    return distances[distances != 0].nsmallest(n=k).index
topic_dist.head(5)
def get_titles(vectr, k, condition=['']):

    recommended = get_k_nearest_docs(vectr, k=k)

    for i in recommended:

        title_ = df_tri['title'][i]

        pr = False

        for l in condition:

            if l in title_:

                pr = True

        if(pr):

            print('- ', df_tri['title'][i] )

            print('==========================')
viru_kws = ['virus', 'corona', 'middle', 'east', 'respiratory']
get_titles([1,0,0,0], 20, viru_kws)
get_titles([0,1,0,0], 20, viru_kws)
get_titles([0,0,1,0], 20, viru_kws)
get_titles([0,0,0,1], 20, viru_kws)