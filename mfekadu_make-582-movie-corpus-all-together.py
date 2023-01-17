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



import json



import gc



from bs4 import BeautifulSoup



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
CSV_FILE = "/kaggle/input/582-movie-corpus/582_spring_2020_movie_corpus.csv"
df = pd.read_csv(CSV_FILE)
df.head(1)
df.columns
df.columns.drop("nameID").drop("Movie Watched").tolist()
prompts = df.columns.drop("nameID").drop("Movie Watched").tolist()

prompts = {

    f"prompt_{i}": p

    for i, p in zip(range(len(prompts)), prompts)

}
df.columns = [

    'nameID', 'Movie Watched',

    *prompts.keys()

]

df.columns
LAST_PROMPT_FOR_FOLDER = (

    "What kind of movies do you enjoy and why? "

    "Give at least three examples using "

    "popular movies released 2017-2020 "

    "that most people would recognize."

)

LAST_PROMPT_KEY = f"prompt_{len(prompts)}"

print(LAST_PROMPT_FOR_FOLDER)
prompts[LAST_PROMPT_KEY] = LAST_PROMPT_FOR_FOLDER
print([prompts])
with open('prompts.json', 'w', encoding='utf-8') as f:

    json.dump([prompts], f, ensure_ascii=False, indent=4)
pd.read_json('prompts.json')
df.head(1)
df = df.set_index("nameID")
df.head(4)
LAST_PROMPT_FOLDER = "/kaggle/input/582-movie-corpus/582_spring_2020_movie_corpus_prompt"

last_prompt_files = [

    os.path.join(dirname, filename)

    for dirname, _, filenames in os.walk(LAST_PROMPT_FOLDER)

    for filename in filenames

]

print(last_prompt_files[:2])
last_prompt_data = dict()



for fname in last_prompt_files:

    path, fn = os.path.split(fname)

    assert fn == "onlinetext.html"

    fol, name_id = os.path.split(path)

    assert fol == LAST_PROMPT_FOLDER

    name_id = int(name_id)

    print("reading ... ", name_id)

    with open(fname) as f:

        assert last_prompt_data.get(name_id, None) is None

        last_prompt_data[name_id] = BeautifulSoup(f.read()).text
last_prompt_df = pd.json_normalize(last_prompt_data).T.rename(columns={0:LAST_PROMPT_KEY})

last_prompt_df.head(2)
corpus_df = df.join(last_prompt_df,)
corpus_df.head(2)
corpus_df.isna().any()
corpus_df = corpus_df.fillna("")
corpus_df.isna().any()
mask = [

    "prompt_0",

    "prompt_6",

    "prompt_7",

    "prompt_8",

    "prompt_9",

    "prompt_11",

]

corpus_df["all_row_text"] = corpus_df[mask].agg(lambda x: "\n\n\n".join(x[0:]), axis=1)
corpus_df["all_reviewer_text"] = corpus_df[["all_row_text"]].reset_index().groupby("index").agg(lambda x: "\n\n\n---\n\n\n".join(x))
corpus_df["prompt_10"].unique()
# TODO: consider that some prompt_10 responsese are 'maybe'

corpus_df["boolean_rewatch"] = corpus_df["prompt_10"].apply(lambda x: True if x == "Yes" else False)
corpus_df["prompt_5"].unique()
def _get_numeric_rating(x):

    mapping = {

        '4 stars': 4,

        '2 stars': 2,

        '5 stars (best)': 5,

        '3 stars': 3,

        '1 star (worst)': 1,

    }

    return mapping[x]

rating_prompt_feats = [

    "plot",

    "acting",

    "production",

    "overall",

]

rating_prompt_cols = [

    "prompt_2", 

    "prompt_3", 

    "prompt_4", 

    "prompt_5",

]

for feat, col in zip(rating_prompt_feats, rating_prompt_cols):

    corpus_df[f"{feat}_rating"] = corpus_df[col].apply(

        lambda x: _get_numeric_rating(

            x

        )

    )
from enum import Enum

from collections import namedtuple
BAD = namedtuple("bad", ["nums", "str"])((1,2), "bad")

GOOD = namedtuple("good", ["nums", "str"])((4,5), "good")

EXCELLENT = namedtuple("excellent", ["nums", "str"])((5,), "excellent")

NON_EXCELLENT = namedtuple("non_excellent", ["nums", "str"])((1,2,3,4,), "non_excellent")

EH = namedtuple("eh", ["nums", "str"])((3,), "eh")

OK = namedtuple("ok", ["nums", "str"])((1,2,3,), "ok")

print(BAD)

print(GOOD)

print(EXCELLENT)

print(NON_EXCELLENT)

print(EH)

print(OK)

print(OK.nums)

print(OK.str)
def _is_good(x):

    return True if x in GOOD.nums else False



def _is_excellent(x):

    return True if x in EXCELLENT.nums else False



def _is_ok(x):

    return True if x in OK.nums else False



def _is_eh(x):

    return True if x in EH.nums else False



def _is_bad(x):

    return True if x in BAD.nums else False



def _categorize_g_e_b(x):

    if _is_good(x):

        return GOOD.str

    elif _is_eh(x):

        return EH.str

    elif _is_bad(x):

        return BAD.str

    else:

        raise NotImplementedError()



print(_is_good(4))

print(_is_good(2))

print(_categorize_g_e_b(5))

print(_categorize_g_e_b(4))

print(_categorize_g_e_b(3))

print(_categorize_g_e_b(2))

print(_categorize_g_e_b(1))

print(_is_ok(2))

print(_is_ok(3))
rating_prompt_feats = [

    "plot_rating",

    "acting_rating",

    "production_rating",

    "overall_rating",

]

for feat in rating_prompt_feats:

    corpus_df[f"{feat}_is_good"] = corpus_df[feat].apply(_is_good)

    corpus_df[f"{feat}_is_eh"] = corpus_df[feat].apply(_is_eh)

    corpus_df[f"{feat}_is_bad"] = corpus_df[feat].apply(_is_bad)

    corpus_df[f"{feat}_is_excellent"] = corpus_df[feat].apply(_is_excellent)

    corpus_df[f"{feat}_is_ok"] = corpus_df[feat].apply(_is_ok)

    corpus_df[f"{feat}_category"] = corpus_df[feat].apply(_categorize_g_e_b)
corpus_df.head()
corpus_df.columns
corpus_df.to_csv("corpus.csv")
author_prediction_task = (

    corpus_df[

        ["prompt_0"]

    ]

    .reset_index()

    .rename(columns={

        "prompt_0": "synopsis", 

        "index": "nameID"

    })

)

author_prediction_task.head(4)
author_prediction_task.to_csv("corpus_author_synopsis.csv")
mask = [

    "Movie Watched",

    "prompt_0",  # synopsis

    "prompt_1",  # 3 topic words

    "prompt_6",  # plot

    "prompt_7",  # acting

    "prompt_8",  # production

    "prompt_9",  # overall

    "prompt_11", # enjoy

]

nlg_task = (

    corpus_df[mask]

    .reset_index()

    .rename(

        columns={

            "index": "nameID",

            "prompt_0": "synopsis",

            "prompt_1": "topics",

            "prompt_6": "plot_review",

            "prompt_7": "acting_review",

            "prompt_8": "production_review",

            "prompt_9": "overall_review",

            "prompt_11": "enjoy",

        }

    )

)

nlg_task.head(1)
nlg_task.to_csv("corpus_generate_overall_review.csv")
grouped_by_movie = nlg_task.groupby("Movie Watched").agg(lambda x: "".join(x))

grouped_by_movie.head()
grouped_by_movie.to_csv("corpus_text_by_movie.csv")
grouped_by_author = nlg_task.groupby("nameID").agg(lambda x: "".join(x))

grouped_by_author.head()
grouped_by_author.to_csv("corpus_text_by_author.csv")
!ls
!pip install textacy
import spacy



import uuid

from pandas_profiling import ProfileReport



import textacy

from textacy.text_utils import KWIC  # key-words-in-context

from textacy import preprocessing



import textacy.ke  # for TextRank



import textacy.vsm  # for Vectorizer



import textacy.tm  # for TopicModel



from pprint import pprint 
nlp = spacy.load("en_core_web_lg")
corpus_df.head(1)
corpus_df.columns
df = corpus_df
df.columns
synopsis = df["prompt_0"].values[0]

doc = synopsis

print(doc)
for context in KWIC(doc, "you", window_width=10):

    pass
for context in KWIC(doc, "learn", window_width=10):

    pass
for context in KWIC(doc, "movie", window_width=10):

    pass
for context in KWIC(doc, "learn", window_width=10):

    pass
for context in KWIC(doc, " he", window_width=10):

    pass
for context in KWIC(doc, "they", window_width=10):

    pass
for context in KWIC(doc, "interview", window_width=10):

    pass
def normalize(text):

    return preprocessing.normalize_whitespace(

        text

    )
print(doc[:80])

print("-----")

print(normalize(doc)[:80])
set([

    str(x) for x in

    textacy.extract.words(

    nlp(normalize(doc)), 

    min_freq=5,

)])
set([

    str(x) for x in

    textacy.extract.noun_chunks(

    nlp(normalize(doc)), 

    min_freq=3,

)])
set([

    str(x) for x in

    textacy.extract.entities(

    nlp(normalize(doc)), 

    min_freq=2,

)])
list(

    textacy.extract.semistructured_statements(

        nlp(normalize(doc)),

        entity="Kim Jong Un",

        ignore_entity_case=True,

))
list(

    textacy.extract.semistructured_statements(

        nlp(normalize(doc)),

        entity="Kim",

        ignore_entity_case=True,

))
list(

    textacy.extract.semistructured_statements(

        nlp(normalize(doc)),

        entity="North Korea",

        ignore_entity_case=True,

))
list(

    textacy.extract.semistructured_statements(

        nlp(normalize(doc)),

        entity="Rapaport",

        ignore_entity_case=True,

))
list(

    textacy.extract.semistructured_statements(

        nlp(normalize(doc)),

        entity="Skylark",

        ignore_entity_case=True,

))
set([

    str(x) for x in

    textacy.extract.ngrams(

        nlp(normalize(doc)), 

        2,

        min_freq=2,

        filter_stops=True, 

        filter_punct=True, 

        filter_nums=False

)])
set([

    str(x) for x in

    textacy.extract.ngrams(

        nlp(normalize(doc)), 

        3,

        min_freq=2,

        filter_stops=True, 

        filter_punct=True, 

        filter_nums=False

)])
textacy.ke.textrank(

    nlp(normalize(doc)), 



    # TextRank: 

    window_size=2, edge_weighting="binary", position_bias=False,



    normalize="lemma", 

    topn=10

)
textacy.ke.textrank(

    nlp(normalize(doc)), 



    # SingleRank:

    window_size=10, edge_weighting="count", position_bias=False,



    normalize="lemma", 

    topn=10

)
textacy.ke.textrank(

    nlp(normalize(doc)), 



    # PositionRank: 

    window_size=10, edge_weighting="count", position_bias=True,



    normalize="lemma", 

    topn=10

)
textacy.ke.yake(

    nlp(normalize(doc)), 

)
textacy.ke.scake(

    nlp(normalize(doc)), 

)
textacy.ke.sgrank(

    nlp(normalize(doc)), 

)
ts = textacy.TextStats(

    nlp(normalize(doc))

)

print("ts.basic_counts:")

pprint(ts.basic_counts)

print("-----")

print("ts.readability_stats:")

pprint(ts.readability_stats)
corpus = textacy.Corpus("en", data=corpus_df["all_row_text"].tolist())

# synopsys_corpus = textacy.Corpus(

#     "en", 

#     data=corpus_df["prompt_0"].tolist()

# )
print(corpus)
corpus[1]._.preview
[doc._.preview for doc in corpus[10:15]]
corpus.n_docs, corpus.n_sents, corpus.n_tokens
word_counts = corpus.word_counts(

    normalize="lemma",

    weighting="count",

    as_strings=True,

    filter_stops=True,

    filter_punct=True,

    filter_nums=True,

)

word_counts = {k:v for k,v in word_counts.items() if len(k) > 2}

sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:50]
tf_idf_vectorizer = textacy.vsm.Vectorizer(

    tf_type="linear",

    apply_idf=True,

    idf_type="smooth",

    apply_dl=True,

    dl_type="sqrt",

    norm=None,  # default None; "l1", "l2"

    min_df=1, # default 1  

    max_df=1.0  # default 1.0

)

tf_idf_vectorizer
bow_vectorizer = textacy.vsm.Vectorizer(

    tf_type="linear",

    apply_idf=False,

    apply_dl=False,

    norm=None,  # default None; "l1", "l2"

    min_df=1, # default 1  

    max_df=0.95  # default 1.0

)

bow_vectorizer
tf_idf_doc_term_matrix = tf_idf_vectorizer.fit_transform(

    (

        doc._.to_terms_list(

            ngrams=1, 

            entities=True, 

            as_strings=True

        )

        for doc in corpus

    )

)

tf_idf_doc_term_matrix
bow_doc_term_matrix = bow_vectorizer.fit_transform(

    (

        doc._.to_terms_list(

            ngrams=3,  # default (1,2,3) grams

            entities=True, 

            as_strings=True,

            filter_stops=True,

            filter_punct=True,

            filter_nums=True,

            normalize="lemma",

            weighting="count",

        )

        for doc in corpus

    )

)

bow_doc_term_matrix
N = 3



lda_model = textacy.tm.TopicModel(

    "lda", n_topics=N

)

nmf_model = textacy.tm.TopicModel(

    "nmf", n_topics=N

)

lsa_model = textacy.tm.TopicModel(

    "lsa", n_topics=N

)

print(lda_model)

print(nmf_model)

print(lsa_model)
lda_model.fit(bow_doc_term_matrix)

nmf_model.fit(tf_idf_doc_term_matrix)

lsa_model.fit(tf_idf_doc_term_matrix)



lda_doc_topic_matrix = lda_model.transform(bow_doc_term_matrix)

nmf_doc_topic_matrix = nmf_model.transform(tf_idf_doc_term_matrix)

lsa_doc_topic_matrix = lsa_model.transform(tf_idf_doc_term_matrix)



print(lda_doc_topic_matrix.shape)

print(nmf_doc_topic_matrix.shape)

print(lsa_doc_topic_matrix.shape)
def print_topic_terms(model, vectorizer, top_n=15):

    for topic_idx, top_terms in model.top_topic_terms(vectorizer.id_to_term, top_n=top_n):

        print("\ntopic", topic_idx, ":", "   ".join(top_terms))



print(lda_model)

print_topic_terms(lda_model, bow_vectorizer)

print("-----"*10)

print(nmf_model)

print_topic_terms(nmf_model, tf_idf_vectorizer)

print("-----"*10)

print(lsa_model)

print_topic_terms(lsa_model, tf_idf_vectorizer)
def visualize_tm(model, doc_term_matrix, vectorizer, sort_terms_by="seriation"):

    model.termite_plot(

        doc_term_matrix,

        vectorizer.id_to_term,

        topics=-1,

        n_terms=100,

        rank_terms_by="topic_weight",

        sort_terms_by=sort_terms_by, # "seriation", "weight", "index", "alphabetical"

        highlight_topics=[0,1,2]

    )
visualize_tm(

    lda_model, 

    lda_doc_topic_matrix,

    bow_vectorizer,

    sort_terms_by="weight"

)
visualize_tm(

    nmf_model,

    nmf_doc_topic_matrix,

    tf_idf_vectorizer,

)
import warnings
warnings.filterwarnings("ignore")
visualize_tm(

    lsa_model,

    lsa_doc_topic_matrix,

    tf_idf_vectorizer,

)
def  _get_first_50_entities(doc):

    return list(textacy.extract.entities(

        doc, 

        exclude_types=("NUMERIC", "ORDINAL", "MONEY", "DATE"),

    ))[:50]





def _get_semistructured_statements(doc):

    ents = _get_first_50_entities(doc)

    def _fun(e):

        return list(textacy.extract.semistructured_statements(

            doc, 

            entity=e.text,

            ignore_entity_case=True,

        ))

    return { str(e): _fun(e) for e in ents }





def _get_first_50_bigrams_with_min_freq_2(doc):

    return list(textacy.extract.ngrams(

        doc, 

        n=2,

        min_freq=2,

        filter_stops=True, 

        filter_punct=True, 

        filter_nums=True,

    ))[:50]





def _get_first_50_trigrams(doc):

    return list(textacy.extract.ngrams(

        doc, 

        n=3,

        min_freq=1,

        filter_stops=True, 

        filter_punct=True, 

        filter_nums=False,

    ))[:50]





def _get_top_10_textrank_key_terms(doc):

    return textacy.ke.textrank(

        doc, 



        # TextRank: 

        window_size=2, edge_weighting="binary", position_bias=False,



        normalize="lemma", 

        topn=10,

    )





def _get_top_10_singlerank_key_terms(doc):

    return textacy.ke.textrank(

        doc, 



        # SingleRank:

        window_size=10, edge_weighting="count", position_bias=False,



        normalize="lemma", 

        topn=10,

    )





def _get_top_10_positionrank_key_terms(doc):

    return textacy.ke.textrank(

        doc,



        # PositionRank: 

        window_size=10, edge_weighting="count", position_bias=True,



        normalize="lemma", 

        topn=10,

    )





def _get_top_10_yake_key_terms(doc):

    return textacy.ke.yake(

        doc, 

        normalize="lemma", 

        topn=10,

    )





def _get_top_10_scake_key_terms(doc):

    return textacy.ke.scake(

        doc,

        normalize="lemma",

        topn=10,

    )





def _get_top_10_sgrank_key_terms(doc):

    return textacy.ke.sgrank(

        doc, 

        normalize="lemma", 

        topn=10

    )





def _get_basic_counts(doc):

    ts = textacy.TextStats(

        doc

    )

    return ts.basic_counts





def _get_readability_stats(doc):

    ts = textacy.TextStats(

        doc

    )

    return ts.readability_stats

nlp
i = 1

def _get_the_document_features(text):

    doc = nlp(normalize(text))

    num_features = 11

    def _with_print(fun):

        global i

        i+=1

        print(str(i/num_features),"running...", fun, " "*100, "\r", end="")

        return fun

    data = {

        "first_50_entities" : (

            _with_print(

                _get_first_50_entities

            )(doc)

        ),

        "semistructured_statements" : (

            _with_print(

                _get_semistructured_statements

            )(doc)

        ),

        "first_50_bigrams_with_min_freq_2" : (

            _with_print(

                _get_first_50_bigrams_with_min_freq_2

            )(doc)

        ),

        "first_50_trigrams" : (

            _with_print(

                _get_first_50_trigrams

            )(doc)

        ),

        "top_10_textrank_key_terms" : (

            _with_print(

                _get_top_10_textrank_key_terms

            )(doc)

        ),

        "top_10_singlerank_key_terms" : (

            _with_print(

                _get_top_10_singlerank_key_terms

            )(doc)

        ),

        "top_10_positionrank_key_terms" : (

            _with_print(

                _get_top_10_positionrank_key_terms

            )(doc)

        ),

        "top_10_yake_key_terms" : (

            _with_print(

                _get_top_10_yake_key_terms

            )(doc)

        ),

        "top_10_scake_key_terms" : (

            _with_print(

                _get_top_10_scake_key_terms

            )(doc)

        ),

        # VERY SLOW!

#         "top_10_sgrank_key_terms" : (

#             _with_print(

#                 _get_top_10_sgrank_key_terms

#             )(doc)

#         ),

        "basic_counts" : (

            _with_print(

                _get_basic_counts

            )(doc)

        ),

        "readability_stats" : (

            _with_print(

                _get_readability_stats

            )(doc)

        ),

    }



    assert len(data) == num_features



    return data
TEXT_COLUMN_KEY = 13

TEXT_COLUMN_NAME = "all_row_text"
df.columns[TEXT_COLUMN_KEY]
df = df.join(

    df.apply(

        (lambda x:_get_the_document_features(x[TEXT_COLUMN_KEY])), 

        axis=1, 

        result_type='expand'

    )

)
df.head(1)
def _get_svo_tuple(sentence):

    svo_extract = textacy.extract.subject_verb_object_triples

    res = tuple()

    for tup in svo_extract(sentence):

        res = tuple(str(phrase) for phrase in tup)

    return res
print(_get_svo_tuple(nlp("Careem was acquired by Uber.")))

print(_get_svo_tuple(nlp("Uber acquired Careem.")))

print(_get_svo_tuple(nlp("Careem acquired Uber.")))
def _get_sents(doc):

    return [s for s in doc.sents]
def get_nlp_data(nlp):

    def fun(full_text):

        doc = nlp(full_text)

        sents = _get_sents(doc)

        #svo_tuples = [_get_svo_tuple(sent) for sent in sents]

        #assert len(sents) == len(svo_tuples)

        nlp_data = []

        sent_position = 0 # in loop init 0+1

        for tup, sent in (

            (_get_svo_tuple(_s), _s)

            for _s in sents

        ):

            sent_position += 1

            #tup = svo_tuples[i]

            if len(tup) == 0:

                s, v, o = None, None, None

            elif len(tup) == 3:

                s, v, o = tup

            else:

                raise NotImplementedError()



            sent_text = str(sent)

            sent_doc = sent.as_doc()

            ts = textacy.TextStats(

                sent_doc

            )

            

            read_stats = ts.readability_stats

            

            def _else_none(data, key):

                if data == None:

                    return None

                else:

                    return data.get( key, None )



            nlp_data.append({

                # perhaps `position` might help

                #       to discover stats based on 

                #       closeness to start/finish

                "id": uuid.uuid4().hex,

                "position": sent_position,

                "sent": sent_text,

                "s": str(s) if s is not None else None,

                "v": str(v) if v is not None else None,

                "o": str(o) if o is not None else None,

                'n_chars': ts.n_chars,

                'n_long_words': ts.n_long_words,

                'n_monosyllable_words': ts.n_monosyllable_words,

                'n_polysyllable_words': ts.n_polysyllable_words,

                'n_sents': ts.n_sents,

                'n_syllables': ts.n_syllables,

                'n_unique_words': ts.n_unique_words,

                'n_words': ts.n_words,

                'automated_readability_index': _else_none(

                    read_stats,

                    "automated_readability_index",

                ),

                'coleman_liau_index': _else_none(

                    read_stats,

                    "coleman_liau_index",

                ),

                'flesch_kincaid_grade_level': _else_none(

                    read_stats,

                    "flesch_kincaid_grade_level",

                ),

                'flesch_reading_ease': _else_none(

                    read_stats,

                    "flesch_reading_ease",

                ),

                'gulpease_index': _else_none(

                    read_stats,

                    "gulpease_index",

                ),

                'gunning_fog_index': _else_none(

                    read_stats,

                    "gunning_fog_index",

                ),

                'lix': _else_none(

                    read_stats,

                    "lix",

                ),

                'smog_index': _else_none(

                    read_stats,

                    "smog_index",

                ),

                'wiener_sachtextformel': _else_none(

                    read_stats,

                    "wiener_sachtextformel",

                ),

            })

        return {

            "id": uuid.uuid4().hex,

            "sent_data": nlp_data,

        }

    return fun
i = 1

n = len(df[TEXT_COLUMN_NAME])

def _with_log(fun):

    def wrapper(x):

        global i

        print(

            f"{i} / {n}", 

            " "*50, 

            "\r", 

            end=""

        )

        i += 1

        return fun(x)

    return wrapper





df["nlp_data"] = (

    df[TEXT_COLUMN_NAME]

    .apply(

        _with_log(get_nlp_data(nlp))

    )

)
# df.reset_index().groupby(["index", "Movie Watched"]).agg(tuple)#.head()

df["nlp_data.id"] = df["nlp_data"].apply(lambda x: x["id"])

df["nlp_data.sent_data"] = df["nlp_data"].apply(lambda x: x["sent_data"])
read_stats_df = pd.json_normalize(df["readability_stats"])

for k in read_stats_df:

    df[f"doc_{k}"] = read_stats_df[k]
basic_counts_df = pd.json_normalize(df["basic_counts"])

for k in basic_counts_df:

    df[f"doc_{k}"] = basic_counts_df[k]
df.head(1)
del read_stats_df

del basic_counts_df

gc.collect()
exploded_df = df.explode("nlp_data.sent_data")



exploded_df["nlp_data.sent_data.id"] = (

    exploded_df["nlp_data.sent_data"]

    .apply(lambda x: x["id"])

)



exploded_df["nlp_data.sent_data.sent"] = (

    exploded_df["nlp_data.sent_data"]

    .apply(lambda x: x["sent"])

)



exploded_df["nlp_data.sent_data.subject"] = (

    exploded_df["nlp_data.sent_data"]

    .apply(lambda x: x["s"])

)



exploded_df["nlp_data.sent_data.verb"] = (

    exploded_df["nlp_data.sent_data"]

    .apply(lambda x: x["v"])

)



exploded_df["nlp_data.sent_data.object"] = (

    exploded_df["nlp_data.sent_data"]

    .apply(lambda x: x["o"])

)



exploded_df.head(1)
the_other_featurers = [

    'n_chars',

    'n_long_words',

    'n_monosyllable_words',

    'n_polysyllable_words',

    'n_sents',

    'n_syllables',

    'n_unique_words',

    'n_words',

    'automated_readability_index',

    'coleman_liau_index',

    'flesch_kincaid_grade_level',

    'flesch_reading_ease',

    'gulpease_index',

    'gunning_fog_index',

    'lix',

    'smog_index',

    'wiener_sachtextformel',

]





for i in range(len(the_other_featurers)):

    exploded_df[

        f"nlp_data.sent_data.{the_other_featurers[i]}"

    ] = (

        exploded_df["nlp_data.sent_data"]

        .apply(lambda x: x[the_other_featurers[i]])

    )



exploded_df.head(1)
exploded_df.columns
doc_feat_mask = [

    'first_50_entities',

    'semistructured_statements',

    'first_50_bigrams_with_min_freq_2',

    'first_50_trigrams',

    'top_10_textrank_key_terms',

    'top_10_singlerank_key_terms',

    'top_10_positionrank_key_terms',

    'top_10_yake_key_terms',

    'top_10_scake_key_terms',

#     'basic_counts',

#     'readability_stats',

    'doc_n_sents',

    'doc_n_words',

    'doc_n_chars',

    'doc_n_syllables',

    'doc_n_unique_words',

    'doc_n_long_words',

    'doc_n_monosyllable_words',

    'doc_n_polysyllable_words',

    'doc_flesch_kincaid_grade_level',

    'doc_flesch_reading_ease',

    'doc_smog_index',

    'doc_gunning_fog_index',

    'doc_coleman_liau_index',

    'doc_automated_readability_index',

    'doc_lix',

    'doc_gulpease_index',

#     'nlp_data',

    'nlp_data.id',

#     'nlp_data.sent_data',

]



sent_feat_mask = [

    'nlp_data.sent_data.id',

    'nlp_data.sent_data.sent',

    'nlp_data.sent_data.subject',

    'nlp_data.sent_data.verb',

    'nlp_data.sent_data.object',

    'nlp_data.sent_data.n_chars',

    'nlp_data.sent_data.n_long_words',

    'nlp_data.sent_data.n_monosyllable_words',

    'nlp_data.sent_data.n_polysyllable_words',

    'nlp_data.sent_data.n_sents',

    'nlp_data.sent_data.n_syllables',

    'nlp_data.sent_data.n_unique_words',

    'nlp_data.sent_data.n_words',

    'nlp_data.sent_data.automated_readability_index',

    'nlp_data.sent_data.coleman_liau_index',

    'nlp_data.sent_data.flesch_kincaid_grade_level',

    'nlp_data.sent_data.flesch_reading_ease',

    'nlp_data.sent_data.gulpease_index',

    'nlp_data.sent_data.gunning_fog_index',

    'nlp_data.sent_data.lix',

    'nlp_data.sent_data.smog_index',

    'nlp_data.sent_data.wiener_sachtextformel',

]

    

mask = [

    *doc_feat_mask,

    *sent_feat_mask,

]

corpus_sents_df = exploded_df[mask]
corpus_sents_df.size
corpus_sents_df.shape
corpus_sents_df.head(3)


def _span_list_in_text(span_list, text):

    return any(

        [span.text in text for span in span_list]

    )



corpus_sents_df["ent_in_text"] = [

    _span_list_in_text(

        row["first_50_entities"], 

        row["nlp_data.sent_data.sent"]

    )

    for index, row in corpus_sents_df.iterrows()

]
corpus_sents_df["ent_in_text"].head(2)
corpus_sents_df["top_10_textrank_key_terms"].values[0]
def _rank_tuple_list_in_text(rank_tuple_list, text):

    return any(

        [rank_tuple[0] in text for rank_tuple in rank_tuple_list]

    )



corpus_sents_df["tr_in_text"] = [

    _rank_tuple_list_in_text(

        row["top_10_textrank_key_terms"], 

        row["nlp_data.sent_data.sent"]

    )

    for index, row in corpus_sents_df.iterrows()

]
corpus_sents_df["tr_in_text"].head(2)
corpus_sents_df.head()
corpus_sents_df.to_csv("corpus_sents.csv")
!ls
corpus_sents_df.columns
list_feats = [

    'first_50_entities', 

    'semistructured_statements',

    'first_50_bigrams_with_min_freq_2', 

    'first_50_trigrams',

    'top_10_textrank_key_terms', 

    'top_10_singlerank_key_terms',

    'top_10_positionrank_key_terms', 

    'top_10_yake_key_terms',

    'top_10_scake_key_terms', 

]



doc_feats = [

    'doc_n_sents', 

    'doc_n_words', 

    'doc_n_chars',

    'doc_n_syllables', 

    'doc_n_unique_words', 

    'doc_n_long_words',

    'doc_n_monosyllable_words', 

    'doc_n_polysyllable_words',

    'doc_flesch_kincaid_grade_level', 

    'doc_flesch_reading_ease',

    'doc_smog_index', 

    'doc_gunning_fog_index', 

    'doc_coleman_liau_index',

    'doc_automated_readability_index', 

    'doc_lix', 

    'doc_gulpease_index',

#     'nlp_data.id', 

]



sent_feats = [

#     'nlp_data.sent_data.id', 

    'nlp_data.sent_data.sent',

    'nlp_data.sent_data.subject', 

    'nlp_data.sent_data.verb',

    'nlp_data.sent_data.object', 

    'nlp_data.sent_data.n_chars',

    'nlp_data.sent_data.n_long_words',

    'nlp_data.sent_data.n_monosyllable_words',

    'nlp_data.sent_data.n_polysyllable_words', 

    'nlp_data.sent_data.n_sents',

    'nlp_data.sent_data.n_syllables',

    'nlp_data.sent_data.n_unique_words',

    'nlp_data.sent_data.n_words',

    'nlp_data.sent_data.automated_readability_index',

    'nlp_data.sent_data.coleman_liau_index',

    'nlp_data.sent_data.flesch_kincaid_grade_level',

    'nlp_data.sent_data.flesch_reading_ease',

    'nlp_data.sent_data.gulpease_index',

    'nlp_data.sent_data.gunning_fog_index',

    'nlp_data.sent_data.lix',

    'nlp_data.sent_data.smog_index',

    'nlp_data.sent_data.wiener_sachtextformel', 

    'ent_in_text',

    'tr_in_text'

]





mask = [

#     *list_feats,

    *doc_feats,

    *sent_feats,

]



for_profile_df = (

    corpus_sents_df[mask]

    .reset_index()

    .drop("index", axis=1)

)

for_profile_df.head(1)
profile = ProfileReport(

    for_profile_df, 

    title='corpus_sents_df Report', 

    html={

        'style':{

            'full_width':True

        }

    }

)
profile.to_notebook_iframe()
profile = ProfileReport(

    corpus_df, 

    title='corpus_df Report', 

    html={

        'style':{

            'full_width':True

        }

    }

)
profile.to_notebook_iframe()