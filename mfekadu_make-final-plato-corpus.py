# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



import gc



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
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
!ls /kaggle/input/stanford-plato-corpus
CSV_FILE = "/kaggle/input/stanford-plato-corpus/data_per_paragraph.csv"
df = pd.read_csv(CSV_FILE)

df.columns
df.head()
def _concat_zip(zip_obj):

    d = dict()

    for k,v in zip_obj:

        if (

            type(v) != float 

            and k != np.nan 

            and v != None 

            and v != np.nan

        ):

            d[str(k)] = f"{d.get(k, '')}\n\n\n{v}".strip()

        else:

            d[str(k)] = None

    return d





def _norm_dict(zip_obj):

    d = dict()

    for k,v in zip_obj:

        if type(v) != float and k != np.nan and v != None:

            d[str(k)] = d.get(k, v).strip()

        else:

            d[str(k)] = None

    return d





def _concat_dicts(d1, d2):

    return _concat_zip([*d1.items(), *d2.items()])



def _dict_to_string(d):

    t = ""

    for k, v in d.items():

        if (

            type(v) != float and k != np.nan and v != None 

            and v != "" and v != np.nan

        ):

            t += v + "\n\n\n"

    return t



z1 = zip([1,1,2,2, np.nan, np.nan, 0.0], ["Heading_ok","heading_wow","Heading_cool","Heading_foo", np.nan, "bad_data", np.nan])

c1 = _concat_zip(z1)



z2 = zip([1,1,2,2], ["_p_haha","_p_lol","_p_lmao","_p_rofl"])

c2 = _concat_zip(z2)



print(c1)

print(c2)



print(c1.items())

print(c2.items())



print([*c1.items(), *c2.items()])

print("-----"*5)

pprint(_concat_dicts(c1, c2))

print("-----"*5)

pprint(_concat_dicts(c1, c2).values())

print("-----"*5)

print(_dict_to_string(_concat_zip([*c1.items(), *c2.items()])))

print("-----"*5)

print(_norm_dict([*c1.items(), *c1.items()]))
mask = [

    "filename", 

    "topic", 

    "related_topic",

    "author", 

    "title",

    "preamble_text", 

    "section.id",

    "section.heading_text", 

    "section.paragraph.text"

]



def _create_article_data(x):

    fnames = set(x["filename"])

    assert len(fnames) == 1, "there should be only 1 filename"

    fname = fnames.pop()

    

    titles = set(x["title"])

    assert len(titles) == 1, "there should be only 1 title"

    title = titles.pop()

    

    topics = set(x["topic"])

    assert len(topics) == 1, "there should be only 1 topic"

    topic = topics.pop()



    related_topics = set()

    for n in set(x["related_topic"]):

        if (  n != np.nan and n != None and n != "" and n != "nan" and type(n) != float ):

            related_topics.add(n)



    authors = set(x["author"])

    assert len(authors) == 1, "there should be only 1 author"

    author = authors.pop()

    

    preamble_texts = set(x["preamble_text"])

    assert len(preamble_texts) == 1, "there should be only 1 preamble_texts"

    pt = preamble_texts.pop()

    preamble_text = ""

    if (type(pt) != float and pt != np.nan and pt != None and pt != "nan"):

        preamble_text = str(pt)

    preamble_text = preamble_text.strip()

    

    headings = _norm_dict(zip(x["section.id"], x["section.heading_text"]))

    paragraphs = _concat_zip(

        zip(

            x["section.id"], 

            x["section.paragraph.text"]

       )

    )



    h_p_dict = _concat_dicts(headings, paragraphs)

    content_text = _dict_to_string(h_p_dict)

    article_text = "\n\n\n".join([preamble_text, content_text]).strip()



    data = {

        "filename": fname or None,

        "title": title if title is not np.nan else None,

        "author": author if author is not np.nan else None,

        "topic": topic if topic is not np.nan else None,

        "preamble_text": preamble_text or None,

        "content_text": content_text or None,

        "related_topics": related_topics or None,

        "article_text": article_text or None,

    }

    return data



article_data = df[mask].groupby("filename").apply(_create_article_data)
article_df = pd.json_normalize(article_data)
article_df.isna().any()
article_df.size
article_df.shape
article_df = article_df.dropna()
article_df.shape
article_df.size
(

    article_df["filename"]

    .str.endswith("index.html")

).all()
article_df.head()
unique_topics = list(article_df["topic"].unique())



unique_related_topics = [

    x.replace("..//", "").replace("../", "").replace("/", "")

    for x in 

    list(article_df["related_topics"].apply(lambda x: tuple(x) if x else tuple()).explode().dropna().unique())

]
from difflib import Differ

d = Differ()


diffs_only = [

    x for x in 

    d.compare(

        sorted(unique_topics), 

        sorted(unique_related_topics),

        #sorted(unique_topics)

    )

    if x.startswith("-") or x.startswith("+")

]

pprint(diffs_only)
del df

gc.collect()
article_df.to_csv("corpus_articles.csv", index=False)
!ls -lah corp*
del article_df

gc.collect()
df = pd.read_csv("corpus_articles.csv")

df.head()
for context in KWIC(df["article_text"].values[1], "you", window_width=10):

    pass
for context in KWIC(df["article_text"].values[1], "require", window_width=10):

    pass
for context in KWIC(df["article_text"].values[1], "learn", window_width=5):

    pass
for context in KWIC(df["article_text"].values[1], "question", window_width=25):

    pass
for context in KWIC(df["article_text"].values[1], "theory", window_width=25):

    pass
for context in KWIC(df["article_text"].values[1], "rule", window_width=25):

    pass
def normalize(text):

    return preprocessing.normalize_whitespace(

        text

    )
print(df["preamble_text"].values[1][:180])

print("-----")

print(normalize(df["preamble_text"].values[1])[:180])
set(x.text for x in 

    textacy.extract.words(

        nlp(normalize(df["article_text"].values[1])), 

        min_freq=8,

    )

)
set(

    x.text for x in 

    textacy.extract.noun_chunks(

    nlp(normalize(df["article_text"].values[1])), 

    min_freq=6,

))
set(

    x.text for x in

    textacy.extract.noun_chunks(

    nlp(normalize(df["article_text"].values[1])), 

    min_freq=3,

))
list(textacy.extract.entities(

    nlp(normalize(df["article_text"].values[1])), 

    exclude_types=("NUMERIC", "ORDINAL", "MONEY", "DATE"),

))[:10]
list(textacy.extract.semistructured_statements(

    nlp(normalize(df["article_text"].values[1])), 

    entity="Abduction",

    ignore_entity_case=True,

))
list(textacy.extract.semistructured_statements(

    nlp(normalize(df["article_text"].values[1])), 

    entity="explanation",

    ignore_entity_case=True,

))
list(textacy.extract.ngrams(

    nlp(normalize(df["article_text"].values[1])), 

    2,

    min_freq=3,

    filter_stops=True, 

    filter_punct=True, 

    filter_nums=False

))
list(textacy.extract.ngrams(

    nlp(normalize(df["article_text"].values[1])), 

    3,

    min_freq=2,

    filter_stops=True, 

    filter_punct=True, 

    filter_nums=False

))[:10]
list(textacy.extract.ngrams(

    nlp(normalize(df["article_text"].values[1])), 

    3,

    min_freq=1,

    filter_stops=True, 

    filter_punct=True, 

    filter_nums=False

))[:10]
pattern = textacy.constants.POS_REGEX_PATTERNS["en"]["NP"]

pattern
list(textacy.extract.pos_regex_matches(

    nlp(normalize(df["article_text"].values[1])), 

    pattern

))[-10:]
textacy.ke.textrank(

    nlp(normalize(df["article_text"].values[1])), 



    # TextRank: 

    window_size=2, edge_weighting="binary", position_bias=False,



    normalize="lemma", 

    topn=10

)
textacy.ke.textrank(

    nlp(normalize(df["article_text"].values[1])), 



    # SingleRank:

    window_size=10, edge_weighting="count", position_bias=False,



    normalize="lemma", 

    topn=10

)
textacy.ke.textrank(

    nlp(normalize(df["article_text"].values[1])),



    # PositionRank: 

    window_size=10, edge_weighting="count", position_bias=True,



    normalize="lemma", 

    topn=10

)
textacy.ke.yake(

    nlp(normalize(df["article_text"].values[1])), 

)
textacy.ke.scake(

    nlp(normalize(df["article_text"].values[1])), 

)
# textacy.ke.sgrank(

#     nlp(normalize(df["article_text"].values[1])), 

# )
ts = textacy.TextStats(

    nlp(normalize(df["article_text"].values[1]))

)

print("ts.basic_counts:")

pprint(ts.basic_counts)

print("-----")

print("ts.readability_stats:")

pprint(ts.readability_stats)
gc.collect()
articles = df["article_text"].tolist()
del df

gc.collect()
def _yield_and_del_article(articles):

    for i in range(len(articles)):

        yield articles.pop()  # pop removes item

        gc.collect()
lst = [1,2,3]

print(lst)

g = _yield_and_del_article(lst)

print(g)

print(next(g));  print(next(g)); print(next(g))

print(lst)
corpus = textacy.Corpus("en")

print(corpus)
len(articles)
gen = _yield_and_del_article(articles)
for a in gen:

    corpus.add(a)

    print(corpus, "\r", end="")

#     if corpus.n_tokens > 9999999:

#         print("break early to conserve memory")

#         break
del articles

gc.collect()
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
vectorizer = textacy.vsm.Vectorizer(

    tf_type="linear", 

    apply_idf=True, 

    idf_type="smooth", 

    norm="l2",

    min_df=2, 

    max_df=0.95

)

vectorizer
doc_term_matrix = vectorizer.fit_transform(

    (

        doc._.to_terms_list(

            ngrams=1, 

            entities=True, 

            as_strings=True

        )

        for doc in corpus

    )

)

doc_term_matrix
N = 10 # corpus.n_docs



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
lda_model.fit(doc_term_matrix)

lda_doc_topic_matrix = lda_model.transform(doc_term_matrix)

print(lda_doc_topic_matrix.shape) 





nmf_model.fit(doc_term_matrix)

nmf_doc_topic_matrix = nmf_model.transform(doc_term_matrix)

print(nmf_doc_topic_matrix.shape)





lsa_model.fit(doc_term_matrix)

lsa_doc_topic_matrix = lsa_model.transform(doc_term_matrix)

print(lsa_doc_topic_matrix.shape)
def print_topic_terms(model, vectorizer, top_n=15):

    for topic_idx, top_terms in model.top_topic_terms(vectorizer.id_to_term, top_n=top_n):

        print("\ntopic", topic_idx, ":", "   ".join(top_terms))



print(lda_model)

print_topic_terms(lda_model, vectorizer)

print("-----"*10)

print(nmf_model)

print_topic_terms(nmf_model, vectorizer)

print("-----"*10)

print(lsa_model)

print_topic_terms(lsa_model, vectorizer)
def visualize_tm(model, doc_term_matrix, vectorizer):

    model.termite_plot(

        doc_term_matrix,

        vectorizer.id_to_term,

        topics=-1,

        n_terms=25,

        sort_terms_by="seriation"

    )
visualize_tm(

    lda_model, 

    lda_doc_topic_matrix,

    vectorizer,

)
visualize_tm(

    nmf_model,

    nmf_doc_topic_matrix,

    vectorizer,

)
import warnings
# ignore matplotlib warning

warnings.filterwarnings("ignore")
visualize_tm(

    lsa_model,

    lsa_doc_topic_matrix,

    vectorizer,

)
# undo ignore

warnings.filterwarnings("default")
del corpus

gc.collect()
gc.collect()
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
# _get_first_50_entities

# _get_semistructured_statements

# _get_first_50_bigrams_with_min_freq_2

# _get_first_50_trigrams

# _get_top_10_textrank_key_terms

# _get_top_10_singlerank_key_terms

# _get_top_10_positionrank_key_terms

# _get_top_10_yake_key_terms

# _get_top_10_scake_key_terms

# _get_top_10_sgrank_key_terms

# _get_basic_counts

# _get_readability_stats
df = pd.read_csv("corpus_articles.csv")
df.head()
# def _fun(x):

#    return normalize(x)# if type(x) == str else ""

#

#df["norm_article_text"] = df["article_text"].apply(_fun)
nlp
i = 1

def _get_the_document_features(text):

    doc = nlp(normalize(text))

    num_features = 11 #12

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
# ignore a bunch of networkx warnings

# tbh  all warnings

warnings.filterwarnings("ignore")
df = df.join(

    df.apply(

        (lambda x:_get_the_document_features(

            x[7]

        )), 

        axis=1, 

        result_type='expand'

    )

)
df.head(3)
df.columns
corpus_doc_feat_mask = [

    "filename",

    "first_50_entities",

    "first_50_bigrams_with_min_freq_2",

    "first_50_trigrams",

    "semistructured_statements",

    "top_10_textrank_key_terms",

    "top_10_singlerank_key_terms",

    "top_10_positionrank_key_terms",

    "top_10_yake_key_terms",

    "top_10_scake_key_terms",

    "basic_counts",

    "readability_stats",

]



(

    df[corpus_doc_feat_mask]

    .to_csv(

        "corpus_article_text_features.csv"

    )

)
!ls -lah corp*
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
z = zip(

   (m for m in [1,2,3,4,5]),

    (o for o in [1,2,3,4,5])

)

print(next(z))

print(next(z))

print(tuple(z))



z = zip(

   (m for m in [1,"length mismatch"]),

    (o for o in [1,2,3,4,5,6,7,8,9])

)

print(tuple(z))
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

n = len(df["article_text"])

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

    df["article_text"]

    .apply(

        _with_log(get_nlp_data(nlp))

    )

)
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
fn_mask = [

    "filename"

]



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

    *fn_mask,

    *doc_feat_mask,

    *sent_feat_mask,

]

corpus_sents_df = exploded_df[mask]
del exploded_df

del df

gc.collect()
corpus_sents_df.size
corpus_sents_df.shape
corpus_sents_df.head(1)
corpus_sents_df.to_csv(

    "corpus_sents.csv"

)
!ls -lah corp*
for_profile_df = corpus_sents_df[

    [

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

    ]

].reset_index().drop("index", axis=1)

for_profile_df.head(1)
profile = ProfileReport(for_profile_df, title='corpus_sents_df Report', html={'style':{'full_width':True}})
# profile.to_widgets()

profile.to_notebook_iframe()