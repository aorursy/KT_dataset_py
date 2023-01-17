#optionally install ujson

!pip install ujson
import functools

import itertools as it

import math

import multiprocessing as mp

import re

import types

from typing import List, Dict, Tuple, Pattern



import numpy as np

import pandas as pd

import scipy.sparse as sp



from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction import stop_words

import joblib



from IPython.display import display_html

import ipywidgets as widgets

import plotly.graph_objects as go

from plotly.subplots import make_subplots

from tqdm.notebook import tqdm

from wordcloud import WordCloud



try:

    import ujson as json 

except ImportError:

    import json 





wnl = WordNetLemmatizer()
FILE_PATTERN = "/kaggle/input/cord-19-parse-data-to-flat-format/{}-parsed.csv.gz"



datasets = [

    "custom_license",

    "noncomm_use",

    "biorxiv",

    "comm_use"

]



all_dfs = {

    ds: pd.read_csv(

        FILE_PATTERN.format(ds),

        usecols=["paper_id", "supersection","text", "source"],

        compression="gzip",

    )

    for ds in tqdm(datasets, total=len(datasets), desc="Reading Data")

}



for df in all_dfs.values():

    df["text"] = df["text"].fillna("")



all_dfs.keys()
tab = widgets.Tab(

    children = [widgets.Output() for ds in datasets],

)



for idx,(ds, out) in enumerate(zip(datasets, tab.children)):

    tab.set_title(idx, ds)

    with out:

        display_html(all_dfs[ds].head())

        display_html(all_dfs[ds].info())

        

display(tab)
def clean_text(text: str, clean_patterns: Tuple[str or Pattern, str]) -> str:

    """Clean text based on provided patterns

    

    Parameters

    ----------

    text: str

    clean_patterns: Tuple[str or Pattern, str]

        tuple of `(regex_pattern, replacement_value)`

        **Order is important**; (e.g. floats parsed first, to prevent INT.INT)

    

    Returns

    -------

    str

    """

    clean_text = text

    for pattern, new_val in clean_patterns:

        if isinstance(pattern, Pattern):

            clean_text = pattern.sub(new_val, clean_text)

        else:

            clean_text = re.sub(pattern, new_val, clean_text)



    return clean_text





# list of (regex, new value) tuples

# order is important

# text converted to lowercase before cleaning 

CLEAN_PATTERNS = (

    (r"(?<=[A-Za-z])'(?=[A-Za-z])", ""),  # remove contractions

    (r"\b[atcgu]{3,}\b", " DNASEQ "),  # not sure what the min char threshold should be...

    (r"http\S+", " URL "),

    ("(?<!\d)2019(?!\d)", " COVIDYEAR "),  # might be important?

    (r"(?<!\d)\d+\.\d+(?!\d)", " FLOAT "),

    (r"(?<!\d)\d+(?!\d)", " INT "),

    (r"[\s_]+", " "),

)
# compile regex patterns

comp_cleanpat = tuple(

    (re.compile(pat, flags=re.I), new_val) for pat, new_val in CLEAN_PATTERNS

)



# note on RAM usage: +~3 GB to peak memory usage, adds +2 GB once done with all datasets

for ds in tqdm(datasets, desc="Cleaning texts..."):

    df = all_dfs[ds]

    with mp.Pool() as pool:

        df["text_clean"] = list(

            tqdm(

                pool.imap(

                    functools.partial(clean_text, clean_patterns=comp_cleanpat),

                    df["text"].str.lower(),

                ),

                total=len(df),

                desc=ds,

            )

        )

# Texts by paper_id

texts_by_paperid = {

    ds: all_dfs[ds].groupby(["paper_id"])["text_clean"].apply(lambda x: " ".join(x)).values

    for ds in tqdm(datasets, desc="Combining by `paper_id`")

}
all_texts = np.concatenate([texts for texts in texts_by_paperid.values()])



vectorizer = CountVectorizer(stop_words="english", analyzer="word", lowercase= False)

_ = vectorizer.fit(all_texts)
@functools.wraps(CountVectorizer.transform)

def transform_parallel(self, raw_documents):

    with mp.Pool() as pool:

        docs_split = np.array_split(raw_documents, mp.cpu_count())

        X = sp.vstack(pool.map(self.transform, docs_split), format='csr')



    return X



vectorizer.transform_parallel = types.MethodType(transform_parallel, vectorizer)



joblib.dump(vectorizer, "CORD19_countvec.joblib")  
def lemmatize(word: str, pos_tags: list= None):

    """`nltk` WordNet-based Lemmatization, regardless of part-of-speech

    

    Parameters

    ----------

    word: str

    pos_values: list, optional

        Part of Speech tags to attempt

        defaults to `["n", "v", "a"]`

    

    Returns

    -------

    str

        lemmatized word

    """

    pos_tags = pos_tags or ["n", "v", "a"]    

    lemms = [wnl.lemmatize(word, pos) for pos in pos_tags]

    lemms = [lemm for lemm in lemms if lemm != word]

    if lemms:

        lemm_lens = [len(lemm) for lemm in lemms]

        return lemms[lemm_lens.index(min(lemm_lens))]

    else:

        return word





def construct_lemm_matrix(words: List[str], as_df=False):

    """Construct a Lemmatization Matrix

    

    Parameters

    ----------

    words: list[str]

    as_df: boolean, optional

        return output as `pandas.DataFrame`, defaults to `False`



    Returns

    -------

    dict or pandas.DataFrame

        if `as_df` returns a pandas.DataFrame

        else {"matrix": sp.csr_matrix, "columns": list[str], "vocabulary": dict[str, int]}

    """

    col_idx, columns = [], []

    lemm_vocab = {}

    for word in words:

        lemm = lemmatize(word)

        if lemm not in lemm_vocab:

            lemm_vocab[lemm] = len(lemm_vocab)

            columns.append(lemm)



        col_idx.append(lemm_vocab[lemm])



    no_of_words = len(words)

    res = {

        "matrix": sp.csr_matrix((np.ones(no_of_words), (np.arange(no_of_words), col_idx))),

        "columns": columns,

        "vocabulary": lemm_vocab,

    }

    

    return pd.DataFrame(res["matrix"].toarray(), columns=res["columns"], index=words) if as_df else res





L = construct_lemm_matrix(vectorizer.get_feature_names())
def gen_wordcount(X, words):

    # sparse matrices are awesome...

    wordcount = pd.DataFrame(

        np.concatenate([X.sum(axis=0), (X > 0).sum(axis=0)], axis=0),

        columns=pd.Index(words, name="word"),

        index=["count", "count_of_doc"]

    ).T



    wordcount["percent_of_doc"] = wordcount["count_of_doc"] / X.shape[0] 

    return wordcount



raw_dt = {ds: vectorizer.transform_parallel(texts) for ds, texts in tqdm(texts_by_paperid.items(), desc="Transforming")}

lemm_dt = {ds: X @ L["matrix"] for ds, X in raw_dt.items()}



# change this line to use raw or lemmatized document term matrix

wordcounts = {ds: gen_wordcount(X, L["columns"]) for ds, X in lemm_dt.items()}



tab = widgets.Tab(

    children = [widgets.Output() for _ in texts_by_paperid],

)

for idx,(out, (ds, wordcount)) in enumerate(zip(tab.children, wordcounts.items())):

    tab.set_title(idx, ds)

    with out:

        display_html(wordcount.describe())

    

display(tab)
def plot_wordcounts(wordcounts: Dict[str, pd.DataFrame], filter_func: callable=None, wc_args: dict = None):    

    wc_args = {**{"width":800, "height": 500}, **(wc_args or {})}

    tab = widgets.Tab(

        children = [widgets.Output() for _ in wordcounts],

    )

    for idx, (out, (ds,wordcount)) in enumerate(zip(tab.children, wordcounts.items())):

        if filter_func:

            wc_df = filter_func(wordcount)

        else:

            wc_df = wordcount



        tab.set_title(idx, ds)

        with out:

            display(

                WordCloud(**wc_args).generate_from_frequencies(

                    wc_df["count"].to_dict()

                ).to_image()

            )

    display(tab)
plot_wordcounts(wordcounts)
COLUMNS = 3

xaxis, yaxis = "% of docs", "% of words (cumulative)"



plot_titles = list(wordcounts.keys())



fig = make_subplots(

    rows=math.ceil(len(plot_titles)/ COLUMNS), cols=COLUMNS,

    subplot_titles=plot_titles,

    horizontal_spacing=0.05, vertical_spacing=0.1

)



for idx, ds in enumerate(plot_titles):

    wordcount = wordcounts[ds]

    row_i, col_i = int(idx / COLUMNS) + 1, idx % COLUMNS + 1

    hist_data = pd.DataFrame(np.histogram(wordcount["percent_of_doc"], bins=50), index=["count", "% of docs"]).T

    hist_data["% of words (cumulative)"] = hist_data["count"].cumsum() / hist_data["count"].sum()

    fig.add_trace(

        go.Scatter(x=hist_data[xaxis], y=hist_data[yaxis], mode="lines+markers"),

        row=row_i, col=col_i

    )



fig.update_layout(

    height=400 * row_i, width=1200, title_text="Word Concentration by Document Occurence", showlegend=False

)

fig.update_xaxes(title_text=xaxis, row=2, col=1)

fig.update_yaxes(title_text=yaxis, row=2, col=1)



fig.show()
docpct_maxthresh = 0.7
plot_wordcounts(

    wordcounts, filter_func=lambda x: x[x["percent_of_doc"] >= docpct_maxthresh]

)
wc_df = wordcounts["biorxiv"]

wc_df[

    (wc_df["percent_of_doc"] >= 0.9) & (wc_df.index.str.upper() != wc_df.index)

].sort_values("count_of_doc", ascending=False)

biorxiv_df = all_dfs["biorxiv"]

with pd.option_context('display.max_colwidth', -1):

    display(biorxiv_df.loc[biorxiv_df["text_clean"].str.contains("preprint", flags=re.I),["text"]].head())
addl_stop_words = {

    "biorxiv": {

        "stop_sentence": "The copyright holder for this preprint, which was not peer-reviewed, is the author or funder.",

        "stop_words": [

            "license",

            "copyright",

            "holder",

            "preprint",

            "peer",

            "reviewed",

            "author",

            "funder",

            "doi",

            "biorxiv",

            "medrxiv",

            "preprint",

        ],

    }

}



json.dump(addl_stop_words, open("addl_stop_words.json", "w"))