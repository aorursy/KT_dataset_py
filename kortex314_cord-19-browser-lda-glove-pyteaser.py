!pip install MulticoreTSNE-modified



import os

import gc

import re

import warnings

from itertools import chain

from collections import Counter



import textwrap



from math import fabs

import numpy as np

import pandas as pd



from nltk.stem import WordNetLemmatizer

from nltk.tokenize import sent_tokenize

from nltk.corpus import words, stopwords



from gensim.corpora import Dictionary

from gensim.models import LdaMulticore, CoherenceModel



from sklearn.metrics.pairwise import cosine_similarity

from sklearn.decomposition import TruncatedSVD



from MulticoreTSNE import MulticoreTSNE as TSNE



import matplotlib.cm as cm



from pyLDAvis.gensim import prepare as prepare_gensim

import pyLDAvis



import plotly

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot



import ipywidgets as widgets

from IPython.html.widgets import interactive

from IPython.core.display import display, HTML





%matplotlib inline





warnings.simplefilter("ignore", DeprecationWarning)

pd.set_option("max_colwidth", 100)

pd.set_option("max_columns", 25)

pd.set_option("max_rows", 25)

init_notebook_mode(connected=True)
# Filesystem

INP_ROOT = "/kaggle/input"

INP_DIR = os.path.join(INP_ROOT, "CORD-19-research-challenge")

META_FILE = os.path.join(INP_DIR, "metadata.csv")

OUT_DIR = "/kaggle/working/"



# Execution ontrol

RANDOM_STATE = 214

N_JOBS = 8



# NLP variables

LEMMATIZER = WordNetLemmatizer()

REMOVE_SENTENCES = [

    "publicly funded repositories, such as the WHO COVID database with rights for unrestricted research re-use and analyses in any form or by any means with acknowledgement of the original source.",

    "These permissions are granted for free by Elsevier for as long as the COVID-19 resource centre remains active.",

]  # sentences to remove

EN_STOPWORDS = list(stopwords.words("english"))

EN_WORDS = words.words("en")

NUMBERS = [

    'zero',

    'one',

    'two',

    'three',

    'four',

    'five',

    'six',

    'seven',

    'eight',

    'nine',

    'ten',

    'eleven',

    'twelve',

    'thirteen',

    'fourteen',

    'fifteen',

    'sixteen',

    'seventeen',

    'eighteen',

    'nineteen',

    'twenty',

    'twenty-one',

    'twenty-two',

    'twenty-three',

    'twenty-four',

    'twenty-five',

    'twenty-six',

    'twenty-seven',

    'twenty-eight',

    'twenty-nine',

    'thirty',

    'thirty-one',

    'thirty-two',

    'thirty-three',

    'thirty-four',

    'thirty-five',

    'thirty-six',

    'thirty-seven',

    'thirty-eight',

    'thirty-nine',

    'forty',

    'forty-one',

    'forty-two',

    'forty-three',

    'forty-four',

    'forty-five',

    'forty-six',

    'forty-seven',

    'forty-eight',

    'forty-nine',

    'fifty',

    'fifty-one',

    'fifty-two',

    'fifty-three',

    'fifty-four',

    'fifty-five',

    'fifty-six',

    'fifty-seven',

    'fifty-eight',

    'fifty-nine',

    'sixty',

    'sixty-one',

    'sixty-two',

    'sixty-three',

    'sixty-four',

    'sixty-five',

    'sixty-six',

    'sixty-seven',

    'sixty-eight',

    'sixty-nine',

    'seventy',

    'seventy-one',

    'seventy-two',

    'seventy-three',

    'seventy-four',

    'seventy-five',

    'seventy-six',

    'seventy-seven',

    'seventy-eight',

    'seventy-nine',

    'eighty',

    'eighty-one',

    'eighty-two',

    'eighty-three',

    'eighty-four',

    'eighty-five',

    'eighty-six',

    'eighty-seven',

    'eighty-eight',

    'eighty-nine',

    'ninety',

    'ninety-one',

    'ninety-two',

    'ninety-three',

    'ninety-four',

    'ninety-five',

    'ninety-six',

    'ninety-seven',

    'ninety-eight',

    'ninety-nine',

    'hundred',

]

ORDINALS = [

    'zeroth',

    'first',

    'second',

    'third',

    'fourth',

    'fifth',

    'sixth',

    'seventh',

    'eighth',

    'ninth',

    'tenth',

    'eleventh',

    'twelfth',

    'thirteenth',

    'fourteenth',

    'fifteenth',

    'sixteenth',

    'seventeenth',

    'eighteenth',

    'nineteenth',

    'twentieth',

    'twenty-first',

    'twenty-second',

    'twenty-third',

    'twenty-fourth',

    'twenty-fifth',

    'twenty-sixth',

    'twenty-seventh',

    'twenty-eighth',

    'twenty-ninth',

    'thirtieth',

    'thirty-first',

    'thirty-second',

    'thirty-third',

    'thirty-fourth',

    'thirty-fifth',

    'thirty-sixth',

    'thirty-seventh',

    'thirty-eighth',

    'thirty-ninth',

    'fortieth',

    'forty-first',

    'forty-second',

    'forty-third',

    'forty-fourth',

    'forty-fifth',

    'forty-sixth',

    'forty-seventh',

    'forty-eighth',

    'forty-ninth',

    'fiftieth',

    'fifty-first',

    'fifty-second',

    'fifty-third',

    'fifty-fourth',

    'fifty-fifth',

    'fifty-sixth',

    'fifty-seventh',

    'fifty-eighth',

    'fifty-ninth',

    'sixtieth',

    'sixty-first',

    'sixty-second',

    'sixty-third',

    'sixty-fourth',

    'sixty-fifth',

    'sixty-sixth',

    'sixty-seventh',

    'sixty-eighth',

    'sixty-ninth',

    'seventieth',

    'seventy-first',

    'seventy-second',

    'seventy-third',

    'seventy-fourth',

    'seventy-fifth',

    'seventy-sixth',

    'seventy-seventh',

    'seventy-eighth',

    'seventy-ninth',

    'eightieth',

    'eighty-first',

    'eighty-second',

    'eighty-third',

    'eighty-fourth',

    'eighty-fifth',

    'eighty-sixth',

    'eighty-seventh',

    'eighty-eighth',

    'eighty-ninth',

    'ninetieth',

    'ninety-first',

    'ninety-second',

    'ninety-third',

    'ninety-fourth',

    'ninety-fifth',

    'ninety-sixth',

    'ninety-seventh',

    'ninety-eighth',

    'ninety-ninth',

    'hundredth',

]

MONTHS = [

    "apr",

    "april",

    "aug",

    "august",

    "dec",

    "december",

    "feb",

    "february",

    "jan",

    "january",

    "jul",

    "july",

    "jun",

    "june",

    "mar",

    "march",

    "may",

    "nov",

    "november",

    "oct",

    "october",

    "sep",

    "sept",

    "september",

    # es:

    "enero",

    "febrero",

    "marzo",

    "abril",

    "mayo",

    "junio",

    "julio",

    "agosto",

    "septiembre",

    "octubre",

    "noviembre",

    "diciembre",

    # en:

    "january",

    "february",

    "march",

    "april",

    "may",

    "june",

    "july",

    "august",

    "september",

    "october",

    "november",

    "december",

    # de:

    "januar",

    "februar",

    "märz",

    "maerz",

    "april",

    "mai",

    "juni",

    "juli",

    "august",

    "september",

    "oktober",

    "november",

    "dezember"

    # fr:

    "janvier",

    "février",

    "fevrier",

    "mars",

    "avril",

    "mai",

    "juin",

    "juillet",

    "août",

    "aout",

    "septembre",

    "octobre",

    "november",

    "décembre",

    "decembre",

]

# Word cleaning / replacement

WORDS_TO_REMOVE = [

    "doi",

    "preprint",

    "copyright",

    "peer",

    "reviewed",

    "review",

    "org",

    "https",

    "http",

    "et",

    "al",

    "author",

    "figure",

    "rights",

    "reserved",

    "permission",

    "used",

    "using",

    "biorxiv",

    "fig",

    "text",

    "full",

    "pubmed",

    "abstract",

    "publisher",

    "free",

    "background",

    "however",

    "including",

    "important",

    "among",

    "conclusion",

    "discussion",

    "compare",

    "compared",

    "also",

    "appendix",

    "discussion",

    "conclusion",

    "results",

    "result",

    "conclusion",

    "keyword",

    "yet",

    "key",

    "use",

    "used",

    "using",

    "end",

    "new",

    "found",

    "show",

    "showed",

    "showing",

    "research",

    "publication",

    "article",

    "published",

    "journal",

    "scientific",

    "effort",

    "efforts",

    "a", "e", "i", "o", "u", "t", "about", "above",

    "above", "across", "after", "afterwards", "again", "against", "all",

    "almost", "alone", "along", "already", "also", "although", "always",

    "am", "among", "amongst", "amoungst", "amount", "an", "and",

    "another", "any", "anyhow", "anyone", "anything", "anyway",

    "anywhere", "are", "around", "as", "at", "back", "be", "became",

    "because", "become", "becomes", "becoming", "been", "before",

    "beforehand", "behind", "being", "below", "beside", "besides",

    "between", "beyond", "both", "bottom", "but", "by", "call", "can",

    "cannot", "can't", "co", "con", "could", "couldn't", "de",

    "did", "do", "done", "down", "due", "during",

    "each", "eg", "eight", "either", "eleven", "else", "elsewhere",

    "empty", "enough", "etc", "even", "ever", "every", "everyone",

    "everything", "everywhere", "except", "few", "fifteen", "fifty",

    "fill", "fire", "first", "five", "for", "former",

    "formerly", "forty", "found", "four", "from", "front", "full",

    "further", "get", "give", "go", "got", "had", "has", "hasnt",

    "have", "he", "hence", "her", "here", "hereafter", "hereby",

    "herein", "hereupon", "hers", "herself", "him", "himself", "his",

    "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed",

    "into", "is", "it", "its", "it's", "itself", "just", "keep", "last",

    "latter", "latterly", "least", "less", "like", "ltd", "made", "make",

    "many", "may", "me", "meanwhile", "might", "mill", "mine", "more",

    "moreover", "most", "mostly", "move", "much", "must", "my", "myself",

    "name", "namely", "neither", "never", "nevertheless", "new", "next",

    "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing",

    "now", "nowhere", "of", "off", "often", "on", "once", "one", "only",

    "onto", "or", "other", "others", "otherwise", "our", "ours",

    "ourselves", "out", "over", "own", "part", "per",

    "perhaps", "please", "put", "rather", "re", "said", "same", "see",

    "seem", "seemed", "seeming", "seems", "several", "she", "should",

    "side", "since", "sincere", "six", "sixty", "so", "some",

    "somehow", "someone", "something", "sometime", "sometimes",

    "somewhere", "still", "such", "take", "ten", "than", "that", "the",

    "their", "them", "themselves", "then", "thence", "there",

    "thereafter", "thereby", "therefore", "therein", "thereupon",

    "these", "they", "thickv", "thin", "third", "this", "those",

    "though", "three", "through", "throughout", "thru", "thus", "to",

    "together", "too", "top", "toward", "towards", "twelve", "twenty",

    "two", "un", "under", "until", "up", "upon", "us", "very",

    "via", "want", "was", "we", "well", "were", "what", "whatever",

    "when", "whence", "whenever", "where", "whereafter", "whereas",

    "whereby", "wherein", "whereupon", "wherever", "whether", "which",

    "while", "whither", "who", "whoever", "whole", "whom", "whose",

    "why", "will", "with", "within", "without", "would", "yet", "you",

    "your", "yours", "yourself", "yourselves", "the",

    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",

    "sunday", "mon", "tue", "wed", "thu", "fri", "sat", "sun",

]

WORDS_TO_REPLACE = {}
# Topic modeling

N_TOPICS = 20

LDA_CONFIG = {

    "num_topics": N_TOPICS,

    "workers": N_JOBS,

    "random_state": RANDOM_STATE,

    "passes": 250,

    "decay": 0.4,

    "iterations": 50,

    "chunksize": 100,

}



# Embeddings

GLOVE_NDIMS = 200

GLOVE_FILE = os.path.join(INP_ROOT, "glove6b", f"glove.6B.{GLOVE_NDIMS}d.txt")



# t-SNE visualization

TSNE_PERPLEXITY = 50

TSNE_N_ITER = 1200



# Summarization

IDEAL_LENGTH = 20

FINDINGS_KEYWORDS = set([

    "results",

    "show",

    "find",

    "conclusion",

    "major",

    "finding",

    "goal",

])
# Utils functions for data cleaning



def pre_clean_sentences(text_series: pd.Series) -> pd.Series:

    """Cleans sentence separators"""

    replacement_dict = {

        " .": ". ",

    }

    text_series = text_series.fillna("")

    for key, value in replacement_dict.items():

        text_series = text_series.str.replace(key, value, regex=False)

    return text_series



def get_words_to_remove(

    base_remove_words: list,

    remove_numbers_and_ordinals: bool = True,

    remove_months: bool = True,

    remove_stopwords: bool = True,

    ) -> list:

    remove_words = list(base_remove_words)

    if remove_numbers_and_ordinals:

        remove_words += NUMBERS + ORDINALS



    if remove_months:

        remove_words += MONTHS



    if remove_stopwords:

        remove_words += EN_STOPWORDS

    return remove_words



def clean_paper_wordlist(

    wordlist: list,

    remove_words: list = None,

    remap_words: dict = None,

    ) -> str:

    """Changes / removes words from a list



    Parameters

        wordlist (list): list of words to clean

        remove_words (list): list of words to remove

        remap_words (dict): dictionary mapping words to replacements (can include several words)



    Returns

        str: cleaned words

    """

    # replace words, lemmatize, and re-split in case of remapping to multiple words

    words = " ".join([LEMMATIZER.lemmatize(remap_words.get(word, word)) for word in wordlist]).split()



    # remove words from a list and too short words

    words = [word for word in words if len(word) > 2 and word not in remove_words]

    return " ".join(words)





def clean_sentences(

    sentences: pd.Series,

    remove_words: list = None,

    remap_words: dict = None,

    ) -> list:

    """Cleans a pd.Series column containing sentences (list)



    Parameters:

        sentences (pd.Series)

        remove_words (list): full list of words to remove (see other remove_* parameters)

        remap_words (dict): dictionary mapping words to replacements (can include several words)

    """

    sentences_cleaned = []

    # As inefficient as it gets...

    for sentence in sentences:

        if sentence in REMOVE_SENTENCES:

            sentences_cleaned.append("")

            continue



        cleaned_wordlists = [

            clean_paper_wordlist(

                re.split("[^a-zA-Z]+", word.lower()),

                remove_words=remove_words,

                remap_words=remap_words,

            ).split()

            for word in sentence.split()

        ]

        sentences_cleaned.append(" ".join(chain.from_iterable(cleaned_wordlists)))

    return sentences_cleaned



def clean_column(

    df: pd.DataFrame,

    column: str = "text",

    remove_words: list = None,

    remap_words: dict = None,

    remove_numbers_and_ordinals: bool = True,

    remove_months: bool = True,

    remove_stopwords: bool = True,

    ) -> pd.DataFrame:

    """Cleans a text column



    Parameters:

        df (pd.DataFrame)

        column (str): column to clean

        remove_words (list): base list of words to remove (see other remove_* parameters)

        remap_words (dict): dictionary mapping words to replacements (can include several words)

        remove_numbers_and_ordinals (bool): remove string numbers ("one", "two", ...) and ordinals ("first", "second", ...)

        remove_months (bool): remove month names & acronyms

        remove_stopwords (str): remove english stopwords

    """

    remove_words = remove_words or []

    remap_words = remap_words or {}



    remove_words = get_words_to_remove(

        remove_words,

        remove_numbers_and_ordinals=remove_numbers_and_ordinals,

        remove_months=remove_months,

        remove_stopwords=remove_stopwords,

    )



    cleaned_column = f"{column}_cleaned"

    df[cleaned_column] = df[column].fillna("").str.lower()

    # Split to get words

    df[cleaned_column] = df[cleaned_column].str.split("[^a-zA-Z]+")



    if column not in ["title", "section"]:

        # Word-by-word cleaning & remapping, too agressive for titles & section names

        df[cleaned_column] = df[cleaned_column].apply(

            clean_paper_wordlist,

            remove_words=remove_words,

            remap_words=remap_words,

        )

    else:

        df[cleaned_column] = df[cleaned_column].apply(lambda x: " ".join(x))



    return df



def clean_sentences_column(

    df: pd.DataFrame,

    column: str = "text",

    remove_words: list = None,

    remap_words: dict = None,

    remove_numbers_and_ordinals: bool = True,

    remove_months: bool = True,

    remove_stopwords: bool = True,

    ) -> pd.DataFrame:

    """Cleans a text column



    Parameters:

        df (pd.DataFrame)

        column (str): column to clean

        remove_words (list): base list of words to remove (see other remove_* parameters)

        remap_words (dict): dictionary mapping words to replacements (can include several words)

        remove_numbers_and_ordinals (bool): remove string numbers ("one", "two", ...) and ordinals ("first", "second", ...)

        remove_months (bool): remove month names & acronyms

        remove_stopwords (str): remove english stopwords

    """

    # Get the full list of words to remove

    remove_words = get_words_to_remove(

        remove_words,

        remove_numbers_and_ordinals=remove_numbers_and_ordinals,

        remove_months=remove_months,

        remove_stopwords=remove_stopwords,

    )



    cleaned_column = f"{column}_cleaned"

    df[cleaned_column] = df[column].apply(

        clean_sentences,

        remove_words=remove_words,

        remap_words=remap_words,

    )

    return df



def filter_duplicates(

    df: pd.DataFrame,

    on="title_cleaned",

    using="abstract_cleaned",

    ) -> pd.DataFrame:

    """Drops papers with duplicated cleaned title, prioritizing non-empty abstracts

    """

    df = df.sort_values(by=[on, using])

    duplicated = df.duplicated(subset=[on], keep="last")

    #print("Dropping {} entries ({:.2%}) which are duplicates according to {}".format(duplicated.sum(), duplicated.mean(), on))

    df = df[~duplicated].sort_index()

    return df



def remove_non_english_papers(

    df: pd.DataFrame,

    column="abstract_cleaned",

    max_en_percentage=0.5,

    min_stopword_percentage=0.04,

    ):

    """Removes rows if their ratio of english words is below `max_en_percentage`, and if the ratio

    of stopwords in another language is above `min_stopword_percentage`.

    (nltk gives english words but only stopwords in other tongues)



    If `return_percentages` is True, doesn't drop anything and returns the Series of ratio of

    english words per entry instead.



    This could all be based on a language detection / translation model...

    """

    assert max_en_percentage <= 1



    mapping = {

        "fr": "french",

        "it": "italian",

        "es": "spanish",

        "ger": "german",

        "ru": "russian",

        "pt": "portuguese",

        "gk": "greek",

    }

    lngs = ["fr", "it", "es", "ger", "ru", "pt", "gk"]

    foreign_tongue_words = {

        lng: stopwords.words(mapping[lng])

        for lng in lngs

    }



    en_words = {w.lower() for w in EN_WORDS}



    def percentage_words(text, lng_words=None):

        splitted = text.split()

        if splitted:

            return len(set(splitted).intersection(set(lng_words))) / len(splitted)

        else:

            return 0



    en_percentages = df[column].apply(percentage_words, lng_words=en_words)

    foreign_percentages = {}

    for lng, lng_words in foreign_tongue_words.items():

        foreign_percentages[lng] = df[column].apply(percentage_words, lng_words=lng_words)



    to_drop = en_percentages > 1  # all false

    for lng, percentages in foreign_percentages.items():

        to_drop |= percentages < min_stopword_percentage

    to_drop &= en_percentages < max_en_percentage



    print("Dropping {} rows with a ratio of english words < {:.2f}".format(to_drop.sum(), max_en_percentage))

    return df[~to_drop]



def filter_paper_lengths(df, column="abstract", min_length=10, max_length=1000):

    """Returns the DataFrame where the length of the `column` is between min_length and max_length (both included)

    """

    length = df[column].str.split().apply(len)

    valid = length <= max_length

    valid &= length >= min_length

    print("Removing %i rows with length < %i or > %i" % ((~valid).sum(), min_length, max_length))

    return df[valid]
%%time



df = pd.read_csv(META_FILE)



for column in df.columns:

    if df[column].dtype == object:

        df[column].fillna("", inplace=True)



# Drop papers with empty titles and empty abstracts

empty = (df["title"] == "") | (df["abstract"] == "")

print("Dropping {} papers out of {} ({:.2%})".format(empty.sum(), len(empty), empty.mean()))

df = df[~empty]

gc.collect()



# Create sentences

df["abstract"] = pre_clean_sentences(df["abstract"])

df["sentences"] = df["abstract"].apply(sent_tokenize)

df = clean_sentences_column(df, "sentences", remove_words=WORDS_TO_REMOVE, remap_words=WORDS_TO_REPLACE)

df["abstract_cleaned"] = df["sentences_cleaned"].apply(lambda sentences_list: " ".join(sentences_list))

df = clean_column(df, "title", remove_words=WORDS_TO_REMOVE, remap_words=WORDS_TO_REPLACE)

df = df[sorted(df.columns)]

gc.collect()

df.head()
df = filter_duplicates(df)

df = remove_non_english_papers(df)

df = filter_paper_lengths(df)



df["length"] = df["abstract_cleaned"].str.split().apply(len)  # not ideal but we'll improve later on

df["length"].hist(bins=50, figsize=(12, 6))
%%time



# Prepare data for gensim LDA

texts = df["abstract_cleaned"].str.split().tolist()

id2word = Dictionary(texts)

corpus = [id2word.doc2bow(text) for text in texts]

gc.collect()



# Compute the LDA

print("Initializing LDA model")

model = LdaMulticore(

    corpus=corpus,

    id2word=id2word,

    **LDA_CONFIG,

)



# Apply the model: create topic distributions per paper

print("Applying LDA model")

topic_columns = ["topic_%i" % (i + 1) for i in range(N_TOPICS)]

labels_series = df["abstract_cleaned"].apply(lambda text: model[id2word.doc2bow(text.split())])



for column in topic_columns:

    df[column] = 0.



for (i, label) in zip(labels_series.index, labels_series.values):

    if label:

        for j, proba in label:

            df.loc[i, topic_columns[j]] = proba



# Show the topic descriptions

print("Showing topic descriptions")

viz = prepare_gensim(

    model,

    corpus,

    id2word,

    sort_topics=False,

    n_jobs=N_JOBS,

    mds="tsne",

)

pyLDAvis.display(viz)
def weights_to_desc(weight_str):

    terms = [w.split("*", 1)[1].strip()[1:-1] for w in weight_str.split(" + ")]

    return " ".join(terms)



# Get the words carrying the highest "weight" per topic

topics = model.print_topics(num_topics=N_TOPICS, num_words=6)

topic_weights = {topic[0] + 1: topic[1] for topic in topics}



topic_descriptions = {

    topic_idx: weights_to_desc(topic_desc)

    for topic_idx, topic_desc in topic_weights.items()

}



# A topic distribution is now its heaviest N words

df_descriptions = pd.Series(topic_descriptions).to_frame("Topic description")

df_descriptions.index.name = "Topic number"



# We add the most "specialized" documents per topic: documents with the highest ratio for each topic

top_doc_columns = ["document_%i" % (i + 1) for i in range(3)]

for column in top_doc_columns:

    df_descriptions[column] = ""



for topic_idx, topic_col in enumerate(topic_columns, 1):

    if df[topic_col].sum() == 0:

        # Empty topic

        continue



    # Most specialised documents per topic

    documents = df.nlargest(3, topic_col)["title"]

    df_descriptions.loc[topic_idx, top_doc_columns] = documents.values



df_descriptions.to_excel(os.path.join(OUT_DIR, "topics.xlsx"))

df_descriptions
# Compute coherence metrics

coherences = {}

for coherence_type in ['c_v', 'c_uci', 'c_npmi']:

    coherence_model_lda = CoherenceModel(

        model=model,

        texts=texts,

        dictionary=id2word,

        coherence=coherence_type,

    )

    coherences[coherence_type] = coherence_model_lda.get_coherence()

coherences
def plot_features_3d(coords, y=None, texts=None, fname=None):

    classes = sorted(set(y))

    colors = [cm.viridis(n) for n in np.linspace(0, 1, len(classes))]

    marker_dict = {

        "size": 3,

        "line": {

            "width": 0,

        },

        "opacity": 0.5,

        "symbol": "circle",

    }

    

    data = []

    for class_name, color in zip(classes, colors):

        boo = y == class_name

        

        marker_class = marker_dict.copy()

        rgb = ["%i" % (c * 256) for c in color[:3]]

        marker_class["color"] = 'rgb(%s)' % ",".join(rgb)

        

    

        trace = go.Scatter3d(

            x=coords[boo, 0],

            y=coords[boo, 1],

            z=coords[boo, 2],

            mode='markers',

            marker=marker_class,

            name=class_name,

            text=texts[boo],

        )

        data.append(trace)

        

    layout = go.Layout(

        margin=dict(

            l=0,

            r=0,

            b=0,

            t=0,

        )

    )

    

    

    fig = go.Figure(data=data, layout=layout)

    return iplot(fig)

    

tsne_dims = 3

tsne_columns = ["tSNE%i" % (i + 1) for i in range(tsne_dims)]



tsne = TSNE(

    n_components=tsne_dims,

    perplexity=TSNE_PERPLEXITY,

    random_state=RANDOM_STATE,

    n_jobs=N_JOBS,

    n_iter=TSNE_N_ITER,

)



# Useful to visually browse through the titles of papers with similar topic distributions

    

# Each document is represented by a topic distribution vector, summing to 1.

# We transform these distributions to create coordinates, allocating documents on a hypersphere.



df_coords = np.sqrt(df[topic_columns])

df_plot = pd.DataFrame(

    data=tsne.fit_transform(df_coords.values),

    index=df.index,

    columns=tsne_columns,

)



# Add the dominant topic

df_plot["dominant_topic"] = df[topic_columns].apply("idxmax", axis=1)



# dominant_topic is of type "Topic {n}"

df_plot["dominant_topic"] = df_plot["dominant_topic"].str.split("_").str[1].apply(int)

topic_str = df_plot["dominant_topic"].map(df_descriptions["Topic description"])



df_plot["dominant_topic"] = df_plot["dominant_topic"].apply(lambda x: "%02d" % x) + ": " + topic_str

gc.collect()



def wrap_text(text, width=75):

    wrapper = textwrap.TextWrapper(width=width)

    return "<br>".join(wrapper.wrap(text))



df_plot["text"] = df_plot["dominant_topic"].values + " <br> " + df["title"]

plot_features_3d(

    df_plot[tsne_columns].values,

    y=df_plot["dominant_topic"].apply(wrap_text, width=30).values,

    texts=df_plot["text"].apply(wrap_text, width=75),

)
df["count"] = np.arange(len(df))

topics_similarity = cosine_similarity(df[topic_columns])



def get_closest_papers(paper_i=0, num=2):

    similarities = topics_similarity[paper_i]

    paper_j = np.argsort(similarities)[::-1]

    return paper_j[1: 1 + num]
def keywords(text):

    """get the top 10 keywords and their frequency scores

    ignores blacklisted words in stopWords,

    counts the number of occurrences of each word

    """

    text = split_words(text)

    numWords = len(text)  # of words before removing blacklist words

    # freq = Counter(x for x in text if x not in stopWords)

    freq = Counter(text)



    minSize = min(10, len(freq))  # get first 10

    keywords = {x: y for x, y in freq.most_common(minSize)}  # recreate a dict



    for k in keywords:

        articleScore = keywords[k] * 1.0 / numWords

        keywords[k] = articleScore * 1.5 + 1



    return keywords





def length_score(sentence):

    return 1 - fabs(IDEAL_LENGTH - len(sentence)) / IDEAL_LENGTH





def title_score(title, sentence):

    title_words = set(title) - set(EN_STOPWORDS)

    if len(title_words) == 0:

        return 0.5



    sentence_words = set(sentence) - set(EN_STOPWORDS + WORDS_TO_REMOVE)

    score = len(sentence_words.union(title_words)) / len(title_words)

    if score == 1:

        return 0

    else:

        return score





def sentence_position(i, size):

    """different sentence positions indicate different

    probability of being an important sentence"""



    normalized = i * 1.0 / size

    if 0 < normalized <= 0.1:

        return 0.17

    elif 0.1 < normalized <= 0.2:

        return 0.23

    elif 0.2 < normalized <= 0.3:

        return 0.14

    elif 0.3 < normalized <= 0.4:

        return 0.08

    elif 0.4 < normalized <= 0.5:

        return 0.05

    elif 0.5 < normalized <= 0.6:

        return 0.04

    elif 0.6 < normalized <= 0.7:

        return 0.06

    elif 0.7 < normalized <= 0.8:

        return 0.04

    elif 0.8 < normalized <= 0.9:

        return 0.04

    elif 0.9 < normalized <= 1.0:

        return 0.15

    else:

        return 0



def split_words(text):

    """split a string into array of words"""

    return text.split()



def findings_score(sentence):

    return len(set(sentence).union(FINDINGS_KEYWORDS)) + 1





def sbs(words, keywords):

    score = 0.0

    if len(words) == 0:

        return 0



    score = sum([keywords.get(word, 0.) for word in words])

    return (1.0 / fabs(len(words)) * score) / 10.0





def dbs(words, keywords):

    if len(words) == 0:

        return 0



    summ = 0

    first = []

    second = []



    for i, word in enumerate(words):

        if word in keywords:

            score = keywords[word]

            if first == []:

                first = [i, score]

            else:

                second = first

                first = [i, score]

                dif = first[0] - second[0]

                summ += (first[1] * second[1]) / (dif ** 2)



    # number of intersections

    k = len(set(keywords.keys()).intersection(set(words))) + 1

    return (1 / (k * (k + 1.0)) * summ)





def score(sentences, titleWords, keywords):

    """score sentences based on different features"""



    senSize = len(sentences)

    ranks = Counter()

    for i, s in enumerate(sentences):

        sentence = split_words(s)

        titleFeature = title_score(titleWords, sentence)

        sentenceLength = length_score(sentence)

        sentencePosition = sentence_position(i + 1, senSize)

        sbsFeature = sbs(sentence, keywords)

        dbsFeature = dbs(sentence, keywords)

        frequency = (sbsFeature + dbsFeature) / 2.0 * 10.0

        findingsScore = findings_score(sentence)



        scores = [

            titleFeature,

            frequency,

            sentenceLength,

            sentencePosition,

            findingsScore,

        ]

        weights = [

            1.5,

            2.0,

            1.0,

            1.0,

            3.0,

        ]

        totalScore = np.average(scores, weights=weights)

        ranks[s] = totalScore

    return ranks





def summarize(title, sentences, n_sentences=5):

    """Returns the top N sentences summarizing the whole content"""

    summaries = []

    # sentences = split_sentences(text)

    text = ". ".join(sentences)

    keys = keywords(text)

    titleWords = split_words(title)



    if len(sentences) <= n_sentences:

        return sentences



    # score setences, and use the top n sentences

    ranks = score(sentences, titleWords, keys).most_common(n_sentences)

    for rank in ranks:

        summaries.append(rank[0])

    return summaries



def get_paper_summary(paper_row):

    summary = summarize(paper_row["title"], paper_row["sentences_cleaned"], n_sentences=3)

    return [

        sentence for (sentence, sentence_cleaned) in zip(paper_row["sentences"], paper_row["sentences_cleaned"])

        if sentence_cleaned in summary

    ]
df["summary"] = df.apply(

    get_paper_summary,

    axis=1,

)

df[["abstract_cleaned", "summary"]].head()
# Extract word vectors



print("Loading word embeddings")

word_embeddings = {}

f = open(GLOVE_FILE, encoding="utf-8")

for line in f:

    values = line.split()

    word = values[0]

    coefs = np.asarray(values[1:], dtype="float32")

    word_embeddings[word] = coefs

f.close()



def embed_text(text):

    return np.nanmean([word_embeddings.get(w, np.nan * np.zeros((GLOVE_NDIMS,))) for w in text.split()], axis=0)



print("Computing embeddings")

embeddings = df["abstract_cleaned"].apply(embed_text)

embeddings = pd.DataFrame(dict(zip(embeddings.index, embeddings.values))).T

gc.collect()

embeddings.head()



print("Computing SVD")

t_svd = TruncatedSVD(n_components=2, random_state=RANDOM_STATE)

t_svd.fit(embeddings.values)



pc = t_svd.components_

gc.collect()



# Remove PCs

embeddings = embeddings - embeddings.values.dot(pc.transpose()).dot(pc)



# Normalize

embeddings /= np.linalg.norm(embeddings, axis=1).reshape(-1, 1)

embeddings.head()
SUMMARY_SENTENCE_TPL = "    <p>{sentence}</p>"



HTML_TPL = """

<h3><a href="{url}" target="_blank">{title:<50s}</a> ({journal:<50s})</h3>

{topics_table}

<br>

<div style="display: inline">

{summary}

</div>

<br>

<hr>

<h4>Related papers:</h4>

{related_papers}

<hr>

"""



RELATED_TPL = """

<h5><a href="{url}" target="_blank">{title:<50s}</a> ({journal:<50s})</h5>

<div style="display: inline">

{summary}

</div>

<br>

"""



def embed_question(question):

    embedded_question = embed_text(question)

    embedded_question -= embedded_question.dot(pc.transpose()).dot(pc)

    embedded_question /= np.linalg.norm(embedded_question) or 1

    embedded_question = np.expand_dims(embedded_question, axis=0)

    embedded_question[np.isnan(embedded_question)] = 0

    if embedded_question.max() == 0:

        print("No embeddings available for [%s]" % question)

    return embedded_question



def get_similarity(embedded_question):

    sim = cosine_similarity(embedded_question, embeddings)

    return sim[0]



def html_view(row, related_papers):

    top_topics = row[topic_columns].sort_values(ascending=False).head(3).to_frame("Distribution")

    top_topics["Topic"] = [df_descriptions.loc[int(c.split("_")[1]), "Topic description"] for c in top_topics.index]

    top_topics = top_topics[["Topic", "Distribution"]]

        

    html_summary = "\n".join([SUMMARY_SENTENCE_TPL.format(sentence=s) for s in row["summary"]])

    html_related_papers = "\n".join([

        RELATED_TPL.format(

            title=related_papers.loc[idx, "title"],

            journal=related_papers.loc[idx, "journal"],

            url=related_papers.loc[idx, "url"],

            summary="\n".join([SUMMARY_SENTENCE_TPL.format(sentence=s) for s in related_papers.loc[idx, "summary"]]),

        )

        for idx in related_papers.index

    ])

        

    return HTML_TPL.format(

        title=row["title"],

        journal=row["journal"],

        url=row["url"],

        topics_table=top_topics.to_html(index=False),

        summary=html_summary,

        related_papers=html_related_papers,

    )



def view(question="", num_documents=5, num_similar_docs=2):

    if question:

        print(question)

        embedded_question = embed_question(question)

        df["similarity_to_question"] = get_similarity(embedded_question)

        head = df.sort_values(by="similarity_to_question", ascending=False).head(num_documents)

        

        papers = []

        for row in head.index:

            related_papers = get_closest_papers(head.loc[row, "count"], num=num_similar_docs)

            related_papers = df.loc[df["count"].isin(related_papers)]            

            papers.append(html_view(head.loc[row], related_papers))

            

        # TODO add

        # * paper metadata (authors, publish time, link)

        # * major findings (need to refine by topic)

        return display(HTML("\n".join(papers)))



question_input = widgets.Text()

num_documents = widgets.IntSlider(min=1, max=20, step=1, value=5)

num_similar_docs = widgets.IntSlider(min=1, max=5, step=1, value=2)

widgets.interactive(view, question=question_input, num_documents=num_documents, num_similar_docs=num_similar_docs)
view(question="coronavirus transmission incubation", num_documents=5, num_similar_docs=2)
view(question="coronavirus asymptomatic shedding transmission", num_documents=3, num_similar_docs=2)
view(question="coronavirus environmental stability", num_documents=3, num_similar_docs=2)
view(question="coronavirus transmission seasonality", num_documents=3, num_similar_docs=2)
view(question="coronavirus propagation model", num_documents=3, num_similar_docs=2)
view(question="coronavirus risk factors", num_documents=5, num_similar_docs=2)
view(question="coronavirus smoking pulmonary disease", num_documents=3, num_similar_docs=2)
view(question="coronavirus neonates pregnant", num_documents=3, num_similar_docs=2)
view(question="coronavirus social behaviour risk", num_documents=3, num_similar_docs=2)
view(question="coronavirus genetics origin evolution", num_documents=5, num_similar_docs=2)
view(question="coronavirus human animal", num_documents=3, num_similar_docs=2)
view(question="coronavirus genome differences", num_documents=3, num_similar_docs=2)
view(question="coronavirus geographic distribution", num_documents=3, num_similar_docs=2)
view(question="coronavirus vaccine", num_documents=5, num_similar_docs=2)
view(question="coronavirus therapeutic", num_documents=5, num_similar_docs=2)
view(question="coronavirus drug effectiveness", num_documents=3, num_similar_docs=2)
view(question="coronavirus vaccine development", num_documents=3, num_similar_docs=2)
view(question="coronavirus antibody dependent enhancement", num_documents=3, num_similar_docs=2)
view(question="coronavirus medical care", num_documents=5, num_similar_docs=2)
view(question="coronavirus patient acute respiratory distress syndrome", num_documents=3, num_similar_docs=2)
view(question="nursing facility", num_documents=3, num_similar_docs=2)
view(question="processes care", num_documents=3, num_similar_docs=2)
view(question="personal protective equipment", num_documents=3, num_similar_docs=2)
view(question="diagnostics surveillance", num_documents=5, num_similar_docs=2)
view(question="coronavirus policy", num_documents=5, num_similar_docs=2)
view(question="coronavirus diagnostics surveillance platform", num_documents=3, num_similar_docs=2)
view(question="school closure", num_documents=3, num_similar_docs=2)
view(question="travel ban", num_documents=3, num_similar_docs=2)
view(question="social distancing", num_documents=3, num_similar_docs=2)
view(question="coronavirus advice compliance", num_documents=3, num_similar_docs=2)
view(question="information sharing collaboration", num_documents=5, num_similar_docs=2)
view(question="government response communication", num_documents=3, num_similar_docs=2)
view(question="coronavirus healthcare worker", num_documents=3, num_similar_docs=2)
view(question="research ethical principles", num_documents=5, num_similar_docs=2)
view(question="coronavirus social science", num_documents=3, num_similar_docs=2)