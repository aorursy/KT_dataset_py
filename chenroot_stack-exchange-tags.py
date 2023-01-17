# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

from bs4 import BeautifulSoup

from imblearn.over_sampling import RandomOverSampler

from stop_words import get_stop_words

from collections import OrderedDict

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



# machine learning

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.multiclass import OneVsRestClassifier

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import SGDClassifier

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2
biology_df = pd.read_csv("../input/biology.csv")

biology_df.info()

cooking_df = pd.read_csv("../input/cooking.csv")

cooking_df.info()

crypto_df = pd.read_csv("../input/crypto.csv")

crypto_df.info()

diy_df = pd.read_csv("../input/diy.csv")

diy_df.info()

robotics_df = pd.read_csv("../input/robotics.csv")

robotics_df.info()

travel_df = pd.read_csv("../input/travel.csv")

travel_df.info()

train_df = pd.concat([biology_df, cooking_df, crypto_df, diy_df, robotics_df, travel_df], ignore_index=True)

train_df.info()

test_df = pd.read_csv("../input/test.csv")

test_df.info()
stop_words = ['a', "a's", 'able', 'about', 'above', 'according', 'accordingly', 'across', 'actually',

              'after', 'afterwards', 'again', 'against', "ain't", 'all', 'allow', 'allows', 'almost',

              'alone', 'along', 'already', 'also', 'although', 'always', 'am', 'among', 'amongst', 'an',

              'and', 'another', 'any', 'anybody', 'anyhow', 'anyone', 'anything', 'anyway', 'anyways',

              'anywhere', 'apart', 'appear', 'appreciate', 'appropriate', 'are', "aren't", 'around',

              'as', 'aside', 'ask', 'asking', 'associated', 'at', 'available', 'away', 'awfully', 'b',

              'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before', 'beforehand',

              'behind', 'being', 'believe', 'below', 'beside', 'besides', 'best', 'better', 'between',

              'beyond', 'both', 'brief', 'but', 'by', 'c', "c'mon", "c's", 'came', 'can', "can't", 'cannot',

              'cant', 'cause', 'causes', 'certain', 'certainly', 'changes', 'clearly', 'co', 'com', 'come',

              'comes', 'concerning', 'consequently', 'consider', 'considering', 'contain', 'containing',

              'contains', 'corresponding', 'could', "couldn't", 'course', 'currently', 'd', 'definitely',

              'described', 'despite', 'did', "didn't", 'different', 'do', 'does', "doesn't", 'doing', "don't",

              'done', 'down', 'downwards', 'during', 'e', 'each', 'edu', 'eg', 'eight', 'either', 'else',

              'elsewhere', 'enough', 'entirely', 'especially', 'et', 'etc', 'even', 'ever', 'every', 'everybody',

              'everyone', 'everything', 'everywhere', 'ex', 'exactly', 'example', 'except', 'f', 'far', 'few',

              'fifth', 'first', 'five', 'followed', 'following', 'follows', 'for', 'former', 'formerly', 'forth',

              'four', 'from', 'further', 'furthermore', 'g', 'get', 'gets', 'getting', 'given', 'gives', 'go',

              'goes', 'going', 'gone', 'got', 'gotten', 'greetings', 'h', 'had', "hadn't", 'happens', 'hardly',

              'has', "hasn't", 'have', "haven't", 'having', 'he', "he's", 'hello', 'help', 'hence', 'her', 'here',

              "here's", 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'hi', 'him', 'himself',

              'his', 'hither', 'hopefully', 'how', 'howbeit', 'however', 'i', "i'd", "i'll", "i'm", "i've", 'ie',

              'if', 'ignored', 'immediate', 'in', 'inasmuch', 'inc', 'indeed', 'indicate', 'indicated', 'indicates',

              'inner', 'insofar', 'instead', 'into', 'inward', 'is', "isn't", 'it', "it'd", "it'll", "it's", 'its',

              'itself', 'j', 'just', 'k', 'keep', 'keeps', 'kept', 'know', 'knows', 'known', 'l', 'last', 'lately',

              'later', 'latter', 'latterly', 'least', 'less', 'lest', 'let', "let's", 'like', 'liked', 'likely',

              'little', 'look', 'looking', 'looks', 'ltd', 'm', 'mainly', 'many', 'may', 'maybe', 'me', 'mean',

              'meanwhile', 'merely', 'might', 'more', 'moreover', 'most', 'mostly', 'much', 'must', 'my',

              'myself', 'n', 'name', 'namely', 'nd', 'near', 'nearly', 'necessary', 'need', 'needs',

              'neither', 'never', 'nevertheless', 'new', 'next', 'nine', 'no', 'nobody', 'non', 'none',

              'noone', 'nor', 'normally', 'not', 'nothing', 'novel', 'now', 'nowhere', 'o', 'obviously',

              'of', 'off', 'often', 'oh', 'ok', 'okay', 'old', 'on', 'once', 'one', 'ones', 'only', 'onto',

              'or', 'other', 'others', 'otherwise', 'ought', 'our', 'ours', 'ourselves', 'out', 'outside',

              'over', 'overall', 'own', 'p', 'particular', 'particularly', 'per', 'perhaps', 'placed',

              'please', 'plus', 'possible', 'presumably', 'probably', 'provides', 'q', 'que', 'quite',

              'qv', 'r', 'rather', 'rd', 're', 'really', 'reasonably', 'regarding', 'regardless',

              'regards', 'relatively', 'respectively', 'right', 's', 'said', 'same', 'saw', 'say',

              'saying', 'says', 'second', 'secondly', 'see', 'seeing', 'seem', 'seemed', 'seeming',

              'seems', 'seen', 'self', 'selves', 'sensible', 'sent', 'serious', 'seriously', 'seven',

              'several', 'shall', 'she', 'should', "shouldn't", 'since', 'six', 'so', 'some', 'somebody',

              'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhat', 'somewhere', 'soon',

              'sorry', 'specified', 'specify', 'specifying', 'still', 'sub', 'such', 'sup', 'sure', 't',

              "t's", 'take', 'taken', 'tell', 'tends', 'th', 'than', 'thank', 'thanks', 'thanx', 'that',

              "that's", 'thats', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'thence', 'there',

              "there's", 'thereafter', 'thereby', 'therefore', 'therein', 'theres', 'thereupon', 'these',

              'they', "they'd", "they'll", "they're", "they've", 'think', 'third', 'this', 'thorough',

              'thoroughly', 'those', 'though', 'three', 'through', 'throughout', 'thru', 'thus', 'to',

              'together', 'too', 'took', 'toward', 'towards', 'tried', 'tries', 'truly', 'try', 'trying',

              'twice', 'two', 'u', 'un', 'under', 'unfortunately', 'unless', 'unlikely', 'until', 'unto',

              'up', 'upon', 'us', 'use', 'used', 'useful', 'uses', 'using', 'usually', 'uucp', 'v', 'value',

              'various', 'very', 'via', 'viz', 'vs', 'w', 'want', 'wants', 'was', "wasn't", 'way', 'we',

              "we'd", "we'll", "we're", "we've", 'welcome', 'well', 'went', 'were', "weren't", 'what',

              "what's", 'whatever', 'when', 'whence', 'whenever', 'where', "where's", 'whereafter',

              'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while',

              'whither', 'who', "who's", 'whoever', 'whole', 'whom', 'whose', 'why', 'will', 'willing',

              'wish', 'with', 'within', 'without', "won't", 'wonder', 'would', 'would', "wouldn't",

              'x', 'y', 'yes', 'yet', 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours',

              'yourself', 'yourselves', 'z', 'zero', '']
def clean_content(row) :

    string = row["content"]

    string = "".join(string.split("\n"))

    soup = BeautifulSoup(string, "html.parser")

    return " ".join(re.findall("[a-zA-Z]+", soup.get_text().lower()))

def clean_title(row) :

    string = row["title"]

    return " ".join(re.findall("[a-zA-Z]+", string.lower()))

def word_in_title(row) :

    tags = set(row["tags"].split())

    words = set(row["title"].split())

    return len(tags & words)

def word_in_content(row) :

    tags = set(row["tags"].split())

    words = set(row["content"].split())

    return len(tags & words)

def tags_count(row) :

    return len(set(row["tags"].split()))

biology_df["title"] = biology_df.apply(clean_title, axis = 1)

biology_df["content"] = biology_df.apply(clean_content, axis = 1)

biology_df["title_in_tags"] = biology_df.apply(word_in_title, axis = 1)

biology_df["content_in_tags"] = biology_df.apply(word_in_content, axis = 1)

biology_df["tags_count"] = biology_df.apply(tags_count, axis = 1)

print(np.sum(biology_df["title_in_tags"]))

print(np.sum(biology_df["content_in_tags"]))

print(np.sum(biology_df["tags_count"]))