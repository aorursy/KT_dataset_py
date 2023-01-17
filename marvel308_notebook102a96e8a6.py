# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

from bs4 import BeautifulSoup

from nltk.corpus import stopwords

import re

import string

import math

from collections import defaultdict, OrderedDict

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
dataframes = {

    "cooking": pd.read_csv("../input/cooking.csv"),

    "crypto": pd.read_csv("../input/crypto.csv"),

    "robotics": pd.read_csv("../input/robotics.csv"),

    "biology": pd.read_csv("../input/biology.csv"),

    "travel": pd.read_csv("../input/travel.csv"),

    "diy": pd.read_csv("../input/diy.csv"),

}
print(dataframes["robotics"].iloc[1])
uri_re = r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))'



def stripTagsAndUris(x):

    if x:

        # BeautifulSoup on content

        soup = BeautifulSoup(x, "html.parser")

        # Stripping all <code> tags with their content if any

        if soup.code:

            soup.code.decompose()

        # Get all the text out of the html

        text =  soup.get_text()

        # Returning text stripping out all uris

        return re.sub(uri_re, "", text)

    else:

        return ""
for df in dataframes.values():

    df["content"] = df["content"].map(stripTagsAndUris)
def removePunctuation(x):

    # Lowercasing all words

    x = x.lower()

    # Removing non ASCII chars

    x = re.sub(r'[^\x00-\x7f]',r' ',x)

    # Removing (replacing with empty spaces actually) all the punctuations

    return re.sub("["+string.punctuation+"]", " ", x)
for df in dataframes.values():

    df["title"] = df["title"].map(removePunctuation)

    df["content"] = df["content"].map(removePunctuation)
print(dataframes["robotics"].iloc[1])
stops = set(stopwords.words("english"))

def removeStopwords(x):

    # Removing all the stopwords

    filtered_words = [word for word in x.split() if word not in stops]

    return " ".join(filtered_words)
for df in dataframes.values():

    df["title"] = df["title"].map(removeStopwords)

    df["content"] = df["content"].map(removeStopwords)
print(dataframes["robotics"].iloc[1])
for df in dataframes.values():

    # From a string sequence of tags to a list of tags

    df["tags"] = df["tags"].map(lambda x: x.split())
def get_words(text):

    word_split = re.compile('[^a-zA-Z0-9_\\+\\-/]')

    return [word.strip().lower() for word in word_split.split(text)]

print(type(get_words("Hey How are you")));
def process_text(doc, idf, text):

    tf = OrderedDict();

    word_count = 0.

    #print(get_words(text));

    for word in get_words(text):

        #print(word);

        #print(type(word));

        if word not in tf:

            tf[word] = 0

        tf[word] += 1

        idf[word].add(doc)

        word_count += 1.



    for word in tf:

        tf[word] = tf[word] / word_count



    return tf, word_count
def getString(df):

    return "".join(df.astype('str').tail(1).tolist());
docs = [];

idf = defaultdict(set);

tf = {};

word_counts = defaultdict(float);

count = 0;

for df in dataframes.values():

    count+=1;

    doc = int(getString(df["id"]));

    text = getString(df["title"])+" "+getString(df["content"]);

    #print(text);

    #print(type(text));

    #docs.append(doc);

    #tf[doc], word_counts[doc] = process_text(doc, idf, text)

    myset = set(get_words(text));

    mynewlist = list(myset);

    #print(mynewlist);

    for word in mynewlist:

        if word not in tf:

            tf[word] = 0

        tf[word] += 1

    #break;

print(count);

print(tf);
nr_docs = len(docs)

for doc in docs:

    for word in tf[doc]:

        tf[doc][word] *= math.log(nr_docs / len(idf[word]))