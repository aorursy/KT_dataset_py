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



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import nltk

import markovify

import sys

from collections import Counter

import csv

import itertools

import re

import urllib.request

from nltk.corpus import wordnet

from scipy.spatial.distance import cosine
grammar = nltk.CFG.fromstring("""

    S -> NP VP



    AP -> A | A AP

    NP -> N | D NP | AP NP | N PP

    PP -> P NP

    VP -> V | V NP | V NP PP



    A -> "big" | "blue" | "small" | "dry" | "wide"

    D -> "the" | "a" | "an"

    N -> "she" | "city" | "car" | "street" | "dog" | "binoculars"

    P -> "on" | "over" | "before" | "below" | "with" | "in"

    V -> "saw" | "walked"

""")



parser = nltk.ChartParser(grammar)



sentence = "she saw a dog in the car".split()

try:

    for tree in parser.parse(sentence):

        tree.pretty_print()

        break

except ValueError:

    print("No parse tree possible.")
# This sentence makes no sense but can be parsed.

sentence = "she walked the car".split()

try:

    for tree in parser.parse(sentence):

        tree.pretty_print()

        break

except ValueError:

    print("No parse tree possible.")
contents=[]

with open("/kaggle/input/nlphp/hp.txt") as f:

    contents.extend([

        word.lower() for word in

        nltk.word_tokenize(f.read())

        if any(c.isalpha() for c in word)

    ])



# Compute n-grams

ngrams = Counter(nltk.ngrams(contents, 4))



# Print most common n-grams

for ngram, freq in ngrams.most_common(10):

    print(f"{freq}: {ngram}")


# Compute n-grams

ngrams = Counter(nltk.ngrams(contents, 2))



# Print most common n-grams

for ngram, freq in ngrams.most_common(10):

    print(f"{freq}: {ngram}")
# Compute n-grams

ngrams = Counter(nltk.ngrams(contents, 1))



# Print most common n-grams

for ngram, freq in ngrams.most_common(10):

    print(f"{freq}: {ngram}")
# Read text from file

with open("/kaggle/input/nlphp/hp.txt") as f:

    text = f.read()



# Train model

text_model = markovify.Text(text)



# Generate sentences

print()

for i in range(5):

    print(text_model.make_sentence())

    print()
#sentiment analysis

def extract_words(document):

    return set(

        word.lower() for word in nltk.word_tokenize(document)

        if any(c.isalpha() for c in word)

    )





def load_data():

    result = []

    for filename in ["positives.txt", "negatives.txt"]:

        file="/kaggle/input/nlphp/"+filename

        with open(file) as f:

            result.append([

                extract_words(line)

                for line in f.read().splitlines()

            ])

    return result





def generate_features(documents, words, label):

    features = []

    for document in documents:

        features.append(({

            word: (word in document)

            for word in words

        }, label))

    return features





def classify(classifier, document, words):

    document_words = extract_words(document)

    features = {

        word: (word in document_words)

        for word in words

    }

    return classifier.prob_classify(features)

# Read data from files

positives, negatives = load_data()



# Create a set of all words

words = set()

for document in positives:

    words.update(document)

for document in negatives:

    words.update(document)



# Extract features from text

training = []

training.extend(generate_features(positives, words, "Positive"))

training.extend(generate_features(negatives, words, "Negative"))



# Classify a new sample

classifier = nltk.NaiveBayesClassifier.train(training)

s = "not good"

result = (classify(classifier, s, words))

for key in result.samples():

    print(f"{key}: {result.prob(key):.4f}")
s = "good but not great"

result = (classify(classifier, s, words))

for key in result.samples():

    print(f"{key}: {result.prob(key):.4f}")
s = "loved it"

result = (classify(classifier, s, words))

for key in result.samples():

    print(f"{key}: {result.prob(key):.4f}")
#Automated Template Generation

with open("/kaggle/input/nlphp/books.csv") as f:

    examples = list(csv.reader(f))

corpus = ""

url=urllib.request.urlopen("https://www.penguinrandomhouse.com/the-read-down/21-books-youve-been-meaning-to-read")

corpus+=str(url.read()).replace("\n", " ")



def find_templates(examples, corpus):

    templates = []

    for a, b in examples:

        templates.extend(match_query(a, b, True, corpus))

        templates.extend(match_query(b, a, False, corpus))



    # Find common middles

    middles = dict()

    for template in templates:

        middle = template["middle"]

        order = template["order"]

        if (middle, order) in middles:

            middles[middle, order].append(template)

        else:

            middles[middle, order] = [template]



    # Filter middles to only those used multiple times

    middles = {

        middle: middles[middle]

        for middle in middles

        if len(middles[middle]) > 1

    }



    # Look for common prefixes and suffixes

    results = []

    for middle in middles:

        found = set()

        for t1, t2 in itertools.combinations(middles[middle], 2):

            prefix = common_suffix(t1["prefix"], t2["prefix"])

            suffix = common_prefix(t1["suffix"], t2["suffix"])

            if (prefix, suffix) not in found:

                if (not len(prefix) or not len(suffix)

                   or not prefix.strip() or not suffix.strip()):

                        continue

                found.add((prefix, suffix))

                results.append({

                    "order": middle[1],

                    "prefix": prefix,

                    "middle": middle[0],

                    "suffix": suffix

                })

    return results





def filter_templates(templates, n):

    return sorted(

        templates,

        key=lambda t: len(t["prefix"]) + len(t["suffix"]),

        reverse=True

    )[:n]





def extract_from_templates(templates, corpus):

    results = set()

    for template in templates:

        results.update(match_template(template, corpus))

    return results





def match_query(q1, q2, order, corpus):

    q1 = re.escape(q1)

    q2 = re.escape(q2)

    regex = f"(.{{0,10}}){q1}((?:(?!{q1}).)*?){q2}(.{{0,10}})"

    results = re.findall(regex, corpus)

    return [

        {

            "order": order,

            "prefix": result[0],

            "middle": result[1],

            "suffix": result[2]

        }

        for result in results

    ]





def match_template(template, corpus):

    prefix = re.escape(template["prefix"])

    middle = re.escape(template["middle"])

    suffix = re.escape(template["suffix"])

    regex = f"{prefix}((?:(?!{prefix}).){{0,40}}?){middle}(.{{0,40}}?){suffix}"

    print(prefix)

    print(middle)

    print(suffix)

    results = re.findall(regex, corpus)

    if template["order"]:

        return results

    else:

        return [(b, a) for (a, b) in results]





def common_prefix(*s):

    # https://rosettacode.org/wiki/Longest_common_prefix#Python

    return "".join(

        ch[0] for ch in itertools.takewhile(

            lambda x: min(x) == max(x), zip(*s)

        )

    )





def common_suffix(*s):

    s = [x[::-1] for x in list(s)]

    return common_prefix(*s)[::-1]



templates = find_templates(examples, corpus)

templates = filter_templates(templates, 2)

results = extract_from_templates(templates, corpus)

for result in results:

    print(result)

i=corpus.find('<h2><a href="/books/88933/ulysses-by-james-joyce/9780679722762/">Ulysses</a></h2>\\n\\t\\t<h2 class="author">by James Joyce</h2>')

i
corpus[i:i+200]
with open("/kaggle/input/nlphp/words.txt") as f:

    words = dict()

    for i in range(50000):

        row = next(f).split()

        word = row[0]

        vector = np.array([float(x) for x in row[1:]])

        words[word] = vector





def distance(w1, w2):

    return cosine(w1, w2)





def closest_words(embedding):

    distances = {

        w: distance(embedding, words[w])

        for w in words

    }

    return sorted(distances, key=lambda w: distances[w])[:10]
words['city']
distance(words['book'],words['book'])
closest_words(words["book"])[:10]
distance(words['book'],words['novella'])
distance(words['book'],words['city'])
closest_words(words['king']- words['man']+words['woman'])[:1]