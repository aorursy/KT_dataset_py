import pandas as pd

import sklearn as sk

import requests

import bs4



import matplotlib.pyplot as plt



%matplotlib inline
# http request

r1 = requests.get('https://www.washingtonpost.com/politics/2019/07/31/transcript-first-night-second-democratic-debate')

r2 = requests.get('https://www.washingtonpost.com/politics/2019/08/01/transcript-night-second-democratic-debate/')
def parse_requests(r, night=None):

    # Parse html

    soup = bs4.BeautifulSoup(r.content)

    graphs = soup.find_all('p')

    utterances = [x.get_text() for x in graphs if 'data-elm-loc' in x.attrs.keys()]



    # Parse utterances

    utterances = utterances [2:]

    seq = 0

    data = []

    for i in utterances:

        i = i.replace('DE BLASIO:', 'DEBLASIO:')

        graph = i.split()

        if graph[0][-1] == ':':

            text = ' '.join(graph[1:])

            num_words = len(graph) - 1

            name = graph[0][:-1]

            seq += 1

        elif len(graph) > 1 and graph[1] == '(?):':

            # Cases like 'WARREN (?):'

            text = ' '.join(graph[2:])

            num_words = len(graph) - 2

            name = graph[0]

            seq += 1

        else:

            text = ' '.join(graph)

        if name == '[Transcript':

            pass

        else:

            data.append({"name": name,

              "graph": text,

              "seq": seq,

              "num_words": num_words,

              "night": night

            })

    return data



data = parse_requests(r1, night=0) + parse_requests(r2, night=1)
df = pd.DataFrame(data)

df.name.unique()
# "Unknown", O'Rourke parsing errors

df = df[df.name != "(UNKNOWN)"]

df['name'] = df['name'].apply(lambda x: ''.join([char for char in x if char.isalpha()]))



# There was also a protestor on night 2

df = df[df.name != "PROTESTOR"]



df.name.unique()
data.head()
# Example quick plotting

plt.style.use('fivethirtyeight')

words_freq_plot = df.groupby('name').sum()['num_words'].plot(

    kind='bar', figsize=(8, 4)

);

words_freq_plot.set_ylabel('Words Spoken')

words_freq_plot.set_title("Candidate Approx Word Totals");
# Squash multiple lines with same name

df['graph'] = df.groupby('seq')['graph'].transform(' '.join)

df = df[['graph', 'seq', 'name', 'night']].drop_duplicates()

df.head()
df.name.unique()
import numpy as np

import sklearn.feature_extraction.text as skt

from wordcloud import WordCloud



import nltk

from nltk import word_tokenize 

from nltk.stem.porter import PorterStemmer



#######

# based on http://www.cs.duke.edu/courses/spring14/compsci290/assignments/lab02.html

stemmer = PorterStemmer()

def stem_tokens(tokens, stemmer):

    stemmed = []

    for item in tokens:

        stemmed.append(stemmer.stem(item))

    return stemmed



def tokenize(text):

    tokens = nltk.word_tokenize(text)

    return tokens

######## 



def topwords_candidate(candidate_name, n):

    vectorizer = skt.TfidfVectorizer(stop_words='english', tokenizer=tokenize)

    X = vectorizer.fit_transform(df[df['name']==candidate_name]['graph'])

    feature_names = vectorizer.get_feature_names()

    doc = 0 # Opening statement

    feature_index = X[doc,:].nonzero()[1]

    

    tfidf_scores = zip(feature_index, [X[doc, x] for x in feature_index])

    scored_features = sorted([(feature_names[i], s) for (i, s) in tfidf_scores], key=lambda x: x[1])

    

    data = scored_features[-n:]

    

    

    # Generate a word cloud image

    wordcloud = WordCloud().generate(' '.join([x[0] for x in data][::-1]))

    plt.imshow(wordcloud, interpolation='bilinear')

    plt.axis("off")

    return data, wordcloud
figs, axs = plt.subplots(4, 5, figsize=(24, 8))

figs.suptitle("""Top TF-IDF Weighted Words in Opening Speeches, by Candidate""",

             fontsize=24)

candidates = list(filter(lambda x: x not in ['BASH', 'TAPPER', 'LEMON'], df.name.unique()))

for k in range(4):

    for i in range(5):

    

        mod = k*5

        axs[k][i].imshow(topwords_candidate(candidates[i+mod], 10)[1])

        axs[k][i].axis('off')

        axs[k][i].set_title(candidates[i+mod], fontsize=16)
def get_all_text(candidate):

    all_docs = df[df.name == candidate]['graph']

    all_docs = ' '.join(all_docs.values)

    return all_docs



corpus = [get_all_text(cand) for cand in df.name.unique()]
vect = skt.TfidfVectorizer(min_df=1)

tfidf = vect.fit_transform(corpus)



distance_matrix = (tfidf * tfidf.T).A



def colorval_name(k):

    if k in ['BASH', 'TAPPER', 'LEMON']:

        return 'r', 'CNN'

    elif k in ['SANDERS', 'WARREN', 'HARRIS', 'BIDEN']:

        return 'b', 'Tier 1'

    else:

        return 'g', 'Tier 2'



drawn_labels = []

for i, name in enumerate(df.name.unique()):

    c, label = colorval_name(name)

    plt.scatter(distance_matrix[i, 0], distance_matrix[i, 1], c=c, label=label if label not in drawn_labels else '')

    drawn_labels.append(label)

plt.title("Candidate Similarity", fontsize=20)



legend = plt.legend(loc='lower right')