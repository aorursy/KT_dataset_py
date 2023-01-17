import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import nltk

from nltk.sentiment import vader

from nltk.sentiment.vader import SentimentIntensityAnalyzer

vader_model = SentimentIntensityAnalyzer()

import spacy

from sklearn import metrics

nlp = spacy.load('en')



import re

from collections import Counter
df = pd.read_csv("../input/mbti_1.csv")

posts = df['posts'].values.tolist()

types = df['type'].values.tolist()

print(len(posts),len(types))
# Preprocess data:



## Replace post separator ||| with semi-colon ;

df['posts'] = df['posts'].replace(to_replace = r'\|\|\|', value = r';',regex=True)



## Replace all http links with 'url'

pattern_url = r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+'

subs_url = r'url'

df['posts'] = df['posts'].replace(to_replace = pattern_url, value = subs_url, regex = True)



## Remove Personality Type Words - to ensure the validity of the estimation for unseen instances

pers_types = ['INFP' ,'INFJ', 'INTP', 'INTJ', 'ENTP', 'ENFP', 'ISTP' ,'ISFP' ,'ENTJ', 'ISTJ','ENFJ', 'ISFJ' ,'ESTP', 'ESFP' ,'ESFJ' ,'ESTJ']

p = re.compile("(" + "|".join(pers_types) + ")")

df['posts'] = df['posts'].replace(to_replace = p, value = r'PTypeToken', regex = True)
print(df.head())

print(df.tail())
total_dict = {}

for i in range(len(posts)):

    total_dict[i] = [types[i],posts[i]]
print('Personality types and their frequencies:')

types = df.groupby('type').count()

types.sort_values("posts", ascending=False, inplace=True)

print(types)
types['posts'].plot(kind="bar", title="Number of Users per Personality type")
#Function that counts the number of parts of speech

def bag_of_words(group, type_label):

    posts = [t for t in group.get_group(type_label)['posts']]

    nlp = spacy.load('en_core_web_sm')

    count_tags = Counter()

    for posts_per_user in posts:

        doc = nlp(str(posts_per_user))

        count_tags.update(Counter([token.pos_ for token in doc]))

    return count_tags
# Function that plots parts of speech

def tags_pie_plot(count_tags):

    bag_of_tags = list(count_tags.keys())

    bag_of_tags_values = [count_tags.get(l) for l in bag_of_tags[:5]]

    

#     fig = figure()

    fig = plt.pie(bag_of_tags_values, labels = bag_of_tags[:5], autopct = '%1.1f%%', startangle = 140)

    

    return fig
types_grouped = df.groupby('type')

for t in pers_types:

    count_tags = bag_of_words(types_grouped, t)

    tags_pie_plot(count_tags)

    print(t)

    plt.show()
def replace_nth_string(string, old, new, n):

    num=0

    start=-1

    while num<n:

        start=string.find(old, start+1)

        if start == -1:return -1

        num+=1

    positioned_string = start

    if n == -1:

        return string

    return string[:positioned_string]+new+string[positioned_string+len(old):]
#Remove 'like' that is not a verb

def pos_sentence(message):

    sen_array = nltk.word_tokenize(message)

    tagged = nltk.pos_tag(sen_array)

    i = 0

    indx = []

    for x,y in tagged:

        if x == 'like':

            i += 1

            if y != 'VB':

                indx.append(i)

    new_msg = message

    for i in indx:

        new_msg = replace_nth_string(message, 'like', '', i)

    return new_msg
from IPython.display import clear_output
pos_feature = []

entity_feature =[]

len_posts = []

i = 0

for posts_per_user in df['posts']:

    clear_output(wait=True)

    doc = nlp(posts_per_user)

    count_tags = Counter([token.pos_ for token in doc])

    count_labels = Counter([token.label_ for token in doc.ents])

    pos_feature.append(count_tags)

    entity_feature.append(count_labels)

    len_posts.append({'length': len(posts_per_user)})

    print("Current progress: ", i)

    i+=1
def run_vader(textual_unit, 

              lemmatize=False, 

              parts_of_speech_to_consider=set(),

              verbose=0):

    """

    Run VADER on a sentence from spacy

    

    :param str textual unit: a textual unit, e.g., sentence, sentences (one string)

    (by looping over doc.sents)

    :param bool lemmatize: If True, provide lemmas to VADER instead of words

    :param set parts_of_speech_to_consider:

    -empty set -> all parts of speech are provided

    -non-empty set: only these parts of speech are considered

    :param int verbose: if set to 1, information is printed

    about input and output

    

    :rtype: dict

    :return: vader output dict

    """

    doc = nlp(textual_unit)

        

    input_to_vader = []



    for sent in doc.sents:

        for token in sent:



            to_add = token.text



            if lemmatize:

                to_add = token.lemma_



                if to_add == '-PRON-': 

                    to_add = token.text



            if parts_of_speech_to_consider:

                if token.pos_ in parts_of_speech_to_consider:

                    input_to_vader.append(to_add) 

            else:

                input_to_vader.append(to_add)



    scores = vader_model.polarity_scores(' '.join(input_to_vader))

    

    if verbose >= 1:

        print()

        print('INPUT SENTENCE', sent)

        print('INPUT TO VADER', input_to_vader)

        print('VADER OUTPUT', scores)



    return scores
def vader_output_to_label(vader_output):

    """

    map vader output e.g.,

    {'neg': 0.0, 'neu': 0.0, 'pos': 1.0, 'compound': 0.4215}

    to one of the following values:

    a) positive float -> 'positive'

    b) 0.0 -> 'neutral'

    c) negative float -> 'negative'

    

    :param dict vader_output: output dict from vader

    

    :rtype: str

    :return: 'negative' | 'neutral' | 'positive'

    """

    compound = vader_output['compound']

    negative = vader_output['neg']

    positive = vader_output['pos']

    neutral = vader_output['neu']

    

    minimum = 2

    to_return = "positive"

    if abs(compound-positive) < minimum:

        minimum = compound-positive

        to_return = 'positive'

    if abs(compound-negative) < minimum:

        minimum = compound-negative

        to_return = 'negative'

    if abs(compound-neutral) < minimum:

        minimum = compound-neutral

        to_return = 'neutral'

    return to_return

    

messages = []

all_vader_output = []

original = {} #wait for mai to convert csv to readable dict

all_id_vader = {}

vader_dict = {'INFP': 0, 'INFJ': 0, 'INTP': 0, 'INTJ': 0, 'ENTP': 0, 'ENFP': 0,\

             'ISTP': 0, 'ISFP': 0, 'ENTJ': 0, 'ISFJ': 0, 'ESTP': 0, 'ESFP': 0,\

             'ESFJ': 0, 'ESFJ': 0, 'ESTJ': 0, 'ISTJ': 0, 'ENFJ': 0,} #add here all the keys/characters



for id_, val_arr in total_dict.items():

    clear_output(wait=True)

    message = val_arr[1]

    new_msg = pos_sentence(message)

    vader_output = run_vader(new_msg)

    vader_label =  vader_output_to_label(vader_output)

    #all_id_vader.append{id:vadel_label}

    messages.append(message)

    all_vader_output.append({'sen': vader_label})

    

    if vader_label == "negative":

        vader_dict[val_arr[0]] -= 1

    if vader_label == "positive":

        vader_dict[val_arr[0]] += 1

    #if neutral, do nothing

    print("Current progress: ", id_)

    

for character, score in vader_dict.items():

    if score < 0:

        print(character, " has a mean negative sentiment.")

    if score == 0:

        print(character, " has a mean neutral sentiment.")

    if score > 0:

        print(character, " has a mean positive sentiment.")
types = df.groupby('type').count()

posts = types["posts"]

for i,j in vader_dict.items():

    if j/posts[i] < 0.07:

        print(i, ' neutral')

    else:

        print(i, 'positive')
i = 0

arr_neg = []

for j in all_vader_output:

    if j['sen'] == 'negative':

        arr_neg.append(i)

    i+=1



key = list(total_dict.values())



types_neg = []

for i in arr_neg:

    types_neg.append(key[i][0])



i = 0

arr_pos = []

for j in all_vader_output:

    if j['sen'] == 'positive':

        arr_pos.append(i)

    i+=1



types_pos = []

for i in arr_pos:

    types_pos.append(key[i][0])



i = 0

arr_neu = []

for j in all_vader_output:

    if j['sen'] == 'neutral':

        arr_neu.append(i)

    i+=1



types_neu = []

for i in arr_neu:

    types_neu.append(key[i][0])



print('negative: ', Counter(types_neg))

print('positive: ', Counter(types_pos))

print('neutral: ', Counter(types_neu))

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction import DictVectorizer

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn import svm
# Combine all features into a list of dicts of features

features = [dict(**a,**b,**c,**d) for a,b,c,d in zip(pos_feature,len_posts,all_vader_output,entity_feature)]

# Extract labels

labels = df['type']

# Vectorize features

vec = DictVectorizer()

new_features = vec.fit_transform(features).toarray()
new_features.shape
def plot_2d_space(X, y, label='Classes'):   

    colors = ['#1F77B4', '#FF7F0E']

    markers = ['o', 's']

    for l, c, m in zip(np.unique(y), colors, markers):

        plt.scatter(

            X[y==l, 0],

            X[y==l, 1],

            c=c, label=l, marker=m

        )

    plt.title(label)

    plt.legend(loc='upper right')

    plt.show()
from sklearn.decomposition import PCA



pca = PCA(n_components=2)

X = pca.fit_transform(new_features)



plot_2d_space(X, labels, 'Imbalanced dataset (2 PCA components)')
#Fix the imbalance of the dataset

from imblearn.combine import SMOTETomek



smt = SMOTETomek(ratio='auto')

X_smt, y_smt = smt.fit_sample(X, labels)



plot_2d_space(X_smt, y_smt, 'SMOTE + Tomek links')
# Split data into training and test

posts_train, posts_test, type_train, type_test = train_test_split(

    X_smt,

    y_smt, 

    test_size = 0.20, # we use 80% for training and 20% for development

    random_state = 123

    ) 
lin_clf = svm.LinearSVC(dual=False)

lin_clf.fit(posts_train,type_train)

predicted_labels = lin_clf.predict(posts_test)
report = metrics.classification_report(y_true=type_test,y_pred=predicted_labels)
print('balanced dataset: ')

print(report)
# Split data into training and test

posts_train, posts_test, type_train, type_test = train_test_split(

    new_features,

    labels, 

    test_size = 0.20, # we use 80% for training and 20% for development

    random_state = 123

    ) 
lin_clf = svm.LinearSVC(dual=False)

lin_clf.fit(posts_train,type_train)

predicted_labels = lin_clf.predict(posts_test)
report = metrics.classification_report(y_true=type_test,y_pred=predicted_labels)
print('unbalanced dataset: ')

print(report)