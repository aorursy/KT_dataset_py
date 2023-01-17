!pip install pymorphy2

import numpy as np
import pandas as pd
import nltk as nltk
import pymorphy2
import string  

from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_recall_fscore_support

analyzer = pymorphy2.MorphAnalyzer()
extra_matrix = []

def get_normal_form_of_single_text(text):
    normalized_text = ""
    tokens = nltk.word_tokenize(text)
    nouns_verbs_adjectives_adverbs_puncto = [0, 0, 0, 0, 0]
    
    for token in tokens:
        pos = analyzer.parse(token)[0].tag

        if 'NOUN' in pos:
            nouns_verbs_adjectives_adverbs_puncto[0] +=1
        elif 'VERB' in pos:
            nouns_verbs_adjectives_adverbs_puncto[1] +=1
        elif 'ADJF' in pos or 'ADJS' in pos:
            nouns_verbs_adjectives_adverbs_puncto[2] +=1
        elif 'ADVB' in pos:
            nouns_verbs_adjectives_adverbs_puncto[3] +=1
        elif 'PNCT' in pos:
            nouns_verbs_adjectives_adverbs_puncto[4] +=1
            
        normalized_text = normalized_text + " " + analyzer.parse(token)[0].normal_form
    extra_matrix.append(nouns_verbs_adjectives_adverbs_puncto)
    return normalized_text
df = pd.read_csv("../input/allreviews/all-reviews.csv", names=["name", "review", "label"], skiprows=1)
df['review'] = df['review'].map(lambda x: get_normal_form_of_single_text(x))
my_films = ["Побег из Шоушенка", "Поймай меня, если сможешь", "Престиж"]
my_reviews = df[df.name.isin(my_films)]
train_reviews = df[~df.name.isin(my_films)]

train_reviews.head()
import numpy as np

count = CountVectorizer()
bag_of_words = count.fit_transform(train_reviews['review']).toarray()
bag_of_words = np.concatenate((bag_of_words, extra_matrix[:len(bag_of_words)]), axis =1)

pd.DataFrame(bag_of_words, columns=count.get_feature_names().extend(["noun", "verb", "adj", "adv", "puncto"]))

svm = LinearSVC(random_state=0, tol=1e-5, max_iter = 100000)
svm.fit(bag_of_words, train_reviews.label)
my_reviews.head()
test_bag_of_words = count.transform(my_reviews['review']).toarray()
test_bag_of_words = np.concatenate((test_bag_of_words, extra_matrix[len(bag_of_words):(len(bag_of_words) + len(test_bag_of_words))]), axis =1)
svm.score(test_bag_of_words, my_reviews['label'])
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(max_depth=20, random_state=0)
clf.fit(bag_of_words, train_reviews.label)
clf.score(test_bag_of_words, my_reviews['label'])
svm_predicted = svm.predict(test_bag_of_words)
svm_precision_recall_fscore = precision_recall_fscore_support(my_reviews['label'].values, svm_predicted)
print(f"Precision(-1, 0, 1) = {svm_precision_recall_fscore[0]}")
print(f"Recall(-1, 0, 1) = {svm_precision_recall_fscore[1]}")
print(f"Fscore(-1, 0, 1) = {svm_precision_recall_fscore[2]}")
rf_predicted = clf.predict(test_bag_of_words)
rf_precision_recall_fscore = precision_recall_fscore_support(my_reviews['label'].values, rf_predicted)
print(f"Precision(-1, 0, 1) = {rf_precision_recall_fscore[0]}")
print(f"Recall(-1, 0, 1) = {rf_precision_recall_fscore[1]}")
print(f"Fscore(-1, 0, 1) = {rf_precision_recall_fscore[2]}")
