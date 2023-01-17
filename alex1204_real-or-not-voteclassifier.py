import numpy as np # linearalgebra and data handling

import pandas as pd # data handling and manipulation 

import spacy # lemmatizing of the text data

import re, unicodedata # cleaning of the text data
train_df = pd.read_csv('../input/nlp-getting-started/train.csv') # loading the train data

train_df
docs = train_df.text

train_target = train_df.target
nlp = spacy.load('en_core_web_lg', parse=True, tag=True, entity=True)
def remove_URL(text):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',text)



def remove_html(text):

    html=re.compile(r'<.*?>')

    return html.sub(r'',text)



def remove_emoji(text):

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', text)
def contractions(text):

    text = re.sub(r"i'm", "i am", text)

    text = re.sub(r"you'll", "you will", text)

    text = re.sub(r"i'll", "i will", text)

    text = re.sub(r"she'll", "she will", text)

    text = re.sub(r"he'll", "he will", text)

    text = re.sub(r"he's", "he is", text)

    text = re.sub(r"she's", "she is", text)

    text = re.sub(r"that's", "that is", text)

    text = re.sub(r"what's", "what is", text)

    text = re.sub(r"where's", "where is", text)

    text = re.sub(r"there's", "there is", text)

    text = re.sub(r"here's", "here is", text)

    text = re.sub(r"who's", "who is", text)

    text = re.sub(r"how's", "how is", text)

    text = re.sub(r"\'ll", " will", text)

    text = re.sub(r"\'ve", " have", text)

    text = re.sub(r"\'re", " are", text)

    text = re.sub(r"\'d", " would", text)

    text = re.sub(r"can't", "cannot", text)

    text = re.sub(r"won't", "will not", text)

    text = re.sub(r"don't", "do not", text)

    text = re.sub(r"shouldn't", "should not", text)

    text = re.sub(r"n't", " not", text)

    return text
normalized_docs = []

for doc in docs:

    doc = doc.lower() # characters to lowercase

    doc = remove_URL(doc)

    doc = remove_html(doc)

    doc = remove_emoji(doc)

    doc = contractions(doc)

    doc = re.sub(r'[0-9]', '', doc) # removing numbers

    doc = re.sub(r'[&(),.#://?!]', '', doc) # verwijderen speciale tekens

    doc = re.sub(r'[\r|\n|\r\n]+', ' ',doc) # Verwijderen extra newlines

    doc = re.sub(' +', ' ', doc) # Verwijderen overbodige whitespace

    doc = unicodedata.normalize('NFKD', doc).encode('ascii', 'ignore').decode('utf-8', 'ignore') # verwijderen accented tekens

    doc = nlp(doc)

    doc = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in doc]) # Lemmatizeren van woorden

    normalized_docs.append(doc)

print('Docs normalized')
test_df = pd.read_csv('../input/nlp-getting-started/test.csv')

test_df
new_docs = test_df.text
normalized_new_docs = []

for doc in new_docs:

    doc = doc.lower() #verkleinen

    doc = contractions(doc)

    doc = remove_URL(doc)

    doc = remove_html(doc)

    doc = remove_emoji(doc)

    doc = re.sub(r'[0-9]', '', doc) # verwijderen cijfers

    doc = re.sub(r'[&(),.#]', '', doc) # verwijderen speciale tekens

    doc = re.sub(r'[\r|\n|\r\n]+', ' ',doc) # Verwijderen extra newlines

    doc = re.sub(' +', ' ', doc) # Verwijderen overbodige whitespace

    doc = unicodedata.normalize('NFKD', doc).encode('ascii', 'ignore').decode('utf-8', 'ignore') # verwijderen accented tekens

    doc = nlp(doc)

    doc = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in doc]) # Lemmatizeren van woorden

    normalized_new_docs.append(doc)

print('New docs normalized')
from sklearn.feature_extraction.text import TfidfVectorizer



tv = TfidfVectorizer(use_idf=True, min_df=10, max_df=0.5, smooth_idf = True)

tv_train_features = tv.fit_transform(normalized_docs)

tv_test_features = tv.transform(normalized_new_docs)



print('TFIDF model: Train features shape: {}'.format(tv_train_features.shape))

print('TFIDF model: Test features shape: {}'.format(tv_test_features.shape))
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score



knn = KNeighborsClassifier(n_neighbors=99)

knn.fit(tv_train_features, train_target)

cv_scores = cross_val_score(knn, tv_train_features, train_target, cv=5)

print('CV Accuracy (5-fold) of K nearest neighbours:', cv_scores)

print('Mean CV Accuracy of K nearest neighbours:', np.mean(cv_scores))
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import cross_val_score



mnb = MultinomialNB(alpha = 0.01)

mnb.fit(tv_train_features, train_target)

cv_scores = cross_val_score(mnb, tv_train_features, train_target, cv=5, scoring='f1')

print('CV Accuracy (5-fold) of Multinomial Naive Bayes:', cv_scores)

print('Mean CV Accuracy of Multinomial Naive Bayes:', np.mean(cv_scores))
from sklearn.naive_bayes import ComplementNB



cnb = ComplementNB(alpha = 1)

cnb.fit(tv_train_features, train_target)

cv_scores = cross_val_score(cnb, tv_train_features, train_target, cv=5, scoring='f1')

print('CV Accuracy (5-fold) of Complement Naive Bayes:', cv_scores)

print('Mean CV Accuracy of Complement Naive Bayes:',np.mean(cv_scores))
from sklearn.svm import LinearSVC



svm = LinearSVC(penalty='l2', C=.1, random_state=42)

svm.fit(tv_train_features, train_target)

cv_scores = cross_val_score(svm, tv_train_features, train_target, cv=5, scoring='f1')

print('CV Accuracy (5-fold) of Support Vector Machine:', cv_scores)

print('Mean CV Accuracy of Support Vector Machine:', np.mean(cv_scores))
from sklearn.linear_model import SGDClassifier



sgd = SGDClassifier(max_iter=1000, tol=0.001)

sgd.fit(tv_train_features, train_target)

cv_scores = cross_val_score(sgd, tv_train_features, train_target, cv=5)

print('CV Accuracy (5-fold) of Support Vector Machine:', cv_scores)

print('Mean CV Accuracy of Support Vector Machine:', np.mean(cv_scores))
conf = []

labels = []

THRESHOLD = 2

for doc in tv_test_features:

    votes = 0

    for clf in [svm, mnb, knn, cnb, sgd]:

        vote = clf.predict(doc)

        votes += vote

    if votes > 2:

        labels.append(1)

        conf.append(votes/5)

    else:

        labels.append(0)

        conf.append(1 - votes/5)
predictions = labels
output_df = pd.DataFrame({'id':test_df.id, 'target':predictions, 'confidence':conf})

output_df
output_df[['id','target']].to_csv('submission.csv', index=False)