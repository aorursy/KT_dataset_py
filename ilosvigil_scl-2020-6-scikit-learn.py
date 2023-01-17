!pip install pyenchant pysastrawi
!wget http://archive.ubuntu.com/ubuntu/pool/main/libr/libreoffice-dictionaries/hunspell-id_6.4.3-1_all.deb

!dpkg -i hunspell-id_6.4.3-1_all.deb
!apt update && apt install -y enchant libenchant1c2a hunspell hunspell-en-us libhunspell-1.6-0
import re

import os

import gc

import random



import numpy as np

import pandas as pd

import sklearn

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

import nltk

import enchant
!pip freeze > requirements.txt
print('Numpy version:', np.__version__)

print('Pandas version:', pd.__version__)

print('Scikit-Learn version:', sklearn.__version__)

print('Matplotlib version:', matplotlib.__version__)

print('Seaborn version:', sns.__version__)

print('NLTK version:', nltk.__version__)
SEED = 42



os.environ['PYTHONHASHSEED']=str(SEED)

random.seed(SEED)

np.random.seed(SEED)
nltk.download('wordnet')
!ls -lha /kaggle/input

!ls -lha /kaggle/input/student-shopee-code-league-sentiment-analysis
df_train = pd.read_csv('/kaggle/input/student-shopee-code-league-sentiment-analysis/train.csv')

df_train.sample(10)
df_train2 = pd.read_csv('/kaggle/input/shopee-reviews/shopee_reviews.csv')



def to_int(r):

    try:

        return np.int32(r)

    except:

        return np.nan



df_train2['label'] = df_train2['label'].apply(to_int)

df_train2 = df_train2.dropna()

df_train2['label'] = df_train2['label'].astype(np.int32)

df_train2
df_test = pd.read_csv('/kaggle/input/student-shopee-code-league-sentiment-analysis/test.csv')

df_test.sample(10)
X_train = pd.concat([df_train['review'], df_train2['text']], axis=0)

X_train = X_train.reset_index(drop=True)

y_train = pd.concat([df_train['rating'], df_train2['label']], axis=0)

y_train = y_train.reset_index(drop=True)



X_test = df_test['review']
rating_count = y_train.value_counts().sort_index().to_list()

total_rating = sum(rating_count)

lowest_rating_count = min(rating_count)

rating_weight = [lowest_rating_count/rc for rc in rating_count]



print(rating_count)

print(total_rating)

print(rating_weight)
class_weight = np.empty((total_rating,))

for i in range(total_rating):

    class_weight[i] = rating_weight[y_train[i] - 1]
from nltk.stem import WordNetLemmatizer

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory



lemmatizer = WordNetLemmatizer() # for en

factory = StemmerFactory() # for id

stemmer = factory.create_stemmer() # for id



tweet_tokenizer = nltk.tokenize.TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)



eng_dict = enchant.Dict('en')

ind_dict = enchant.Dict('id_ID')



def remove_char(text):

    text = re.sub(r'[^a-z ]', ' ', text)

    return text





def stem_lemma(tokens):

    new_token = []

    for token in tokens:

        if eng_dict.check(token):

            new_token.append(lemmatizer.lemmatize(token))

        elif ind_dict.check(token):

            new_token.append(stemmer.stem(token))

        else:

            new_token.append(token)

    return new_token



def upper_or_lower(tokens):

    new_token = []

    for token in tokens:

        total_lower = len(re.findall(r'[a-z]',token))

        total_upper = len(re.findall(r'[A-Z]',token))

        if total_lower == 0 or total_upper == 0:

            new_token.append(token)

        elif total_lower > total_upper:

            new_token.append(token.lower())

        else:

            new_token.append(token.upper())

    return new_token

    



def preprocess(X):

    X = X.apply(tweet_tokenizer.tokenize)

    X = X.apply(lambda token: [t for t in token if t != ''])

    X = X.apply(upper_or_lower)

    X = X.apply(stem_lemma)

#     X = X.apply(lambda token: ' '.join(token)) # need to join token because sklearn tf-idf only accept string, not list of string

    

#     X = X.apply(remove_char)

    return X
X_train = preprocess(X_train)

X_test = preprocess(X_test)
X_train.sample(10)
from sklearn.feature_extraction.text import TfidfVectorizer



bow_vectorizer = TfidfVectorizer(lowercase=False, ngram_range=(1,2), analyzer=lambda t:t, min_df=5, sublinear_tf=True)



X_train = bow_vectorizer.fit_transform(X_train)

X_test = bow_vectorizer.transform(X_test)

print(X_train.shape)

print(X_test.shape)
from sklearn.metrics import classification_report, f1_score, confusion_matrix



def predict(model, X):

    y = model.predict(X)

    return y



def metrics(y_true, y_pred):

    print('F1 Score :', f1_score(y_true, y_pred, average='macro'))

    print(classification_report(y_true, y_pred))



    cm = confusion_matrix(y_true, y_pred)

    cm = pd.DataFrame(cm, [1,2,3,4,5], [1,2,3,4,5])



    sns.heatmap(cm, annot=True, cmap="YlGnBu", fmt="d")

    plt.show()
from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()

clf.fit(X_train, y_train, class_weight)
y_train_pred = predict(clf, X_train)

metrics(y_train, y_train_pred)
y_test_pred = predict(clf, X_test)



df_submission = pd.concat([df_test['review_id'], pd.Series(y_test_pred, name='rating')], axis=1)

df_submission.to_csv('submission_MultinomialNB.csv', index=False)



df_submission
from sklearn.naive_bayes import ComplementNB

clf = ComplementNB()

clf.fit(X_train, y_train, class_weight)
y_train_pred = predict(clf, X_train)

metrics(y_train, y_train_pred)
y_test_pred = predict(clf, X_test)



df_submission = pd.concat([df_test['review_id'], pd.Series(y_test_pred, name='rating')], axis=1)

df_submission.to_csv('submission_ComplementNB.csv', index=False)



df_submission
# from sklearn.ensemble import RandomForestClassifier



# clf = RandomForestClassifier(random_state=SEED)

# clf.fit(X_train, y_train)
# y_train_pred = predict(clf, X_train)

# metrics(y_train, y_train_pred)
# y_test_pred = predict(clf, X_test)



# df_submission = pd.concat([df_test['review_id'], pd.Series(y_test_pred, name='rating')], axis=1)

# df_submission.to_csv('submission.csv', index=False)



# df_submission
# from sklearn.svm import SVC



# clf = SVC(kernel='rbf', C=1, cache_size=10240)

# clf.fit(X_train, y_train)
# y_train_pred = predict(clf, X_train)

# metrics(y_train, y_train_pred)
# y_test_pred = predict(clf, X_test)



# df_submission = pd.concat([df_test['review_id'], pd.Series(y_test_pred, name='rating')], axis=1)

# df_submission.to_csv('submission.csv', index=False)



# df_submission