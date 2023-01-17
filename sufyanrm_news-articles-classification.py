import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn.model_selection import cross_validate, cross_val_score, train_test_split, KFold, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

import time
import re
import pickle
from string import punctuation
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud

from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
df = pd.read_json('/kaggle/input/news-category-dataset/News_Category_Dataset_v2.json', lines=True)
df.head()
df.isna().sum()
# top 5 Categories of news in our datset
df.category.value_counts()[:5]
# our unique labels of text which are to be classified.
np.unique(df.category)
df['text'] = df['headline'] + df['short_description'] + df['authors']
df['label'] = df['category']
del df['headline']
del df['short_description']
del df['date']
del df['authors']
del df['link']
del df['category']
df.head(10)
df['text'].apply(lambda x: len(x.split(' '))).sum()
def get_simple_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
REMOVE_SPECIAL_CHARACTER = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS = re.compile('[^0-9a-z #+_]')

STOPWORDS = set(stopwords.words('english'))
punctuation = list(punctuation)
STOPWORDS.update(punctuation)

lemmatizer = WordNetLemmatizer()

def clean_text(text):
    # part 1
    text = text.lower() # lowering text
    text = REMOVE_SPECIAL_CHARACTER.sub('', text) # replace REPLACE_BY_SPACE symbols by space in text
    text = BAD_SYMBOLS.sub('', text) # delete symbols which are in BAD_SYMBOLS from text
    
    # part 2
    clean_text = []
    for w in word_tokenize(text):
        if w.lower() not in STOPWORDS:
            pos = pos_tag([w])
            new_w = lemmatizer.lemmatize(w, pos=get_simple_pos(pos[0][1]))
            clean_text.append(new_w)
    text = " ".join(clean_text)
    
    return text
df['text'] = df['text'].apply(clean_text)
df.to_csv('news_text_cleaned.csv', columns=['text', 'label'])
df.head(10)
df['text'].apply(lambda x: len(x.split(' '))).sum()
X=df.text
y=df.label

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
# Creating Models
models = [('Logistic Regression', LogisticRegression(max_iter=500)),('Random Forest', RandomForestClassifier()),
          ('Linear SVC', LinearSVC()), ('Multinomial NaiveBayes', MultinomialNB()), ('SGD Classifier', SGDClassifier())]

names = []
results = []
model = []
for name, clf in models:
    pipe = Pipeline([('vect', CountVectorizer(max_features=30000, ngram_range=(1, 2))),
                    ('tfidf', TfidfTransformer()),
                    (name, clf),
                    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    accuracy = accuracy_score(y_pred, y_test)
    
    names.append(name)
    results.append(accuracy)
    model.append(pipe)
    
    msg = "%s: %f" % (name, accuracy)
    print(msg)
# Logistic Regression
filename = 'model_lr.sav'
pickle.dump(model[0], open(filename, 'wb'))

# Linear SVC
filename = 'model_lin_svc.sav'
pickle.dump(model[2], open(filename, 'wb'))
lr_model = pickle.load(open('model_lin_svc.sav', 'rb'))
text1 = 'Could have been best all-rounder India ever produced in ODIs: Irfan Pathan'
text2 = "Ashwin Kkumar dances to Kamal Haasan's Annathe song on a treadmill. Actor is proud"
text3 = "The independent rights expert does not speak for the United Nations but reports her findings to it. Her report on targeted killings through armed drones — around half of which deals with the Soleimani case — is to be presented to the UN Human Rights Council session in Geneva on Thursday. The United States withdrew from the council in 2018. US President Donald Trump ordered the killing of Soleimani in a January 3 drone strike near Baghdad international airport. Soleimani, a national hero at home, was the worlds top terrorist and should have been terminated long ago, Trump said at the time."
print(lr_model.predict([text1, text2, text3]))