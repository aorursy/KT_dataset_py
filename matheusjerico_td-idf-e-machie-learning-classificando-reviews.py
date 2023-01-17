import numpy as np
import pandas as pd
import re
import unicodedata
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Markdown, display
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk import stem

# ! unzip -n ../dataset/imdb-ptbr.zip  -d ../dataset
# ! ls ../dataset/
dataset = pd.read_csv("../input/imdb-ptbr/imdb-reviews-pt-br.csv")
dataset.drop(columns=['text_en', 'id'], inplace=True)
dataset.head()
dataset.tail()
dataset.shape
display(Markdown('> '+dataset['text_pt'][2]))
dataset.isnull().sum()
blk = []

for index, text, label in dataset.itertuples():
    if type(text) == str:
        if text.isspace():
            blk.append(i)

print(f"{len(blk)}, vazios: {blk}")
dataset['sentiment'].value_counts()
BAD_SYMBOLS_RE = re.compile(r'[^0-9a-z]')
STOPWORDS = set(stopwords.words('portuguese'))
stemmer = stem.RSLPStemmer()
def strip_accents(text):
    try:
        text = unicode(text, 'utf-8')
    except (TypeError, NameError):  # unicode is a default on python 3
        pass
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore')
    text = text.decode("utf-8")

    return str(text)

def preprocess_stemming_text(text):
    text = text.lower()

    text = strip_accents(text)
    text = BAD_SYMBOLS_RE.sub(' ', text)
    text = ' '.join(stemmer.stem(word)
                    for word in text.split() if word not in STOPWORDS)

    return text 
dataset['text_pt_stemm'] = dataset['text_pt'].apply(preprocess_stemming_text)
dataset.head()
X = dataset['text_pt_stemm']
y = dataset['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 17)
clf_nb = Pipeline([('TF-IDF', TfidfVectorizer(lowercase=True, 
                                              strip_accents='unicode',
                                              stop_words=stopwords.words('portuguese'))),
                   ('Classificador', MultinomialNB()),
])
clf_svc = Pipeline([('TF-IDF', TfidfVectorizer(lowercase=True, 
                                               strip_accents='unicode',
                                               stop_words=stopwords.words('portuguese'))),
                   ('Classificador', LinearSVC()),
])
clf_rf = Pipeline([('TF-IDF', TfidfVectorizer(lowercase=True, 
                                              strip_accents='unicode', 
                                              stop_words=stopwords.words('portuguese'))),
                   ('Classificador', RandomForestClassifier()),
])
clf_nb.fit(X_train, y_train)
preds = clf_nb.predict(X_test)
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt="d");
ax.set_title("Matriz de Confusão - Naive Bayes", fontsize=20)
ax.set_ylabel('Classe Verdadeira', fontsize=15)
ax.set_xlabel('Classe Predita', fontsize=15)
print(classification_report(y_test,preds))
print(accuracy_score(y_test,preds))
clf_svc.fit(X_train, y_train)
preds = clf_svc.predict(X_test)
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt="d");
ax.set_title("Matriz de Confusão - Linear SVC", fontsize=20)
ax.set_ylabel('Classe Verdadeira', fontsize=15)
ax.set_xlabel('Classe Predita', fontsize=15)
print(classification_report(y_test,preds))
print(accuracy_score(y_test,preds))
clf_rf.fit(X_train, y_train)
preds = clf_rf.predict(X_test)
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt="d");
ax.set_title("Matriz de Confusão - Random Forest", fontsize=20)
ax.set_ylabel('Classe Verdadeira', fontsize=15)
ax.set_xlabel('Classe Predita', fontsize=15)
print(accuracy_score(y_test,preds))
avaliacao_supernatural = "A série Supernatural é longa, mas é muito boa e emocionante. \
Você se apega aos personagens, sofre por eles.. É uma série cheia de altos e baixos."
avaliacao_purge = "A série é chata, o enredo é péssimo, horrível, Não gostei da śerie"
print(clf_svc.predict([avaliacao_supernatural]))
print(clf_svc.predict([avaliacao_purge]))