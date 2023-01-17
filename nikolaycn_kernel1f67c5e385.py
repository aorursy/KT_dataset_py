import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
%matplotlib inline

from tqdm import tqdm
# Датасет можно скачать здесь

#!wget https://www.dropbox.com/s/tg55q9mrziroyrs/train_subset.csv
data = pd.read_csv("../input/train_subset.csv", index_col='id')

data.head()
X = data[['title', 'description']].to_numpy()
y = data['Category'].to_numpy()

del data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from nltk.tokenize import WordPunctTokenizer


tokenizer = WordPunctTokenizer()


def preprocess(text: str) -> str:
    return ' '.join(tokenizer.tokenize(text.lower()))


text = 'Здраствуйте. Я, Кирилл. Хотел бы чтобы вы сделали игру, 3Д-экшон суть такова...'
print("before:", text,)
print("after:", preprocess(text),)
def preprocess_array(array):
    for i in range(len(array)):
        for j in range(len(array[0])):
              array[i][j] = preprocess(array[i][j])
    return array


X_test = preprocess_array(X_test)
X_train = preprocess_array(X_train)
def word_counter(array):
    voc = {}
    for i in range(len(array)):
        for j in range(len(array[0])):
            string = array[i][j].split()
            for k in string:
                if k not in voc:
                    voc[k] = 1
                else:
                    voc[k] += 1
    return voc
def text_to_bow(text: str) -> np.array:
    """
    Возвращает вектор, где для каждого слова из bow_vocabulary
    указано количество его употреблений
    """ 
    # Your code here
    text = text.split()
    answer = np.zeros((len(bow_vocabulary)))
    for w in range(len(text)):
        for i, word in enumerate(bow_vocabulary.keys()):
            if word == text[w]:
                answer[i] += 1
                break
    return answer
def items_to_bow(items: np.array, use_title=False) -> np.array:
    """ Для каждого товара возвращает вектор его bow """
    # Давайте строить bow только из description товара
    answer = np.zeros(shape=(len(items), len(bow_vocabulary)))
    if not use_title:
        for i in tqdm(range(len(items))):
            answer[i] = text_to_bow(items[i][1])
    else:
        for i in tqdm(range(len(items))):
            answer[i] = text_to_bow(items[i][0] + " " + items[i][1])
    return answer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
!pip install pymystem3
from pymystem3 import Mystem
m = Mystem()
m.lemmatize("Здравствуйте, ваше благородие")
def title_desc_lemmatizer(items: np.array):
    for i in tqdm(range(len(items))):
        items[i][0] = ' '.join(m.lemmatize(items[i][0]))
        items[i][1] = ' '.join(m.lemmatize(items[i][1]))
    return items


title_desc_lemmatizer(X_train)
title_desc_lemmatizer(X_test)
def docs_with_term():
    answer = np.zeros((len(bow_vocabulary)))
    for i in tqdm(range(len(answer))):
        for j in X_train_bow[:,i]:
            if j > 0:
                answer[i] += 1
    return answer
def text_to_tfidf(text: str) -> np.array:
    """
    Возвращает вектор, где для каждого слова из bow_vocabulary указан tf-idf
    """

    text = text.split()
    answer = np.zeros((len(bow_vocabulary)))
    for i, word in enumerate(bow_vocabulary.keys()):
        if word in text:
            tf = text.count(word) / len(text)
            idf = np.log(len(X_train) / count_arr[i])
            answer[i] = tf * idf
    return answer
def items_to_tfidf(items: np.array, use_title=False) -> np.array:
    """ Для каждого товара возвращает вектор его tfidf """
    # Давайте строить bow только из description товара
    answer = np.zeros(shape=(len(items), len(bow_vocabulary)))
    if not use_title:
        for i in tqdm(range(len(items))):
            answer[i] = text_to_tfidf(items[i][1])
    else:
        for i in tqdm(range(len(items))):
            answer[i] = text_to_tfidf(items[i][0] + " " + items[i][1])
    return answer
!wget https://www.dropbox.com/s/0x7oxso6x93efzj/ru.tar.gz
!tar -xzf ru.tar.gz
import gensim
from gensim.models.wrappers import FastText

model = FastText.load_fasttext_format('ru.bin')
# Эмбеддинг предложения -- сумма эмбеддингов токенов


def sentence_embedding(sentence: str) -> np.array:
    """
    Складывает вектора токенов строки sentence
    """

    embedding_dim = model['кек'].shape[0]
    features = np.zeros([embedding_dim], dtype='float32')
    
    for word in sentence.split():
        if word in model:
            features += model[word]
    
    return features
assert np.allclose(sentence_embedding('сдаётся уютный , тёплый гараж для стартапов в ml')[::50],
                   np.array([ 0.08189847,  0.07249198, -0.15601222,  0.03782297,  0.09215296, -0.23092946]))
def items_to_embed(items: np.array) -> np.array:
    """ Для каждого товара возвращает вектор его tfidf """
    # Давайте строить bow только из description товара
    answer = []
    for i in tqdm(range(len(items))):
            answer.append(sentence_embedding(items[i]))
    answer = np.array(answer)
    return answer
X_train1 = np.array([' '.join(line) for line in X_train])
X_test1 = np.array([' '.join(line) for line in X_test])
X_train_emb = items_to_embed(X_train1)
X_test_emb = items_to_embed(X_test1)
emb_model_lr = LogisticRegression(max_iter=100)
emb_model_lr.fit(X_train_emb, y_train)
acc_lr_emb = accuracy_score(emb_model_lr.predict(X_test_emb), y_test)
print(acc_lr_emb)
emb_model_svc = LinearSVC(max_iter=70)
emb_model_svc.fit(X_train_emb, y_train)
acc_svc_emb = accuracy_score(emb_model_svc.predict(X_test_emb), y_test)
print(acc_svc_emb)