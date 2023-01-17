import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
%matplotlib inline

from tqdm import tqdm
# Датасет можно скачать здесь

#!wget https://www.dropbox.com/s/tg55q9mrziroyrs/train_subset.csv
data = pd.read_csv("../input/train-subset/train_subset.csv", index_col='id')

data.head()
data.shape
X = data[['title', 'description']].to_numpy()
y = data['Category'].to_numpy()

del data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train[:5]
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
X_train[:10]
assert X_train[10][1] == 'продам иж планета 3 , 76 год , ( стоит на старом учёте , документы утеряны ) на ходу , хорошее состояние , все интересующие вопросы по телефону ( с родной коляской на 3 тысячи дороже ) . торга не будет .'
# Your code here
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


bow_vocabulary = word_counter(X_train)
bow_vocabulary =np.array(sorted(bow_vocabulary.items(), key = lambda y: y[1], reverse = True)[:10000])
bow_vocabulary = dict(zip(bow_vocabulary[:,0], bow_vocabulary[:,0]))
print(sorted(bow_vocabulary)[::200])
assert sorted(bow_vocabulary)[::200] == ['!', '12500', '270', '700', 'by', 'gh', 'michael', 'sonata', 'ø', 'аудиоподготовка', 'большим', 'веса', 'воспроизведения', 'габариты', 'гтд', 'джинсами', 'доступность', 'загрузки', 'зимней', 'использовался', 'квартала', 'коммуникации', 'кошки', 'лакированные', 'магазин', 'металл', 'мск', 'натуральным', 'носке', 'одному', 'отвечаем', 'пассат', 'плотно', 'покраску', 'постоянные', 'примеры', 'просьба', 'размещайте', 'репетитор', 'сантехник', 'сидения', 'современного', 'стала', 'схема', 'тон', 'удлиненная', 'фасад', 'цветами', 'шея', 'эту']
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
assert np.allclose(np.where(text_to_bow("сдаётся уютный , тёплый гараж для стартапов в ml") != 0)[0],
                   np.array([   1,    4,   12,  565,  866, 1601, 2539, 4063])
)
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
assert np.allclose(np.where(items_to_bow([X_train[42]])[0] != 0),
                   np.array([   0, 1, 2, 5, 6, 7, 12, 27, 41, 49, 110,
                                189,  208,  221, 2032, 3052, 7179, 9568]),
)
X_train_bow = items_to_bow(X_train)
X_test_bow = items_to_bow(X_test)
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
bow_model_lr = LogisticRegression(max_iter=100)
bow_model_lr.fit(X_train_bow, y_train)
acc_lr_bow = accuracy_score(bow_model_lr.predict(X_test_bow), y_test)
print(acc_lr_bow)

assert acc_lr_bow > 0.7
from sklearn.svm import LinearSVC
bow_model_svc = LinearSVC(max_iter=70)
bow_model_svc.fit(X_train_bow, y_train)
acc_svc_bow = accuracy_score(bow_model_svc.predict(X_test_bow), y_test)
print(acc_svc_bow)

assert acc_svc_bow > 0.68
X_train_bow = items_to_bow(X_train, use_title=True)
X_test_bow = items_to_bow(X_test, use_title=True)
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
bow_vocabulary = word_counter(X_train)
bow_vocabulary = np.array(sorted(bow_vocabulary.items(), key = lambda y: y[1], reverse = True)[:10000])
bow_vocabulary = dict(zip(bow_vocabulary[:,0], bow_vocabulary[:,0]))
X_train_bow = items_to_bow(X_train, use_title=True)
X_test_bow = items_to_bow(X_test, use_title=True)
bow_model_lr = LogisticRegression(max_iter=100)
bow_model_lr.fit(X_train_bow, y_train)
acc_lr_bow = accuracy_score(bow_model_lr.predict(X_test_bow), y_test)
print(acc_lr_bow)
bow_model_svc = LinearSVC(max_iter=70)
bow_model_svc.fit(X_train_bow, y_train)
acc_svc_bow = accuracy_score(bow_model_svc.predict(X_test_bow), y_test)
print(acc_svc_bow)