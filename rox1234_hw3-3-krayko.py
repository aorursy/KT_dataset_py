import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
%matplotlib inline

from tqdm import tqdm
# Датасет можно скачать здесь

!wget https://www.dropbox.com/s/tg55q9mrziroyrs/train_subset.csv
data = pd.read_csv("train_subset.csv", index_col='id')

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
def tokenizer_punct(array):
    for i in range(len(array)):
        for n in range(len(array[i])):
            array[i][n] = preprocess(array[i][n])
    return array
print(tokenizer_punct(X_train))
assert X_train[10][1] == 'продам иж планета 3 , 76 год , ( стоит на старом учёте , документы утеряны ) на ходу , хорошее состояние , все интересующие вопросы по телефону ( с родной коляской на 3 тысячи дороже ) . торга не будет .'
bow_vocabulary = []
for i in range(len(X_train)):
        for n in range(len(X_train[i])):
            words = X_train[i][n].split()
            for word in words:
                bow_vocabulary.append(word)
from collections import Counter
bow = dict((Counter(bow_vocabulary)).most_common(10000))
print(bow)
assert sorted(bow)[::200] == ['!', '12500', '270', '700', 'by', 'gh', 'michael', 'sonata', 'ø', 'аудиоподготовка', 'большим', 'веса', 'воспроизведения', 'габариты', 'гтд', 'джинсами', 'доступность', 'загрузки', 'зимней', 'использовался', 'квартала', 'коммуникации', 'кошки', 'лакированные', 'магазин', 'металл', 'мск', 'натуральным', 'носке', 'одному', 'отвечаем', 'пассат', 'плотно', 'покраску', 'постоянные', 'примеры', 'просьба', 'размещайте', 'репетитор', 'сантехник', 'сидения', 'современного', 'стала', 'схема', 'тон', 'удлиненная', 'фасад', 'цветами', 'шея', 'эту']
def text_to_bow(text: str) -> np.array:
    """
    Возвращает вектор, где для каждого слова из bow_vocabulary
    указано количество его употреблений
    """ 
    text1 = preprocess(text)
    words = text1.split()
    lst_vec = []
    for element in bow:
        freq = words.count(element)
        lst_vec.append(freq)
    npArray = np.array(lst_vec)
    return npArray
assert np.allclose(np.where(text_to_bow("сдаётся уютный , тёплый гараж для стартапов в ml") != 0)[0],
                   np.array([   1,    4,   12,  565,  866, 1601, 2539, 4063])
)
def items_to_bow(items: np.array, use_title=False) -> np.array:
    """ Для каждого товара возвращает вектор его bow """
    big_massive = []
    for massive in items:
        description = massive[1]
        desc = preprocess(description)
        words = desc.split()
        lst_vec = []
        for element in bow:
            freq = words.count(element)
            lst_vec.append(freq)
        big_massive.append(lst_vec)
    npArray = np.array(big_massive)
    return npArray
assert np.allclose(np.where(items_to_bow([X_train[42]])[0] != 0),
                   np.array([   0, 1, 2, 5, 6, 7, 12, 27, 41, 49, 110,
                                189,  208,  221, 2032, 3052, 7179, 9568]),
)
X_train_bow = items_to_bow(X_train)
X_test_bow = items_to_bow(X_test)
X_train_bow.shape
X_train.shape
X_test_bow.shape
X_test.shape
import scipy
X_train_bow_sparsed = scipy.sparse.csr_matrix(X_train_bow)
X_test_bow_sparsed = scipy.sparse.csr_matrix(X_test_bow)
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

bow_model_lr = LogisticRegression(max_iter=100)
bow_model_lr.fit(X_train_bow_sparsed, y_train)
acc_lr_bow = accuracy_score(bow_model_lr.predict(X_test_bow_sparsed), y_test)
print(acc_lr_bow)

assert acc_lr_bow > 0.7
from sklearn.svm import LinearSVC

bow_model_svc = LinearSVC(max_iter=70)
bow_model_svc.fit(X_train_bow, y_train)
acc_svc_bow = accuracy_score(bow_model_svc.predict(X_test_bow_sparsed), y_test)
print(acc_svc_bow)

assert acc_svc_bow > 0.68
def title_to_bow(items: np.array) -> np.array:
    big_massive = []
    for massive in items:
        description = massive[0]
        desc = preprocess(description)
        words = desc.split()
        lst_vec = []
        for element in bow:
            freq = words.count(element)
            lst_vec.append(freq)
        big_massive.append(lst_vec)
    npArray = np.array(big_massive)
    return npArray
X_train_bow_title = title_to_bow(X_train)
X_test_bow_title = title_to_bow(X_test)
X_train_bow_title.shape
X_test_bow_title.shape
X_train_bow_full = X_train_bow_title + X_train_bow
X_test_bow_full = X_test_bow_title + X_test_bow
print(X_train_bow_full.shape)
print(X_test_bow_full.shape)
from sklearn.linear_model import LogisticRegression
bow_model = LogisticRegression(max_iter=100).fit(X_train_bow_full, y_train)
print(accuracy_score(bow_model.predict(X_test_bow_full), y_test))

assert accuracy_score(bow_model.predict(X_test_bow_full), y_test) > 0.7
from sklearn.svm import LinearSVC

bow_model = LinearSVC(max_iter=70).fit(X_train_bow_full, y_train)
print(accuracy_score(bow_model.predict(X_test_bow_full), y_test))

assert accuracy_score(bow_model.predict(X_test_bow_full), y_test) > 0.68
!pip3 install pymystem3
!pip install pymystem3
from pymystem3 import Mystem

m = Mystem()
m.lemmatize("Здравствуйте, ваше благородие")
def lemma_mystem(array):
    for i in range(len(array)):
        for n in range(len(array[i])):
            array[i][n] = ' '.join(m.lemmatize(str(array[i][n])))
    print(array)
lemma_mystem(X_train)
lemma_mystem(X_test)
mystem_bow_vocabulary = []
for i in range(len(X_train)):
        for n in range(len(X_train[i])):
            words = X_train[i][n].split()
            for word in words:
                mystem_bow_vocabulary.append(word)
mystem_bow = dict((Counter(mystem_bow_vocabulary)).most_common(10000))
print(mystem_bow)
def mystem_items_to_bow(items: np.array) -> np.array:
    big_massive = []
    for massive in items:
        description = massive[1]
        desc = preprocess(description) # тогда ниже не descripption, a desc
        words = description.split()
        lst_vec = []
        for element in mystem_bow:
            freq = words.count(element)
            lst_vec.append(freq)
        big_massive.append(lst_vec)
    npArray = np.array(big_massive)
    return npArray
X_train_mystem_bow = mystem_items_to_bow(X_train)
X_test_mystem_bow = mystem_items_to_bow(X_test)
print(X_train_mystem_bow)
print(X_train_mystem_bow.shape)
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
bow_model = LogisticRegression(max_iter=100).fit(X_train_mystem_bow, y_train)
print(accuracy_score(bow_model.predict(X_test_mystem_bow), y_test))
from sklearn.svm import LinearSVC
bow_model = LinearSVC(max_iter=70).fit(X_train_mystem_bow, y_train)
print(accuracy_score(bow_model.predict(X_test_mystem_bow), y_test))
def simple(massive):
    full = []
    for product in massive:
        half = []
        for sentence in product:
            words = sentence.split()
            for word in words:
                half.append(word)
        full.append(half)
    return full
X_train_conc = simple(X_train)
X_test_conc = simple(X_test)
print(X_train_conc[0])
print(X_train[0])
count_arr = {}

for el in bow:
    counter = 0
    counter_texts = 0
    for text in X_train_conc:
        counter += text.count(el)
        if el in text:
            counter_texts += 1
    count_arr[el] = [counter, counter_texts]
print(count_arr)
def text_to_tfidf(arr: np.array) -> np.array:
    """
    Возвращает вектор, где для каждого слова из bow_vocabulary указан tf-idf
    """
    arr_tfidf = []
    keys = list(bow.keys())
    d = len(arr)
    for text in arr:
        vec = np.zeros(10000)
        for word in keys:
            if word in text:
                i = keys.index(word)
                tf = text.count(word)/len(text)
                if count_arr[word][1] == 0:
                    idf = np.log(d/1)
                else:
                    idf = np.log(d/count_arr[word][1])
                vec[i] = tf * idf
        arr_tfidf.append(vec)
    return np.array(arr_tfidf)
tf_idf_train = text_to_tfidf(X_train_conc)
print(tf_idf_train.shape)
print(X_test_conc[0])
tf_idf_test = text_to_tfidf(X_test_conc)
print(tf_idf_test.shape)
from sklearn.linear_model import LogisticRegression
bow_model = LogisticRegression(max_iter=100).fit(tf_idf_train, y_train)
print(accuracy_score(bow_model.predict(tf_idf_test), y_test))
from sklearn.svm import LinearSVC

bow_model = LinearSVC(max_iter=100).fit(tf_idf_train, y_train)
print(accuracy_score(bow_model.predict(tf_idf_test), y_test))
from sklearn.preprocessing import normalize

tf_idf_train = normalize(tf_idf_train )
tf_idf_test = normalize(tf_idf_test)
from sklearn.linear_model import LogisticRegression
bow_model = LogisticRegression(max_iter=100).fit(tf_idf_train, y_train)
print(accuracy_score(bow_model.predict(tf_idf_test), y_test))
from sklearn.svm import LinearSVC
bow_model = LinearSVC(max_iter=100).fit(tf_idf_train, y_train)
print(accuracy_score(bow_model.predict(tf_idf_test), y_test))
from sklearn.feature_extraction.text import HashingVectorizer
hashvectrzr = HashingVectorizer()
X_train_sum = [item[0] + ' ' + item[1] for item in X_train]
X_test_sum = [item[0] + ' ' + item[1] for item in X_test]
Hashvec_Train = hashvectrzr.fit_transform(X_train_sum)
Hashvec_Test = hashvectrzr.fit_transform(X_test_sum)
X_train_sum[0]
print(Hashvec_Train)
from sklearn.linear_model import LogisticRegression
bow_model = LogisticRegression(max_iter=100).fit(Hashvec_Train, y_train)
print(accuracy_score(bow_model.predict(Hashvec_Test), y_test))
from sklearn.svm import LinearSVC
bow_model = LinearSVC(max_iter=100).fit(Hashvec_Train, y_train)
print(accuracy_score(bow_model.predict(Hashvec_Test), y_test))
!wget https://www.dropbox.com/s/0x7oxso6x93efzj/ru.tar.gz
!tar -xzf ru.tar.gz
!ls
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
embedding_X_train = [] 
for text in X_train_conc:
    embedding_X_train.append(sentence_embedding(' '.join(text)))
print(len(embedding_X_train))
print(len(embedding_X_train[0]))
print(embedding_X_train[0])
embedding_X_test = [] 
for text in X_test_conc:
    embedding_X_test.append(sentence_embedding(' '.join(text)))
print(len(embedding_X_test))
print(len(embedding_X_test[0]))
print(embedding_X_test[0])
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
bow_model = LogisticRegression(max_iter=100).fit(embedding_X_train, y_train)
print(accuracy_score(bow_model.predict(embedding_X_test), y_test))
bow_model = LinearSVC(max_iter=100).fit(embedding_X_train, y_train)
print(accuracy_score(bow_model.predict(embedding_X_test), y_test))



















