import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import gc



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# читаем данные

df = pd.read_csv("../input/sentiment140/training.1600000.processed.noemoticon.csv", 

                 usecols=[0,5],

                 sep=',',

                 header=None,

                 encoding='latin',

                 dtype={0:int, 5:object})



# названия колонок

df.columns = ['target', 'text']



df.head()
df.info()
# количество градаций target

df.loc[:,'target'].value_counts()
# преобразуем target в 0/1 (0 - негативные эмоции (0), 1 - позитивные (4))                                        

df['target'] = df['target'].apply(lambda x: 0 if x == 0 else 1)



# проверяем: количество градаций target после переименования

df.loc[:,'target'].value_counts()
# проверяем пропуски в cтолбцах

df.isna().sum()
# описание данных

print(df.groupby('target')['target'].count())



# процент target=1

target_count = df[df['target'] == 1]['target'].count()

total = df['target'].count()

target_share = target_count/total

print("Доля данных, показывающих целевую группу \"target=1\" {0:.2f}".format(target_share))



# гистограмма

df[df['target'] == 0]['target'].astype(int).hist(label='Негативные', grid = False, bins=1, rwidth=0.8)

df[df['target'] == 1]['target'].astype(int).hist(label='Позитивные', grid = False, bins=1, rwidth=0.8)

plt.xticks((0,1),('Негативные', 'Позитивные'))

plt.show()
df = df.sample(frac=0.001).reset_index(drop=True)

df.shape
from sklearn.model_selection import train_test_split



# разделение выборки на X и y

X = df['text']

y = df['target']



# разделение выборки на train и test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,random_state = 42)
# regular expressions library

import re



def text_clean(text):

    # преобразуем текст в нижний регистр

    text.lower()

    # преобразуем https:// и т.п. адреса в текст "URL"

    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',text)

    # преобразуем имя пользователя @username в "AT_USER"

    text = re.sub('@[^\s]+','AT_USER', text)

    # преобразуем множественые пробелы в один пробел

    text = re.sub('[\s]+', ' ', text)

    # преобразуем хэштег #тема в "тема"

    text = re.sub(r'#([^\s]+)', r'\1', text)

    return text



X_train = X_train.apply(text_clean)

X_test = X_test.apply(text_clean)

X_train
import nltk

from nltk.corpus import stopwords

import string



stop = set(stopwords.words('english'))

punctuation = list(string.punctuation)

stop.update(punctuation)
# функция для определения прилагательных (ADJ), глаголов (VERB), существительных (NOUN) и наречий (ADV)

from nltk.corpus import wordnet as wn



def get_simple_pos(tag):

    if tag.startswith('J'):

        return wn.ADJ

    elif tag.startswith('V'):

        return wn.VERB

    elif tag.startswith('N'):

        return wn.NOUN

    elif tag.startswith('R'):

        return wn.ADV

    else:

        return wn.NOUN



# Лемматизация 

from nltk.stem import WordNetLemmatizer

from nltk import pos_tag



# (отбрасываем всё лишнее в предложении, приводим слова к нормальной формеи получаем их список)

# 'Once upone a time a man walked into a door' -> ['upone', 'time', 'man', 'walk', 'door']

lemmatizer = WordNetLemmatizer()

def lemmatize_words(text):

    final_text = []

    for i in text.split():

        if i.strip().lower() not in stop:

            pos = pos_tag([i.strip()])

            word = lemmatizer.lemmatize(i.strip(),get_simple_pos(pos[0][1]))

            final_text.append(word.lower())

    return final_text   



# Объединяем лемматизированный список в предложение

# ['upone', 'time', 'man', 'walk', 'door'] -> 'upone time man walk door '

def join_text(text):

    string = ''

    for i in text:

        string += i.strip() +' '

    return string



# Запуск лемматизации и создание нового поля с лемматизированными предложениями 'text_lemma'

X_train = X_train.apply(lemmatize_words).apply(join_text)

X_test = X_test.apply(lemmatize_words).apply(join_text)



# проверка

X_train
from sklearn.feature_extraction.text import TfidfVectorizer



# Присвоение весов словам с использованием TfidfVectorizer

tv = TfidfVectorizer(min_df=0,max_df=1,use_idf=True,ngram_range=(1,2))

tv_X_train = tv.fit_transform(X_train)

tv_X_test = tv.transform(X_test)



print('TfidfVectorizer_train:', tv_X_train.shape)

print('TfidfVectorizer_test:', tv_X_test.shape)
# перевернём столбцы со словами из sparse матрицы с одним столбцом в numpy массив со многими столбцами

tv_X_train = tv_X_train.toarray()

tv_X_test = tv_X_test.toarray()
from sklearn.preprocessing import StandardScaler



# Нормализуем данные в tv_X_train/test

scaler_tv = StandardScaler(copy=False)



tv_X_train_scaled = scaler_tv.fit_transform(tv_X_train)

tv_X_test_scaled = scaler_tv.transform(tv_X_test)



# трансформируем tv_X_train/test sparse numpy матрицы в pandas Data Frame

tv_X_train_pd_scaled = pd.DataFrame(data=tv_X_train_scaled, 

                             index=X_train.index, 

                             columns=np.arange(0, np.size(tv_X_train_scaled,1)))

tv_X_test_pd_scaled = pd.DataFrame(data=tv_X_test_scaled, 

                             index=X_test.index, 

                             columns=np.arange(0, np.size(tv_X_test_scaled,1)))



# проверяем

tv_X_train_pd_scaled.shape
tv_X_train = tv_X_train_pd_scaled

tv_X_test = tv_X_test_pd_scaled
tv_X_train.loc[:,:3]
from sklearn.preprocessing import PolynomialFeatures



def create_polinomial(X, degree = 2):

    return PolynomialFeatures(degree).fit_transform(X)



X_train_poly = create_polinomial(tv_X_train.loc[:,:3], 2)

X_test_poly = create_polinomial(tv_X_test.loc[:,:3], 2)



# проверяем

print(X_train_poly.shape, X_test_poly.shape)

X_train_poly
# трансформируем X_train_poly/test numpy матрицы в pandas Data Frame

X_train_poly_pd = pd.DataFrame(data=X_train_poly, 

                             index=tv_X_train.index, 

                             columns=np.arange(0, np.size(X_train_poly,1)))

X_test_poly_pd = pd.DataFrame(data=X_test_poly, 

                             index=tv_X_test.index, 

                             columns=np.arange(0, np.size(X_test_poly,1)))



# объединяем текстовые переменные со сгенерированными полиномиальными

tv_X_train = tv_X_train.join(X_train_poly_pd, rsuffix='_poly')

tv_X_test = tv_X_test.join(X_test_poly_pd, rsuffix='_poly')



# проверяем

tv_X_train.shape, tv_X_test.shape
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix, plot_confusion_matrix

from sklearn.metrics import r2_score

from sklearn.metrics import mean_absolute_error



lr = LogisticRegression(penalty='l2', max_iter=500, C=1, random_state=42)



# тренируем

lr_tfidf=lr.fit(tv_X_train, y_train)

print(lr_tfidf)



# предсказываем

lr_tfidf_predict=lr.predict(tv_X_test)



# Accuracy

lr_tfidf_score=accuracy_score(y_test,lr_tfidf_predict)

print("lr_tfidf_score :",lr_tfidf_score)



# Classification report

lr_tfidf_report=classification_report(y_test,lr_tfidf_predict,target_names=['0','1'])

print(lr_tfidf_report)



# Confusion matrix

plot_confusion_matrix(lr_tfidf, tv_X_test, y_test,display_labels=['Отрицательные','Положительные'],cmap="Blues",values_format = '')



# R^2

r2 = r2_score(y_test, lr_tfidf_predict)

print (f"R2 score / LR = {r2}")



# MAE

meanae = mean_absolute_error(y_test, lr_tfidf_predict)

print ("MAE (Mean Absolute Error) {0}".format(meanae))
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV



knn = KNeighborsClassifier()



# Зададим сетку - среди каких значений выбирать наилучший параметр.

knn_grid = {'n_neighbors': np.array(np.linspace(1, 31, 3), dtype='int')} # перебираем по параметру <<n_neighbors>>, по сетке заданной np.linspace



# Создаем объект кросс-валидации

gs = GridSearchCV(knn, knn_grid, cv=3)



# Обучаем его

gs.fit(tv_X_train, y_train)



# Функция отрисовки графиков

def grid_plot(x, y, x_label, title, y_label):

    plt.figure(figsize=(12, 6))

    plt.grid(True)

    plt.plot(x, y, 'go-')

    plt.xlabel(x_label)

    plt.ylabel(y_label)

    plt.title(title)



# Строим график зависимости качества от числа соседей

grid_plot(knn_grid['n_neighbors'], gs.cv_results_['mean_test_score'], 'n_neighbors', 'KNeighborsClassifier', 'accuracy')



# лучший параметр

print('лучший параметр: ', gs.best_params_, gs.best_score_)
# KNN с лучшим параметром

# лучший параметр варьируется от выборки к выбоке, т.к. данных меньше, чем хотелось бы, ввиду низкой производительности Kaggel kernel

clf_knn = KNeighborsClassifier(n_neighbors=31)



# предсказания

clf_knn.fit(tv_X_train, y_train)

y_knn = clf_knn.predict(tv_X_test)



# classification report

print(classification_report(y_test, y_knn))



# confusion matrix

plot_confusion_matrix(clf_knn, tv_X_test, y_test, display_labels=['Отрицательные','Положительные'], cmap="Blues", values_format = '')
from sklearn.svm import SVC



alg = SVC()



grid = {'C': np.array(np.linspace(-5, 5, 3), dtype='float'),

        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],

        }



gs = GridSearchCV(alg, grid, verbose=2, n_jobs = -1, scoring = 'f1')

gs.fit(tv_X_train, y_train)

gs.best_params_, gs.best_score_
# лучший параметр варьируется от выборки к выбоке, т.к. данных меньше, чем хотелось бы, ввиду низкой производительности Kaggel kernel

alg = SVC(C = 5.0, kernel = 'linear')

alg.fit(tv_X_train, y_train)

preds = alg.predict(tv_X_test)



# classification report

print(classification_report(y_test, preds))



# confusion matrix

plot_confusion_matrix(alg, tv_X_test, y_test, display_labels=['Отрицательные','Положительные'], cmap="Blues", values_format = '')
from sklearn.naive_bayes import BernoulliNB



alg = BernoulliNB()



grid = {'alpha': np.array(np.linspace(0, 6, 30), dtype='float'),}



gs = GridSearchCV(alg, grid, verbose=2, n_jobs = -1, scoring = 'f1')

gs.fit(tv_X_train, y_train)

gs.best_params_, gs.best_score_
# Функция отрисовки графиков

def grid_plot(x, y, x_label, title, y_label='f1'):

    plt.figure(figsize=(12, 6))

    plt.grid(True)

    plt.plot(x, y, 'go-')

    plt.xlabel(x_label)

    plt.ylabel(y_label)

    plt.title(title)
# Строим график зависимости качества от числа соседей

grid_plot(grid['alpha'], gs.cv_results_['mean_test_score'], 'n_neighbors', 'BernoulliNB')
# прогноз

preds = gs.predict(tv_X_test)
# classification report

print(classification_report(y_test, preds))



# confusion matrix

plot_confusion_matrix(gs, tv_X_test, y_test, display_labels=['Отрицательные','Положительные'], cmap="Blues", values_format = '')