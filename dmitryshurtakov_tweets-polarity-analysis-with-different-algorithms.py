# импортируем необходимые для начала библиотеки

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
# загрузим данные в соответствующей кодировке, размечая названия стобцов

DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]

DATASET_ENCODING = "ISO-8859-1"

data = pd.read_csv('../input/sentiment140/training.1600000.processed.noemoticon.csv', 

                   encoding=DATASET_ENCODING, 

                   names=DATASET_COLUMNS)

data.head()
# удалим признаки (столбцы), не влияющие на дальнейшую итоговую классификацию данных 

df = data.drop(['ids', 'date', 'flag', 'user'], axis=1)

df.head()
# заменим значение для позитивного твита с "4" на более привычное "1"

# и отобразим общее количество позитивных и негативных постов

df['target'] = df['target'].replace(4, 1)

df['target'].value_counts()
# визуализируем распределение целевой переменной

# видим, что классы сбалансированы

ax = df.groupby('target').count().plot(kind='bar', title='Target distribution', legend=False)

ax.set_xticklabels(['Negative','Positive'], rotation=0)
# сохраним значения текстового признака и целевой переменной в списки

text, target = list(df['text']), list(df['target'])
import re

import string

from nltk.stem import WordNetLemmatizer



# создадим функцию для предобработки текстового признака

def preprocess(doc):

    prepdoc = []

    

    lemmatizer = WordNetLemmatizer()

    

    urlptr = r'((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)'

    usrptr = '@[^\s]+'

    alhptr = '[^a-zA-Z0-9]'

    sqcptr = r'(.)\1\1+'

    rplptr = r'\1\1'

    

    for text in doc:

        # приводим весь текст к нижнему регистру

        text = text.lower()

        # заменяем ссылки на 'URL'

        text = re.sub(urlptr, ' URL', text)      

        # заменяем имя пользователя на 'USER'

        text = re.sub(usrptr, ' USER', text)        

        # убираем все символы, отличные от буквенных или цифровых

        text = re.sub(alhptr, ' ', text)

        # обрезаем последовательности из трёх и более одинаковых букв

        text = re.sub(sqcptr, rplptr, text)

        

        words = ' '

        for word in text.split():

            # проверяем короткие слова и приводим словоформы к лемме (словарной форме)

            if len(word) > 1:

                word = lemmatizer.lemmatize(word)

                words += (word + ' ')

            

        prepdoc.append(words)

        

    return prepdoc
# обработаем список признака 'text' и отобразим часть сообщений

%time preptext = preprocess(text)

preptext[:10]
from wordcloud import WordCloud



# визуализируем облако слов, наиболее часто появляющихся в позитивных твитах

wordpos = preptext[800000:]

wc = WordCloud(max_words=1000, width=1600, height=800, 

               collocations=False).generate(' '.join(wordpos))

plt.figure(figsize=(20, 20))

plt.imshow(wc)
# визуализируем облако слов, наиболее часто появляющихся в негативных твитах

wordneg = preptext[:800000]

wc = WordCloud(max_words=1000, width=1600, height=800, 

               collocations=False).generate(' '.join(wordneg))

plt.figure(figsize=(20, 20))

plt.imshow(wc)
from sklearn.model_selection import train_test_split



# разбиваем данные на обучающие и испытательные наборы

X_train, X_test, y_train, y_test = train_test_split(preptext, target, test_size=0.1, random_state=0)
from sklearn.feature_extraction.text import TfidfVectorizer



# инициализируем класс TfidfVectorizer, преобразующий текст в матрицу tfidf

# с использованием ngram и ограничением максимального количества признаков

vectorizer = TfidfVectorizer(ngram_range=(1, 2), 

                             max_features=500000)

%time vectorizer.fit(preptext)
# создаём вектора признаков

X_train = vectorizer.transform(X_train)

X_test  = vectorizer.transform(X_test)



# смотрим на слова в отдельном твите, преобразовав получившийся вектор

vectorizer.inverse_transform(X_train[66])[0][np.argsort(X_train[66].data)]
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix 



# создадим функцию для обучения модели и визуализации результатов

def evaluate(model, matrix):

    

    # предсказываем результаты на тестовой выборке

    y_pred = model.predict(X_test)



    # выводим метрики

    print(classification_report(y_test, y_pred))

    

    # строим сonfusion matrix

    plot_confusion_matrix(matrix, X_test, y_test, 

                          display_labels=['Negative', 'Positive'], 

                          cmap='Blues', values_format=' ')   
from sklearn.svm import LinearSVC



svc = LinearSVC()

svc_mtx = svc.fit(X_train, y_train)

%time evaluate(svc, svc_mtx)
from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression(max_iter=1000, n_jobs=-1)

logreg_mtx = logreg.fit(X_train, y_train)

%time evaluate(logreg, logreg_mtx)
from sklearn.naive_bayes import BernoulliNB



bnb = BernoulliNB()

bnb_mtx = bnb.fit(X_train, y_train)

%time evaluate(bnb, bnb_mtx)
from sklearn.naive_bayes import ComplementNB



cnb = ComplementNB()

cnb_mtx = cnb.fit(X_train, y_train)

%time evaluate(cnb, cnb_mtx)
import pickle



# сохраняем параметры матрицы tfidf

file = open('vectorizer_ngram(1,2).pickle','wb')

pickle.dump(vectorizer, file)

file.close()



# сохраняем модель логистической регресии

file = open('LogReg_model.pickle','wb')

pickle.dump(logreg, file)

file.close()