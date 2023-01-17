import numpy as np

import pandas as pd

from matplotlib import pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
df.head()
df.info()
# количество уникальных ключевых слов

df.loc[:,'keyword'].nunique()
# частота использования keywords

df.loc[:,'keyword'].value_counts()
# количество уникальных локаций

df.loc[:,'location'].nunique()
# частота использования различных location

df.loc[:,'location'].value_counts()
# описание данных

print(df.groupby('target')['target'].count())



# процент target=1

target_count = df[df['target'] == 1]['target'].count()

total = df['target'].count()

target_share = target_count/total

print("Доля данных, показывающих реалные происшествия \"target=1\" {0:.2f}".format(target_share))



# гистограмма

df[df['target'] == 0]['target'].astype(int).hist(label='Фейк', grid = False, bins=1, rwidth=0.8)

df[df['target'] == 1]['target'].astype(int).hist(label='Реальные', grid = False, bins=1, rwidth=0.8)

plt.xticks((0,1),('Фейк', 'Реальные'))

plt.show()
# т.к. в признаке location отсутствуют координаты, необходимые для построения карты, то добавим новую базу world-cities-database

# в этой базе каждой стране или городу проставлены координаты latitude / longitude



latlong = pd.read_csv("/kaggle/input/world-cities-database/worldcitiespop.csv")

latlong.head()
# перемименуем колонку AccentCity в Location для объединения с нашей исходной базой

latlong.rename(columns={"AccentCity": "location"}, inplace=True)

latlong.head()
# Получим уникальные сочетания location и latitude / longitude для добавления в исходную базу, удалим дублированные строки и отсутствующие location

latlong_grouped = latlong[['location', 'Latitude', 'Longitude', 'Population']].drop_duplicates()

latlong_grouped = latlong_grouped[latlong_grouped.location != 'None']
# отсортируем базу сначала по алфавиту по location (по возрастанию) и затем по population (по убыванию)

latlong_grouped.sort_values(['location', 'Population'], ascending=[True, False], inplace=True)
latlong_grouped.shape
# удалим повторяющиеся города из базы, оставим только те города из дублей, для которых население максимально

# для этого сравним текущую строку и следующую (shift смещает на одну строку вперёд), если они повторяются, то мы их не включаем в базу

latlong_cleaned = latlong_grouped[latlong_grouped['location'] != latlong_grouped['location'].shift()]

# проверим размерность базы

latlong_cleaned.shape
# для базы скоординатами, в которой дубли не удалялись

latlong_grouped[latlong_grouped['location'] == 'Birmingham']
# для базы с координатами, в которой дубли локаций были очищены

latlong_cleaned[latlong_cleaned['location'] == 'Birmingham']
# Добавим координаты к нашей исходной базе твитов

df_latlong = pd.merge(df, latlong_cleaned, left_on='location', right_on='location', how='left')

#df_latlong = df.set_index('location').join(latlong.set_index('location'), how='left', on=['location'])

df_latlong.head()
df_latlong.shape
# Отобразим локации, для которых удалось определить координаты

df_latlong[~df_latlong['Latitude'].isna()]
# количество реальных твитов с координатами

df_latlong_real = df_latlong[(~df_latlong['Latitude'].isna()) & (df_latlong['target'] == 1)]

print( len( df_latlong_real ) )
# пример реальных твитов с координатами

df_latlong_real.head()
# Нанесём реальные твиты на карту

import folium

from folium.plugins import HeatMap



df_latlong_real.Latitude.fillna(0, inplace = True)

df_latlong_real.Longitude.fillna(0, inplace = True) 

twits = df_latlong_real[['Latitude', 'Longitude']]



RealTwitsMap=folium.Map(location=[0,0],zoom_start=2)

HeatMap(data=twits, radius=12).add_to(RealTwitsMap)

RealTwitsMap
# количество фейковых твитов с координатами

df_latlong_fake = df_latlong[(~df_latlong['Latitude'].isna()) & (df_latlong['target'] == 0)]

print( len( df_latlong_fake ) )
# пример фейковых твитов с координатами

df_latlong_fake.head()
# Нанесём фейковые твиты на карту

import folium

from folium.plugins import HeatMap



df_latlong_fake.Latitude.fillna(0, inplace = True)

df_latlong_fake.Longitude.fillna(0, inplace = True) 

twits = df_latlong_fake[['Latitude', 'Longitude']]



FakeTwitsMap=folium.Map(location=[0,0],zoom_start=2)

HeatMap(data=twits, radius=12).add_to(FakeTwitsMap)

FakeTwitsMap
# функция для отрисовки облака часто употребляемых слов

from wordcloud import WordCloud,STOPWORDS



def wordcloud_img(data):

    plt.figure(figsize = (20,20))

    wc = WordCloud(min_font_size = 3, 

                   background_color="white",  

                   max_words = 3000, 

                   width = 1000, 

                   height = 600, 

                   stopwords = STOPWORDS).generate(str(" ".join(data)))

    plt.imshow(wc,interpolation = 'bilinear')
wordcloud_img(df[df['target'] == 1]['text'])
wordcloud_img(df[df['target'] == 0]['text'])
from urllib.parse import unquote
# выведем все слова с urlencoded-символами (у них всегда стоит знак "%")

for phrase in df['keyword']:

    phrase  = str(phrase)

    if('%' in phrase):

        print(phrase)
# делаем замену

for i, phrase in enumerate(df['keyword']):

    phrase  = str(phrase)

    if('%20' in phrase):

        df.loc[i, 'keyword'] = df.loc[i, 'keyword'].replace('%20', ' ')



# проверим

# выведем все слова с urlencoded-символами (у них всегда стоит знак "%")

for phrase in df['keyword']:

    phrase  = str(phrase)

    if('%' in phrase):

        print(phrase)
# проверяем пропуски в cтолбцах

df.isna().sum()
def location_and_keyword_fill_nan(df):

    #пропуски для keywords заменим на None

    df.keyword.fillna('None', inplace = True)



    #пропуски для location заменим на None

    df.location.fillna('None', inplace = True)



location_and_keyword_fill_nan(df)



# проверим

df.isna().sum()
# вернём индексы тех элементов массива target (целевой переменной), где значение 0

target_np = df['target'].astype(int).to_numpy()

fake_twits_ids = np.argwhere(target_np == 0).flatten()

print('Всего fake_twits: ', len(fake_twits_ids))

fake_twits_ids
# перемешаем массив с id фейковых твитов

from sklearn.utils import shuffle

fake_twits_ids = shuffle(fake_twits_ids, random_state = 42)



# выберем в нем "лишние" id фейковых твитов

# т.е. возьмём все элементы после номера 3271 из перемешанного массива fake_twits_ids

fake_twits_ids_to_drop = fake_twits_ids[len(np.argwhere(target_np == 1).flatten()):]



# отображаем кол-во фейковых твитов, которые нужно выбросить из выборки для балансировки, а также их id

print(len(fake_twits_ids_to_drop))

fake_twits_ids_to_drop
# т.к. данные пермешаны, то после отбрасыания лишних элементов, выборка будет репрезентативной

df_balanced = df.drop(df.index[fake_twits_ids_to_drop])



# отобразим итоговый размер признаков датасета

df_balanced.shape
# теперь видим, что классы сбалансированы.

df_balanced['target'].value_counts()
# гистограмма

df_balanced[df_balanced['target'] == 0]['target'].astype(int).hist(label='Фейк', grid = False, bins=1, rwidth=0.8)

df_balanced[df_balanced['target'] == 1]['target'].astype(int).hist(label='Реальные', grid = False, bins=1, rwidth=0.8)

plt.xticks((0,1),('Фейк', 'Реальные'))

plt.show()
def location_and_keyword_to_bool_category(df):

    df.loc[df['location'] == 'None', 'location_bool'] = 0

    df.loc[df['location'] != 'None', 'location_bool'] = 1

    df.loc[df['keyword'] == 'None', 'keyword_bool'] = 0

    df.loc[df['keyword'] != 'None', 'keyword_bool'] = 1



location_and_keyword_to_bool_category(df_balanced)



# проверяем

df_balanced
# делаем подобные преобразования до разбивки на train и test,т.к. такие преобразование каждого текста происходят независимо отвыборки

# к тому же X_train и X_test, которые образуются после train_test_split являются копиями базы и на копиях такие преобразования делать сложнее (может возвращаться ошибкао том, что работы ведутся на копии и т.п.)

# regular expressions library

import re



# функция для преобразования единичного текста

def single_text_clean(text):

    # преобразуем текст в нижний регистр

    text = text.lower()

    # преобразуем https:// и т.п. адреса в текст "URL"

    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',text)

    # преобразуем имя пользователя @username в "AT_USER"

    text = re.sub('@[^\s]+','AT_USER', text)

    # преобразуем множественые пробелы в один пробел

    text = re.sub('[\s]+', ' ', text)

    # преобразуем хэштег #тема в "тема"

    text = re.sub(r'#([^\s]+)', r'\1', text)

    return text



# функция для преобразования текста в разных столбцах датафрейма 

# (применяет предыдущую функцию single_text_clean к разным столбцам)

def text_columns_clean(df):

    text_columns_to_clean = ['keyword', 'location', 'text']

    for column in text_columns_to_clean:

        df.loc[:, column] = df[column].apply(single_text_clean)



text_columns_clean(df_balanced)



# проверим

df_balanced
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

df_balanced.loc[:, 'text'] = df_balanced['text'].apply(lemmatize_words).apply(join_text)



# проверим

df_balanced
# удалим не нужное нам больше дополнительное поле id (оно есть в индексе)

df_balanced = df_balanced.drop(['id'], axis=1)



from sklearn.model_selection import train_test_split



# разделение выборки на X и y

y = df_balanced['target']

X = df_balanced.drop(['target'], axis=1)



# разделение выборки на train и test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)



# проверяем

X_train.shape
X_train
from sklearn.feature_extraction.text import TfidfVectorizer



# Присвоение весов словам с использованием TfidfVectorizer

tv = TfidfVectorizer(min_df=0,max_df=1,use_idf=True,ngram_range=(1,2))

tv_X_train = tv.fit_transform(X_train['text'])

tv_X_test = tv.transform(X_test['text'])



print('TfidfVectorizer_train:', tv_X_train.shape)

print('TfidfVectorizer_test:', tv_X_test.shape)
# перевернём столбцы со словами из sparse матрицы с одним столбцом в numpy массив со многими столбцами

tv_X_train = tv_X_train.toarray()

tv_X_test = tv_X_test.toarray()
# Нормализуем данные в tv_X_train/test

from sklearn.preprocessing import StandardScaler



def scaling_train(tv_X_train):

    scaler_tv = StandardScaler(copy=False)

    tv_X_train_scaled = scaler_tv.fit_transform(tv_X_train)

    # трансформируем tv_X_train/test sparse numpy матрицы в pandas Data Frame

    tv_X_train_pd_scaled = pd.DataFrame(data=tv_X_train_scaled, 

                             index=X_train.index, 

                             columns=np.arange(0, np.size(tv_X_train_scaled,1)))

    return tv_X_train_pd_scaled, scaler_tv



def scaling_test(tv_X_test, scaler_tv):

    tv_X_test_scaled = scaler_tv.transform(tv_X_test)

    # трансформируем tv_X_train/test sparse numpy матрицы в pandas Data Frame

    tv_X_test_pd_scaled = pd.DataFrame(data=tv_X_test_scaled, 

                             index=X_test.index, 

                             columns=np.arange(0, np.size(tv_X_test_scaled,1)))

    return tv_X_test_pd_scaled



tv_X_train_pd_scaled, scaler_tv = scaling_train(tv_X_train)

tv_X_test_pd_scaled = scaling_test(tv_X_test, scaler_tv)



# проверяем

tv_X_train_pd_scaled.shape, tv_X_test_pd_scaled.shape
tv_X_train = tv_X_train_pd_scaled

tv_X_test = tv_X_test_pd_scaled
X_train_to_join = X_train[['location_bool', 'keyword_bool']]

X_test_to_join = X_test[['location_bool', 'keyword_bool']]

tv_X_train_joined = tv_X_train.join(X_train_to_join)

tv_X_test_joined = tv_X_test.join(X_test_to_join)



# проверяем

tv_X_train_joined.shape, tv_X_test_joined.shape
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

plot_confusion_matrix(lr_tfidf, tv_X_test, y_test,display_labels=['Фейковые','Реальные'], cmap="Blues", values_format = '')



# R^2

r2 = r2_score(y_test, lr_tfidf_predict)

print (f"R2 score / LR = {r2}")



# MAE

meanae = mean_absolute_error(y_test, lr_tfidf_predict)

print ("MAE (Mean Absolute Error) {0}".format(meanae))
from sklearn.naive_bayes import BernoulliNB

from sklearn.model_selection import GridSearchCV



alg = BernoulliNB()



grid = {'alpha': np.array(np.linspace(0, 10, 10), dtype='float'),}



gs = GridSearchCV(alg, grid, verbose=2, n_jobs = -1, scoring = 'f1')

gs.fit(tv_X_train, y_train)

gs.best_params_, gs.best_score_
# Функция отрисовки графиков

def grid_plot(x, y, x_label, title, y_label='f1'):

    # определили размер графика

    plt.figure(figsize=(12, 6))

    # добавили сетку на фон

    plt.grid(True)

    # построили по х - число соседей, по y - точность

    plt.plot(x, y, 'go-')

    # добавили подписи осей и название графика

    plt.xlabel(x_label)

    plt.ylabel(y_label)

    plt.title(title)
# Строим график зависимости качества от числа соседей

# замечание: результаты обучения хранятся в атрибуте cv_results_ объекта gs

grid_plot(grid['alpha'], gs.cv_results_['mean_test_score'], 'n_neighbors', 'BernoulliNB')
# прогноз

preds = gs.predict(tv_X_test)
# classification report

print(classification_report(y_test, preds))



# confusion matrix

plot_confusion_matrix(gs, tv_X_test, y_test, display_labels=['Фейковые','Реальные'], cmap="Blues", values_format = '')