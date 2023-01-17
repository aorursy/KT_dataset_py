import numpy as np

import pandas as pd

from matplotlib import pyplot as plt
df = pd.read_csv("../input/dataisbeautiful/r_dataisbeautiful_posts.csv")
df.head()
df.info()
df.iloc[:,5].value_counts()
df.iloc[:,7].value_counts()
df['id'].nunique()
df = pd.read_csv("../input/dataisbeautiful/r_dataisbeautiful_posts.csv", 

                 usecols=[1,2,3,4,5,6,8,9,10,11],

                 dtype={5:object})
df.head()
df.info()
df.over_18.replace(True,1,inplace = True)

df.over_18.replace(False,0,inplace = True)

df.over_18.value_counts()
df.isna().sum()
df.total_awards_received.fillna(0, inplace = True)



# проверим

df.isna().sum()
df[df['title'].isna()]
pd.set_option('display.max_colwidth', None)

df[df['title'].isna()]['full_link']
# заполняем пустой title пустой строкой, что бы была корректная обработка в WordCloud

df.title.fillna(" ",inplace = True)
print('author_flair_text is NaN: \n', df[df['author_flair_text'].isna()]['full_link'].head(), "\n")

print('removed_by is NaN: \n', df[df['removed_by'].isna()]['full_link'].head(), "\n")

print('author_flair_text is NaN: \n', df[df['total_awards_received'].isna()]['full_link'].head(), "\n")

print('author_flair_text is NaN and score >10 000: \n', df[df['total_awards_received'].isna() & (df['score'] > 10000)]['full_link'].head())
df = df.drop(['full_link'], axis = 1)

df = df.drop(['created_utc'], axis = 1)

df.head()
df['text'] = df['title'] + ' ' + df['author']
df['author_encoded'] = df['author']



# перекодируем NaN в строку "not available" (object type) (иначе кодировщик отказывается работать с NaN)

df.author_flair_text.fillna("not available",inplace = True)

df.removed_by.fillna("not available",inplace = True)

df_to_encode = df[['author_encoded', 'author_flair_text', 'removed_by']]

df_to_encode.head()
# Подключаем класс для предобработки данных

from sklearn import preprocessing



# Напишем функцию, которая принимает на вход DataFrame, кодирует числовыми значениями категориальные признаки

# и возвращает обновленный DataFrame и сами кодировщики.

def number_encode_features(init_df):

    result = init_df.copy() # копируем нашу исходную таблицу

    encoders = {}

    for column in result.columns:

        if result.dtypes[column] == np.object: # np.object -- строковый тип / если тип столбца - строка, то нужно его закодировать

            encoders[column] = preprocessing.LabelEncoder() # для колонки column создаем кодировщик

            result[column] = encoders[column].fit_transform(result[column]) # применяем кодировщик к столбцу и перезаписываем столбец

    return result, encoders



df_encoded, encoders = number_encode_features(df_to_encode) # Теперь encoded data содержит закодированные кат. признаки 

encoders
df_encoded.head()
df = df.drop(['author_encoded', 'author_flair_text', 'removed_by'], axis = 1).join(df_encoded)

df.head()
# описание данных

print(df.groupby('over_18')['over_18'].count())



# процент over_18

over_18_count = df[df['over_18'] == 1]['over_18'].count()

total = df['over_18'].count()

over_18_share = over_18_count/total

print("Доля данных, показывающих целевую группу \"over_18\" {0:.4f}".format(over_18_share))



# гистограмма

df[df['over_18'] == 0]['over_18'].astype(int).hist(label='False', grid = False, bins=1, rwidth=0.8)

df[df['over_18'] == 1]['over_18'].astype(int).hist(label='True', grid = False, bins=1, rwidth=0.8)

plt.xticks((0,1),('False', 'True'))

plt.show()
df.hist(figsize=(15,12), bins=20)
import seaborn as sns

plt.figure(figsize=(8, 6))

spearman = df.corr(method = 'spearman')

sns.heatmap(spearman, annot = True)
# np.argwhere вернет индексы тех элементов массива y (целевой переменной), где значение 0

not_over_18 = df['over_18'].astype(int).to_numpy()

not_over_18_ids = np.argwhere(not_over_18 == 0).flatten()

print('Всего over_18 = False: ', len(not_over_18_ids))

not_over_18_ids
# Перемешаем массив с выбранным random state (чтоб в дальнейшем у нас совпадали выборки) выберем в нем "лишние" id тех, кто not over_18 (кто портит нам прогноз алгоритма). 

# Кол-во "лишних" = кол-во оставшихся - кол-во over_18.

from sklearn.utils import shuffle



not_over_18_ids = shuffle(not_over_18_ids, random_state = 42)

# найдем "лишних", для этого обрежем найденные id на кол-во over_18 (внутри len)

not_over_18_ids = not_over_18_ids[len(np.argwhere(not_over_18 == 1).flatten()):]

print(len(not_over_18_ids))

# отображаем кол-во и сами id, которые мы должны выкинуть

not_over_18_ids
# Проверим, сбалансированны ли классы

# по идее (оставшиеся) - ("лишние") = (over_18)

len(np.argwhere(not_over_18 == 0).flatten()) - len(not_over_18_ids) == len(np.argwhere(not_over_18 == 1).flatten())
# Разделение выборки на X и y



X = df.drop(['over_18'],axis = 1)

y = df['over_18']



X.shape, y.shape
# выбрасываем лишних по индексам

X = X.drop(X.index[not_over_18_ids])

y = y.drop(y.index[not_over_18_ids])



# отобразим итоговый размер признаков датасета

X.shape, y.shape
# Теперь видим, что классы сбалансированы.

y.value_counts()
from sklearn.preprocessing import OneHotEncoder



# Категориальные признаки

cat_cols = ['author_encoded', 'author_flair_text', 'removed_by']



# Кодируем категориальные признаки

ohe_df = pd.DataFrame(index=X.index)

ohe = OneHotEncoder(handle_unknown='ignore')



for col in cat_cols:

    ohe.fit(X[[col]])

    ohe_result = pd.DataFrame(ohe.transform(X[[col]]).toarray(),

                              columns=ohe.get_feature_names(input_features=[col]),

                              index=X.index)

    ohe_df = ohe_df.join(ohe_result)



ohe_df.head()
from sklearn.preprocessing import StandardScaler



num_cols = ['score', 'total_awards_received', 'num_comments']

std_df = pd.DataFrame(index=X.index)

scaler = StandardScaler()



for col in num_cols:

    scaler.fit(X[[col]])

    std_result = pd.DataFrame(scaler.transform(X[[col]]),

                              columns=[col],

                              index=X.index)

    std_df = std_df.join(std_result)



std_df.head()
cols_to_drop = ['author_encoded', 'author_flair_text', 'removed_by', 'score', 'total_awards_received', 'num_comments']

X_prepared = X.drop(cols_to_drop, axis=1).join(std_df).join(ohe_df)

X = X_prepared

X.head()
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,random_state = 42)
# title для X train, у которых over_18 = true

train_true_title = X_train.loc[y_train[y_train == 1].index,:]['title']



# author для X train, у которых over_18 = true

train_true_author = X_train.loc[y_train[y_train == 1].index,:]['title']



# text для X train, у которых over_18 = true

train_true_text = X_train.loc[y_train[y_train == 1].index,:]['text']
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
wordcloud_img(train_true_title)
wordcloud_img(train_true_author)
def top10words(data):

    wc = WordCloud(min_font_size = 3,  

                   max_words = 3000 , 

                   width = 1000 , 

                   height = 600 , 

                   stopwords = STOPWORDS).generate(str(" ".join(data)))

    text_true = wc.process_text(str(" ".join(data)))

    print('10 слов \n', list(text_true.keys())[:10])

    print('Общее количество слов: ', len(text_true.keys()), '\n')



# поле title

print(top10words(train_true_title))

# поле author

print(top10words(train_true_author))
wc = WordCloud(min_font_size = 3,  

                   max_words = 3000, 

                   width = 1600, 

                   height = 800, 

                   stopwords = STOPWORDS).generate(str(" ".join(train_true_text)))

train_dictionary = wc.process_text(str(" ".join(train_true_text)))

# количество слов в словаре

n_of_words = len(train_dictionary.keys())



print('10 слов \n', list(train_dictionary.keys())[:10])

print('Общее количество слов: ', n_of_words)
# Сортируем словарь по частоте употребления

train_dictionary_sorted = sorted(train_dictionary.items(), key = lambda word:(word[1], word[0]))

# Топ 20% наиболее употребляемых слов в сообщениях over_18

train_top_words = train_dictionary_sorted[round(n_of_words*0.8):]



# слова-исключения

exclusion_words =['data', 'year', 'month', 'week', 'day', 'post', 'every', 'average', 'word', 'the world', 

                  'years of', 'new', 'graph', 'US', 'time', 'result', 'by state', 'UK', 'tags', 'countries', 

                  "I've", 'comment', 'last year', 'Chart', 'Countries', 'State', 'Map of', 'We', 'analysis of']



# сортированный словарь топ 20% наиболее употребляемых слов в категории over_18

ans_true = []

for i in train_top_words:

    # исключаем некоторые слова из словаря

    if i[0] in exclusion_words: continue

    ans_true.append(i[0])



# Топ 20% наиболее употребляемых слов в категории over_18

ans_true[len(ans_true)-20:]
predictions = []

for i in X_test['text']:

    x = i.split()

    for j in x:

        if j in ans_true:

            predictions.append(1)

            break

        else:

            predictions.append(0)

            break

len(predictions)
len(y_test)
count = 0

for i in range(len(predictions)):

    y_test = list(y_test)

    if(predictions[i] == int(y_test[i])):

        count += 1

print(count)
accuracy = (count/len(predictions))*100

print('Accuracy с использованием Word Cloud и Исключениями равна', np.round(accuracy, 2))
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
X_train['text_lemma'] = X_train['text'].apply(lemmatize_words).apply(join_text)

X_test['text_lemma'] = X_test['text'].apply(lemmatize_words).apply(join_text)
X_train[['text', 'text_lemma']].head()
from sklearn.feature_extraction.text import TfidfVectorizer



# Присвоение весов словам с использованием TfidfVectorizer

tv = TfidfVectorizer(min_df=0,max_df=1,use_idf=True,ngram_range=(1,2))

tv_X_train = tv.fit_transform(X_train['text_lemma'])

tv_X_test = tv.transform(X_test['text_lemma'])



print('TfidfVectorizer_train:', tv_X_train.shape)

print('TfidfVectorizer_test:', tv_X_test.shape)

# сбросим поля,которые мы больше не будем использовать (мы преобразовали их в другие формы)

cols_to_drop = ['title', 'author', 'text','text_lemma']



X_train_dropped = X_train.drop(cols_to_drop, axis=1)

X_test_dropped = X_test.drop(cols_to_drop, axis=1)
# Нормализуем данные в tv_X_train/test

scaler_tv = StandardScaler()

tv_X_train_scaled = scaler_tv.fit_transform(tv_X_train.toarray())

tv_X_test_scaled = scaler_tv.transform(tv_X_test.toarray())



# трансформируем tv_X_train/test sparse numpy матрицы в pandas Data Frame

tv_X_train_pd_scaled = pd.DataFrame(data=tv_X_train_scaled, 

                             index=X_train.index, 

                             columns=np.arange(0, np.size(tv_X_train_scaled,1)))

tv_X_test_pd_scaled = pd.DataFrame(data=tv_X_test_scaled, 

                             index=X_test.index, 

                             columns=np.arange(0, np.size(tv_X_test_scaled,1)))

# проверяем

tv_X_train_pd_scaled
# объединение ранее обработанных переменных с текстовыми переменными

X_train = X_train_dropped.join(tv_X_train_pd_scaled)

X_test = X_test_dropped.join(tv_X_test_pd_scaled)



# проверяем

X_train
from sklearn.linear_model import LogisticRegression



lr = LogisticRegression(penalty='l2', max_iter=500, C=1, random_state=42)



# для Tf-idf

lr_tfidf=lr.fit(tv_X_train, y_train)

print(lr_tfidf)
# для Tf-idf

lr_tfidf_predict=lr.predict(tv_X_test)
from sklearn.metrics import accuracy_score



# для Tf-idf

lr_tfidf_score=accuracy_score(y_test,lr_tfidf_predict)

print("lr_tfidf_score :",lr_tfidf_score)
from sklearn.metrics import classification_report



# для Tf-idf

lr_tfidf_report=classification_report(y_test,lr_tfidf_predict,target_names=['0','1'])

print(lr_tfidf_report)
from sklearn.metrics import confusion_matrix, plot_confusion_matrix



# для Tf-idf

plot_confusion_matrix(lr_tfidf, tv_X_test, y_test,display_labels=['0','1'],cmap="Blues",values_format = '')
from sklearn.metrics import r2_score

r2 = r2_score(y_test, lr_tfidf_predict)

print (f"R2 score / LR = {r2}")
from sklearn.metrics import mean_absolute_error

meanae = mean_absolute_error(y_test, lr_tfidf_predict)

print ("MAE (Mean Absolute Error) {0}".format(meanae))
from sklearn.linear_model import LogisticRegression



lr = LogisticRegression(penalty='l2', max_iter=500, C=1, random_state=42)



lr_all=lr.fit(X_train, y_train)

print(lr_all)
lr_all_predict=lr.predict(X_test)
from sklearn.metrics import accuracy_score



lr_all_score=accuracy_score(y_test,lr_all_predict)

print("lr_all_score :",lr_all_score)
from sklearn.metrics import classification_report



lr_all_report=classification_report(y_test,lr_all_predict,target_names=['0','1'])

print(lr_all_report)
from sklearn.metrics import confusion_matrix, plot_confusion_matrix



plot_confusion_matrix(lr_all, X_test, y_test, display_labels=['0','1'], cmap="Blues", values_format = '')
from sklearn.metrics import r2_score



r2 = r2_score(y_test, lr_all_predict)

print (f"R2 score для всех переменных / LR = {r2}")
from sklearn.metrics import mean_absolute_error

meanae = mean_absolute_error(y_test, lr_all_predict)

print ("MAE (Mean Absolute Error) для всех переменных {0}".format(meanae))
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV



knn = KNeighborsClassifier()



# Зададим сетку - среди каких значений выбирать наилучший параметр.

knn_grid = {'n_neighbors': np.array(np.linspace(1, 30, 3), dtype='int')} # перебираем по параметру <<n_neighbors>>, по сетке заданной np.linspace



# Создаем объект кросс-валидации

gs = GridSearchCV(knn, knn_grid, cv=3)



# Обучаем его

gs.fit(X_train, y_train)
# Функция отрисовки графиков

def grid_plot(x, y, x_label, title, y_label):

    plt.figure(figsize=(12, 6))

    plt.grid(True)

    plt.plot(x, y, 'go-')

    plt.xlabel(x_label)

    plt.ylabel(y_label)

    plt.title(title)
grid_plot(knn_grid['n_neighbors'], gs.cv_results_['mean_test_score'], 'n_neighbors', 'KNeighborsClassifier', 'accuracy')
gs.best_params_, gs.best_score_
clf_knn = KNeighborsClassifier(n_neighbors=15)
# предсказания

clf_knn.fit(X_train, y_train)

y_knn = clf_knn.predict(X_test)
print(classification_report(y_test, y_knn))
plot_confusion_matrix(lr_all, X_test, y_test, display_labels=['0','1'], cmap="Blues", values_format = '')