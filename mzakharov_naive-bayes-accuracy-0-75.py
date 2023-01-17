import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('../input/dataisbeautiful/r_dataisbeautiful_posts.csv')
df.head()
df.shape
# проверим пропущенные значения
fig, ax = plt.subplots(figsize=(14,10))
sns.heatmap(df.isnull(), cbar=False, cmap="YlGnBu_r")
plt.show()
df.columns
# удалим колонки с пропущенными значениями
#df.drop(columns=['id', 'author_flair_text', 'removed_by', 'created_utc', 'full_link'], inplace=True)
df.drop(columns=['id', 'author_flair_text', 'removed_by', 'total_awards_received', 'awarders', 'created_utc', 'full_link'], inplace=True)
df.head()
# проверим пропущенные значения еще раз
fig, ax = plt.subplots(figsize=(14,10))
sns.heatmap(df.isnull(), cbar=False, cmap="YlGnBu_r")
plt.show()
len(df)
df.info()
df.dropna(inplace = True)
len(df)
df.describe()
# Подключаем класс для предобработки категориальных признаках
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

# кодируем все, что можно кроме колонки "title"
encoded_data, encoders = number_encode_features(df.drop(columns='title')) # Теперь encoded data содержит закодированные кат. признаки 
encoded_data.head()
# посмотрим на распределение признаков
non_obj_cols = []
for column in encoded_data.columns:
        if df.dtypes[column] != np.bool:
            non_obj_cols.append(column)

            
fig = plt.figure(figsize=(16,8))
cols = 3

rows = np.ceil(float(encoded_data[non_obj_cols].shape[1]) / cols)
for i, column in enumerate(encoded_data[non_obj_cols].columns):
    ax = fig.add_subplot(rows, cols, i + 1)
    ax.set_title(column)
    encoded_data[non_obj_cols][column].hist(axes=ax)
    plt.xticks(rotation="vertical")
plt.subplots_adjust(hspace=0.7, wspace=0.2)
# проверим ко
plt.subplots(figsize=(12, 10))
sns.heatmap(encoded_data.corr(), square = True, annot=True)
plt.show()
print(df.groupby(['over_18'])['over_18'].count())
print(f"Доля значений \'over_18\'==True : {round(len(df[df.over_18==True])/len(df), 4)}")
labels = (df['over_18'].unique())
y_pos = np.arange(len(labels))
amount = df.groupby(['over_18'])['over_18'].count().tolist()
plt.bar(y_pos, amount)[1].set_color('orange')
plt.xticks(y_pos, labels)

plt.show()
X = np.array(encoded_data.drop(['over_18'], axis=1))
X
y = np.array(encoded_data['over_18'].astype(int))
y
# from sklearn.preprocessing import scale
# X_scaled = scale(np.array(X, dtype='float'), with_std=True, with_mean=True)
# X_scaled
# np.argwhere вернет индексы тех элементов массива y (целевой переменной), где значение 0
not_over_18_ids = np.argwhere(y == 0).flatten()
print('Всего не 18+', len(not_over_18_ids))
not_over_18_ids
from sklearn.utils import shuffle

not_over_18_ids = shuffle(not_over_18_ids, random_state = 42)
# найдем "лишних", для этого обрежем найденные id на кол-во over_18 (внутри len)
not_over_18_ids = not_over_18_ids[len(np.argwhere(y == 1).flatten()):]
print(len(not_over_18_ids))
# отображаем кол-во и сами id, которые мы должны выкинуть
not_over_18_ids
# 182948(всего нулей) - 182005(нулей после обрезки на количество единиц) = 943(осталось нулей, должно быть равно кол-ву единиц)
len(np.argwhere(y == 0).flatten()) - len(not_over_18_ids) == len(np.argwhere(y == 1).flatten())
# из X и y выкинем избыточные нули (в количестве 182005)
# np.delete принимает массив, индексы, которые выбросить и по какой оси выкидывать
X = np.delete(X, not_over_18_ids, 0)
#X_scaled = np.delete(X_scaled, not_over_18_ids, 0)
y = np.delete(y, not_over_18_ids, 0)
X.shape, y.shape
pd.Series(y).value_counts()
# Нормализуем набор весь набор для кросс-валидации
from sklearn.preprocessing import scale
X_scaled = scale(np.array(X, dtype='float'), with_std=True, with_mean=True)
X_scaled
#a = df.loc[df.index.difference(df.iloc[not_over_18_ids].index)]
from sklearn.model_selection import train_test_split

# разбиваем отбалансированные, но ненормализованные
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean)/std
X_test = (X_test - mean)/std
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDRegressor
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import GridSearchCV
from sklearn import metrics

import warnings
warnings.filterwarnings("ignore")
knn = KNeighborsClassifier()
grid = {'n_neighbors': np.array(np.linspace(30, 50, 20), dtype='int')}
gs = GridSearchCV(knn, grid)
gs.fit(X_scaled, y)
gs.best_params_, gs.best_score_
knn = KNeighborsClassifier(n_neighbors = gs.best_params_['n_neighbors'], n_jobs=-1)

knn.fit(X_train, y_train)
preds = knn.predict(X_test)
knn_res = metrics.classification_report(y_test, preds)
print(knn_res)
alg = SVC()

grid = {'C': np.array(np.linspace(0.1, 5, 10), dtype='float'),
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        }

gs = GridSearchCV(alg, grid, verbose=2, n_jobs = -1)
gs.fit(X_scaled, y)
gs.best_params_, gs.best_score_
svm = SVC(C=gs.best_params_['C'], kernel = gs.best_params_['kernel'])

svm.fit(X_train, y_train)
preds = svm.predict(X_test)
svm_res = metrics.classification_report(y_test, preds)
print(svm_res)
alg = LogisticRegression()

grid = {'penalty': ['l1', 'l2'],
        'C': np.array(np.logspace(-3, 2, num = 10), dtype='float'),
        }

gs = GridSearchCV(alg, grid, verbose=2, n_jobs = -1)
gs.fit(X_scaled, y)
gs.best_params_, gs.best_score_
logreg = LogisticRegression(penalty=gs.best_params_['penalty'], C = gs.best_params_['C'])

logreg.fit(X_train, y_train)
preds = logreg.predict(X_test)
logreg_res = metrics.classification_report(y_test, preds)
print(logreg_res)
sgd = SGDRegressor()

grid = {'penalty': ['l1', 'l2'],
        'alpha': [1e-4, 1e-5, 1e-6, 1e-7]}

gs = GridSearchCV(sgd, grid, verbose = 2, scoring = 'r2')
gs.fit(X_scaled, y)
gs.best_params_, gs.best_score_
sgd = SGDRegressor(alpha = gs.best_params_['alpha'], penalty = gs.best_params_['penalty'])
sgd.fit(X_train, y_train)
preds = sgd.predict(X_test)
sgd_res = metrics.r2_score(y_test, preds)
print('R2 sgd (sklearn): ', sgd_res)
plt.hist(y_test - preds)
gbr = GradientBoostingRegressor()

grid = {'max_depth': [3, 4, 5],
        'min_samples_split': [2, 3, 4, 5]}

gs = GridSearchCV(gbr, grid, verbose = 2)
gs.fit(X_scaled, y)
gs.best_params_, gs.best_score_
gbr = GradientBoostingRegressor(max_depth = gs.best_params_['max_depth'], min_samples_split = gs.best_params_['min_samples_split'])
gbr.fit(X_train, y_train)
preds = gbr.predict(X_test)
gbr_res = metrics.r2_score(y_test, preds)
print('R2 gbr: ', gbr_res)
plt.hist(y_test - preds)
plt.hist(y_test)
plt.hist(preds)
df = pd.read_csv('../input/dataisbeautiful/r_dataisbeautiful_posts.csv')
df.drop(columns=['id', 'author_flair_text', 'removed_by', 'total_awards_received', 'awarders', 'created_utc', 'full_link'], inplace=True)
df.dropna(inplace = True)
df.head()
import string
# реализуем предобработку
def preprocess(doc):
    # к нижнему регистру
    doc = doc.lower()
    # убираем пунктуацию, пробелы, прочее
    for p in string.punctuation + string.whitespace:
        doc = doc.replace(p, ' ')
    # убираем лишние пробелы, объединяем обратно
    doc = doc.strip()
    doc = ' '.join([w for w in doc.split(' ') if w != ''])
    return doc
#  применим к этим столбцам нашу функцию понижения текста
for colname in df.select_dtypes(include= np.object).columns:
    df[colname] = df[colname].map(preprocess)
df.head()
encoded_data, encoders = number_encode_features(df.drop(columns='title')) # Теперь encoded data содержит закодированные кат. признаки 
encoded_data.head()
y = np.array(encoded_data['over_18'].astype(int))
X = np.array(encoded_data.drop(['over_18'], axis=1))

from sklearn.preprocessing import scale
X_scaled = scale(np.array(X, dtype='float'), with_std=True, with_mean=True)
X
# импортируем tfidf преобразование
from sklearn.feature_extraction.text import TfidfVectorizer

# инициализировали алгоритм
vectorizer = TfidfVectorizer(max_features = 100)
# преобразовали его в матрицу tfidf как в примере на картинке выше
X_np = vectorizer.fit_transform(df['title'].values)
# отобразили его размерность
print(X_np.shape)
import gc
del df, encoded_data
gc.collect()
# т.к. сам тип матрицы из scipy - преобразуем в tfidf
X_np = X_np.toarray()
# отобразим произвольные слова
#print(vectorizer.get_feature_names()[13000:13010])
# добавление TF-IDF к X
X = np.append(X, X_np, axis=1)
X = np.delete(X, not_over_18_ids, 0)
y = np.delete(y, not_over_18_ids, 0)
X.shape, y.shape
pd.Series(y).value_counts()
X
X_scaled = scale(np.array(X, dtype='float'), with_std=True, with_mean=True)
X_scaled
X_scaled.shape
# разбиваем отбалансированные, но ненормализованные
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# X_train = X[:int(len(X)*0.8)]
# y_train = y[:int(len(X)*0.8)]

# X_test = X[int(len(X)*0.8):]
# y_test = y[int(len(X)*0.8):]
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean)/std
X_test = (X_test - mean)/std
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
knn = KNeighborsClassifier()
grid = {'n_neighbors': np.array(np.linspace(30, 50, 20), dtype='int')}
gs = GridSearchCV(knn, grid)
gs.fit(X_scaled, y)
gs.best_params_, gs.best_score_
knn = KNeighborsClassifier(n_neighbors = gs.best_params_['n_neighbors'], n_jobs=-1)

knn.fit(X_train, y_train)
preds = knn.predict(X_test)
knn_tfidf = metrics.classification_report(y_test, preds)
print(knn_tfidf)
alg = SVC()

grid = {'C': np.array(np.linspace(0.1, 5, 10), dtype='float'),
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        }

gs = GridSearchCV(alg, grid, verbose=2, n_jobs = -1)
gs.fit(X_scaled, y)
gs.best_params_, gs.best_score_
svm = SVC(C=gs.best_params_['C'], kernel = gs.best_params_['kernel'])

svm.fit(X_train, y_train)
preds = svm.predict(X_test)
svm_tfidf = metrics.classification_report(y_test, preds)
print(svm_tfidf)
alg = LogisticRegression()

grid = {'penalty': ['l1', 'l2'],
        'C': np.array(np.logspace(-3, 2, num = 10), dtype='float'),
        }

gs = GridSearchCV(alg, grid, verbose=2, n_jobs = -1)
gs.fit(X_scaled, y)
gs.best_params_, gs.best_score_
logreg = LogisticRegression(penalty=gs.best_params_['penalty'], C = gs.best_params_['C'])

logreg.fit(X_train, y_train)
preds = logreg.predict(X_test)
logreg_tfidf = metrics.classification_report(y_test, preds)
print(logreg_tfidf)
sgd = SGDRegressor()

grid = {'penalty': ['l1', 'l2'],
        'alpha': [1e-4, 1e-5, 1e-6, 1e-7]}

gs = GridSearchCV(sgd, grid, verbose = 2, scoring = 'r2')
gs.fit(X_scaled, y)
gs.best_params_, gs.best_score_
sgd = SGDRegressor(alpha = gs.best_params_['alpha'], penalty = gs.best_params_['penalty'])
sgd.fit(X_train, y_train)
preds = sgd.predict(X_test)
sgd_tfidf = metrics.r2_score(y_test, preds)
print('R2 sgd (sklearn): ', sgd_tfidf)
plt.hist(y_test - preds)
gbr = GradientBoostingRegressor()

grid = {'max_depth': [3, 4, 5],
        'min_samples_split': [2, 3, 4, 5]}

gs = GridSearchCV(gbr, grid, verbose = 2)
gs.fit(X_scaled, y)
gs.best_params_, gs.best_score_
gbr = GradientBoostingRegressor(max_depth = gs.best_params_['max_depth'], min_samples_split = gs.best_params_['min_samples_split'])
gbr.fit(X_train, y_train)
preds = gbr.predict(X_test)
gbr_tfidf = metrics.r2_score(y_test, preds)
print('R2 gb: ', gbr_tfidf)
plt.hist(y_test - preds)
plt.hist(y_test)
plt.hist(preds)
print('KNN\n', knn_res, knn_tfidf)
print('SVM\n', svm_res, svm_tfidf)
print('LogReg\n', logreg_res, logreg_tfidf)
print('SGD\n', sgd_res, sgd_tfidf)
print('GBR\n', gbr_res, gbr_tfidf)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.naive_bayes import ComplementNB, MultinomialNB, BernoulliNB

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('../input/dataisbeautiful/r_dataisbeautiful_posts.csv')
df.drop(columns=['id', 'author_flair_text', 'removed_by', 'total_awards_received', 'awarders', 'created_utc', 'full_link'], inplace=True)
df.dropna(inplace = True)
df['over_18'] = df['over_18'].astype(int)
df.head()
df_train, df_test = train_test_split(df, test_size=0.2, shuffle=True)
import nltk
nltk.download("stopwords")

from nltk.corpus import stopwords
stopWords = stopwords.words('english')

from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

# ngram_range=(1, 2) - это сами слова (unigrams) и пары слов(bigrams)
vectorizer = TfidfVectorizer(stop_words=stopWords + list(ENGLISH_STOP_WORDS), ngram_range=(1, 2))
vectorizer = vectorizer.fit(df['title'])

X_train_vectors = vectorizer.transform(df_train['title'])
X_test_vectors = vectorizer.transform(df_test['title'])
# массив значений разреженной матрицы 65-й строки в CSR-формате 
num = 65
X_train_vectors[num].data
# исходный title 65-й строки
df_train['title'].iloc[65]
# Выведем слова и пары слов, составляющие title 65-й строки, в порядке увеличения их меры TF-IDF:
vectorizer.inverse_transform(X_train_vectors[num])[0][np.argsort(X_train_vectors[num].data)]
from sklearn.model_selection import GridSearchCV

alg = ComplementNB()

grid = {'alpha': np.array(np.linspace(0, 6, 30), dtype='float'),}

gs = GridSearchCV(alg, grid, verbose=2, n_jobs = -1, scoring = 'f1')
gs.fit(X_train_vectors, df_train['over_18'])
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
grid_plot(grid['alpha'], gs.cv_results_['mean_test_score'], 'alpha', 'ComplementNB')
clf = ComplementNB(alpha = gs.best_params_['alpha'])

clf.fit(X_train_vectors, df_train['over_18'])
predicts = clf.predict(X_test_vectors)
compnb_tfidf_ngr_stpwds = classification_report(df_test['over_18'], predicts)
print(compnb_tfidf_ngr_stpwds)
df = pd.read_csv('../input/dataisbeautiful/r_dataisbeautiful_posts.csv')
df.drop(columns=['id', 'author_flair_text', 'removed_by', 'total_awards_received', 'awarders', 'created_utc', 'full_link'], inplace=True)
df.dropna(inplace = True)
df['over_18'] = df['over_18'].astype(int)
df.head()
# np.argwhere вернет индексы тех элементов массива y (целевой переменной), где значение 0
not_over_18_ids = np.argwhere(np.array(df['over_18']) == 0).flatten()
print('Всего не 18+', len(not_over_18_ids))
not_over_18_ids
from sklearn.utils import shuffle

not_over_18_ids = shuffle(not_over_18_ids, random_state = 42)
# найдем "лишних", для этого обрежем найденные id на кол-во over_18 (внутри len)
not_over_18_ids = not_over_18_ids[len(np.argwhere(np.array(df['over_18']) == 1).flatten()):]
print(len(not_over_18_ids))
# отображаем кол-во и сами id, которые мы должны выкинуть
not_over_18_ids
# из X и y выкинем избыточные нули
df = df.loc[df.index.difference(df.iloc[not_over_18_ids].index)]
pd.Series(df['over_18']).value_counts()
df_train, df_test = train_test_split(df, test_size=0.2, shuffle=True)
vectorizer = vectorizer.fit(df['title'])

X_train_vectors = vectorizer.transform(df_train['title'])
X_test_vectors = vectorizer.transform(df_test['title'])
from sklearn.model_selection import GridSearchCV

alg = ComplementNB()

grid = {'alpha': np.array(np.linspace(0, 6, 30), dtype='float'),}

gs = GridSearchCV(alg, grid, verbose=2, n_jobs = -1, scoring = 'f1')
gs.fit(X_train_vectors, df_train['over_18'])
gs.best_params_, gs.best_score_
grid_plot(grid['alpha'], gs.cv_results_['mean_test_score'], 'alpha', 'ComplementNB')
clf = ComplementNB(alpha = gs.best_params_['alpha'])

clf.fit(X_train_vectors, df_train['over_18'])
predicts = clf.predict(X_test_vectors)
compnb_tfidf_ngr_stpwds = classification_report(df_test['over_18'], predicts)
print(compnb_tfidf_ngr_stpwds)
print('LogReg\n', logreg_tfidf)
