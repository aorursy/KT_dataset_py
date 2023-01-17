import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('../input/dataisbeautiful/r_dataisbeautiful_posts.csv')

df.head()
df.describe()
df.info()
# визуально посмотрим на распределение незаполненных значений по признакам

fig, ax = plt.subplots(figsize=(14,10))

sns.heatmap(df.isnull(), cbar=False, cmap="YlGnBu_r")

plt.show()
# удалим признаки с наибольшим количеством незаполненных данных, а также не влияющие на итоговую классификацию

df.drop(columns=['id', 'author_flair_text', 'removed_by', 'total_awards_received', 'awarders', 'created_utc', 'full_link'], inplace=True)

# удалим строку с пустым 'title'

df.dropna(inplace=True)
df.head()
# ещё раз визуально проверим распределение незаполненных значений по признака

fig, ax = plt.subplots(figsize=(14,10))

sns.heatmap(df.isnull(), cbar=False, cmap="YlGnBu_r")

plt.show()
# преобразуем булевый признак 'over_18' в числовые значения (0 и 1)

df['over_18'] = df['over_18'].astype(int)
# посмотрим на распределение признака по классам

df['over_18'].value_counts()

# видим, что классы несбалансированные
# посмотрим на признаки с категориальными данными

df.select_dtypes(include= np.object).head()
# закодируем признак 'author' числовыми значениям

from sklearn.preprocessing import LabelEncoder



class_le = LabelEncoder()

encoded_df = df.copy()

encoded_df['author'] = class_le.fit_transform(encoded_df['author'].values)

encoded_df.head()
# построим гистограммы различных признаков для оценки корректности данных

encoded_df.hist(figsize=(18, 8), layout=(2,2), bins=20)
# построим на матрице корреляций зависимость между признаками, а также между признаками и целевой переменной

plt.subplots(figsize=(12, 10))

sns.heatmap(encoded_df.corr(), square = True, annot=True)

plt.show()
encoded_df.info()
# отобразим размерность матрицы с признаками перед последующей обработкой

encoded_df.shape
# для оперативности дальнейших вычислений обрежем данные, предварительно их перемешав

encoded_df = encoded_df.sample(frac=1).reset_index(drop=True)

encoded_df = encoded_df[:50000]
# проверим размерность обрезанной матрицы

encoded_df.shape
# создадим функцию для предобработки текста в оставшемся признаке 'title'

import string



def preprocess(doc):

    doc = doc.lower()

    for p in string.punctuation + string.whitespace:

        doc = doc.replace(p, ' ')

    doc = doc.strip()

    doc = ' '.join([w for w in doc.split(' ') if w != ''])

    return doc
# и обработаем 'title'

for colname in encoded_df.select_dtypes(include= np.object).columns:

    encoded_df[colname] = encoded_df[colname].map(preprocess)

encoded_df.head()
# преобразуем текст признака 'title'  в матрицу tfidf, ограничив итоговое количество признаков

from sklearn.feature_extraction.text import TfidfVectorizer



vectorizer = TfidfVectorizer(max_features=10000)

X_np = vectorizer.fit_transform(encoded_df['title'].values)
# посмотрим размерность получившейся после обработки матрицы

X_np.shape
# посмотрим на названия признаков, получившихя после обработки

print(vectorizer.get_feature_names()[7880:7890])
# выделим обучающую выборку и целевую переменную

X = np.array(encoded_df.drop(columns=['over_18', 'title']), float)

y = np.array(encoded_df['over_18'])

X.shape, y.shape
# разбиваем данные на обучающие и испытательные наборы

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

X_train.shape, X_test.shape, y_train.shape, y_test.shape
# приведём признаки к одному и тому же масштабу с помощью стандратизации

from sklearn.preprocessing import StandardScaler



stdsc = StandardScaler()

X_train = stdsc.fit_transform(X_train)

X_test = stdsc.transform(X_test)
# посмотрим на получившуюся размерность признаков после стандратизации

X_train, X_test
# добавляем к обучающим данным матрицу признаков tfidf

X_train = np.append(X_train, X_np.toarray()[:40000], axis=1)

X_test = np.append(X_test, X_np.toarray()[40000:], axis=1)

X_train.shape, X_test.shape
from sklearn.model_selection import GridSearchCV

from sklearn import metrics

from sklearn.metrics import confusion_matrix, plot_confusion_matrix



import warnings

warnings.filterwarnings("ignore")
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV



# перебор параметров с помощью GridSearchCV занимает длительное время

# knn = KNeighborsClassifier()

# knn_grid = {'n_neighbors': np.array(np.linspace(1, 100, 5), dtype='int')}

# gs = GridSearchCV(knn, knn_grid, cv=3)

# gs.fit(X_train, y_train)

# gs.best_params_, gs.best_score_



# поэтому применим KNeighborsClassifier со стандратными настройками

knn = KNeighborsClassifier()

knn_mtx = knn.fit(X_train, y_train)

# делаем предсказания на тестовой выборке и выводим метрики  

y_knn = knn.predict(X_test)

print(metrics.classification_report(y_test, y_knn))
# отобразим результаты на conf-matrix

plot_confusion_matrix(knn_mtx, X_test, y_test, display_labels=['0','1'], cmap="Blues", values_format = '')

# видим, что при высокой точности алгоритм совсем не распознаёт класс 'True'
from sklearn.linear_model import LogisticRegression



# переберём модели с различными параметрами с помощью GridSearchCV

alg = LogisticRegression()

grid = {'penalty': ['l1', 'l2'],

        'C': np.array(np.logspace(-3, 2, num = 5), dtype='float'),

        }

gs = GridSearchCV(alg, grid, verbose=2)

gs.fit(X_train, y_train)
# посмотрим лучшие параметры, получившиеся в результате GridSearchCV

gs.best_params_, gs.best_score_
# инициализируем алгоритм с лучшими параметрами без балансировки весов и обучаем модель

logreg = LogisticRegression(penalty='l2', C = 0.001)

logreg_mtx = logreg.fit(X_train, y_train)
# делаем предсказания на тестовой выборке и выводим метрики (без балансировки)

y_logreg = logreg.predict(X_test)

print(metrics.classification_report(y_test, y_logreg))
# отобразим результаты без балансировки весов на conf-matrix 

plot_confusion_matrix(logreg_mtx, X_test, y_test, display_labels=['0','1'], cmap="Blues", values_format = '')

# видим, что при высокой точности алгоритм совсем не распознаёт класс 'True'
# инициализируем алгоритм с лучшими параметрами и с балансировкой весов

logreg_balanced = LogisticRegression(penalty='l2', C = 0.001, class_weight='balanced')

logreg_balanced_mtx = logreg_balanced.fit(X_train, y_train)
# делаем предсказания на тестовой выборке и выводим метрики (с балансировкой)

y_logreg_balanced = logreg_balanced.predict(X_test)

print(metrics.classification_report(y_test, y_logreg_balanced))
# отобразим результаты с балансировкой весов на conf-matrix 

plot_confusion_matrix(logreg_balanced_mtx, X_test, y_test, display_labels=['0','1'], cmap="Blues", values_format = '')

# видим, что точность предсказаний упала, но при этом алгоритм смог верно распознать несколько образцов класса 'True' 
from sklearn.svm import SVC



# перебор параметров с помощью GridSearchCV занимает длительное время

# alg = SVC()

# grid = {'C': np.array(np.linspace(0, 100, 5), dtype='float'),

#        'kernel': ['rbf', 'sigmoid'],

#        }

# gs = GridSearchCV(alg, grid, verbose=2)

# gs.fit(X_train, y_train)



# поэтому применим SVM со стандратными настройками и сразу с балансировкой весов

svm = SVC(class_weight='balanced')

svm_mtx = svm.fit(X_train, y_train)
# делаем предсказания на тестовой выборке и выводим метрики

y_svm = svm.predict(X_test)

print(metrics.classification_report(y_test, y_svm))
# отобразим результаты на conf-matrix 

plot_confusion_matrix(svm_mtx, X_test, y_test, display_labels=['0','1'], cmap="Blues", values_format = '')

# видим, что аналогично  KNN, при высокой точности алгоритм совсем не распознаёт класс 'True'
from sklearn.ensemble import RandomForestClassifier



# инициализируем алгоритм и обучаем модель

rfc = RandomForestClassifier(class_weight='balanced')

rfc_mtx = rfc.fit(X_train, y_train)
# делаем предсказания на тестовой выборке и выводим метрики

y_rfc = rfc.predict(X_test)

print(metrics.classification_report(y_test, y_rfc))
# отобразим результаты на conf-matrix 

plot_confusion_matrix(rfc_mtx, X_test, y_test, display_labels=['0','1'], cmap="Blues", values_format = '')

# видим, что как в KNN, SVM и в логистической регрессии без (балансировки весов), при высокой точности алгоритм совсем не распознаёт класс 'True'
## from keras.models import Sequential

from keras.models import Sequential

from keras.layers import Dense



# инициализируем нейросеть с 6 слоями и функцией активации ReLU

model = Sequential()

model.add(Dense(units = 128, activation = 'relu' , input_dim = X_train.shape[1]))

model.add(Dense(units = 64 , activation = 'relu'))

model.add(Dense(units = 32 , activation = 'relu'))

model.add(Dense(units = 32 , activation = 'relu'))

model.add(Dense(units = 16 , activation = 'relu'))

model.add(Dense(units = 1 , activation = 'sigmoid'))

model.compile(optimizer = 'adam' , loss = 'binary_crossentropy' , metrics = ['accuracy'])

model.summary()
# обучим модель на 5 эпохах с размером партии в 128 элементов

history = model.fit(X_train, y_train, epochs = 5, batch_size = 128, validation_data = (X_test, y_test))
# делаем предсказания на тестовой выборке и выводим метрики

y_sqn = model.predict_classes(X_test)

print(metrics.classification_report(y_test, y_sqn))
# отобразим результаты на conf-matrix 

cm = confusion_matrix(y_test, y_sqn)

cm = pd.DataFrame(cm, index = ['0', '1'], columns = ['0', '1'])

plt.figure(figsize = (10,10))

sns.heatmap(cm, cmap= "Blues", linecolor = 'black', linewidth = 1, annot = True, fmt='')

# видим, что хоть нейросеть и старалась что-то распознать (есть ошибки в классе 'False'), но в классе 'True' нет верных результатов
# это пока в процессе...

from simpletransformers.classification import ClassificationModel



# Create a TransformerModel

model = ClassificationModel('roberta', 'roberta-base')



# Train the model

model.train_model(train_df)



# Evaluate the model

result, model_outputs, wrong_predictions = model.eval_model(eval_df)