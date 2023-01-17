import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
df = pd.read_csv('../input/dou-jobs/df_26-12-2019.csv', index_col=0)
df.head()
df_1 = df[['name', 'description', 'salary']]
df_1.head()
# Создаём корпус текстов
corpus = ['Студент студент написал заявление в деканате на имя декана.',
          'Студент получил студенческий билет у декана.',
          'В апреле было тепло и солнце светило ярко.'
         ]

# Подключаем класс CountVectorizer, он реализует модель "мешок слов"
from sklearn.feature_extraction.text import CountVectorizer

# Создаём экземпляр модели с параметрами по умолчанию
bow = CountVectorizer()

# Обучаем мешок слов на корпусе и преобразовываем текст в фичи
bow_matrix = bow.fit_transform(corpus)

# Смотрим, какие фичи получились
bow.get_feature_names()
# Создаём датафрейм с использованием матрицы bow_matrix и колонками --- названиями фич
# Метод toarray() нужен для преобразования разреженной матрицы в обычную
pd.DataFrame(bow_matrix.toarray(), columns=bow.get_feature_names())
# Создаём список стоп-слов
stop_words_rus=['на', 'в', 'из', 'под', 'у'] # можно взять готовый список в библиотеке

# Создаём экземпляр модели с параметром stop_words, обучаем на корпусе и смотрим результаты
bow = CountVectorizer(stop_words=stop_words_rus)

bow_matrix = bow.fit_transform(corpus)
pd.DataFrame(bow_matrix.toarray(), columns=bow.get_feature_names())
# Словарь из биграмм
bow = CountVectorizer(stop_words=stop_words_rus, ngram_range=(2, 2))

bow_matrix = bow.fit_transform(corpus)
pd.DataFrame(bow_matrix.toarray(), columns=bow.get_feature_names())
# Словарь из отдельных слов + биграмм
bow = CountVectorizer(stop_words=stop_words_rus, ngram_range=(1, 2))

bow_matrix = bow.fit_transform(corpus)
pd.DataFrame(bow_matrix.toarray(), columns=bow.get_feature_names())
# Убираем популярные слова
bow = CountVectorizer(stop_words=stop_words_rus, max_df=0.5)

bow_matrix = bow.fit_transform(corpus)
pd.DataFrame(bow_matrix.toarray(), columns=bow.get_feature_names())
# Убираем редкие слова
bow = CountVectorizer(stop_words=stop_words_rus, min_df=2)

bow_matrix = bow.fit_transform(corpus)
pd.DataFrame(bow_matrix.toarray(), columns=bow.get_feature_names())
# Подключаем класс TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Создаём экземпляр модели с параметрами по умолчанию
tfidf = TfidfVectorizer()

# Обучаем TF-IDF на корпусе и преобразовываем текст в фичи
tfidf_matrix = tfidf.fit_transform(corpus)

# Смотрим, какие фичи получились
tfidf.get_feature_names()
tfidf = TfidfVectorizer(stop_words=stop_words_rus)

tfidf_matrix = tfidf.fit_transform(corpus)
pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names())
# Подключимся с библиотеке stop_words, которая содержит стоп-слова разных языков
from stop_words import get_stop_words

stop_words_ukr = get_stop_words('ukrainian')
print('UKR stop words:\n', stop_words_ukr[:40])

stop_words_rus = get_stop_words('russian')
print('\nRUS stop words:\n', stop_words_rus[:40])
# Так как наш корпус на двух языках, объединим два списка стоп-слов в один
stop_words_ukr_rus = list(set(stop_words_ukr) | set(stop_words_rus))
# Преобразуем каждую текстовую колонку в "мешок слов"
bow = CountVectorizer(stop_words=stop_words_ukr_rus, min_df=5)

name_bow = bow.fit_transform(df_1['name'])
desc_bow = bow.fit_transform(df_1['description'])

print(name_bow.shape, desc_bow.shape)
# Посмотрим, какие фичи выделились по описаниям вакансий
bow.get_feature_names()[-20:]
# Объединим две матрицы в одну
from scipy.sparse import hstack
X = hstack([name_bow, desc_bow])

print(type(X))
print(X.shape)
print(X.count_nonzero())
# Попробуем обучить линейную регрессию с параметрами по умолчанию на полученной матрице
y = df_1['salary']

from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=1)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_valid)
# Вычислим различные метрики
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score
print('MSE:', mean_squared_error(y_valid, y_pred))
print('MAE:', mean_absolute_error(y_valid, y_pred))
print('MedAE:', median_absolute_error(y_valid, y_pred))
print('R2:', r2_score(y_valid, y_pred))
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

sns.distplot(df_1['salary'])
plt.show()
sns.boxplot(df_1['salary'])
plt.show()
# Видим, что распределение зарплат сильно смещено влево, 
# поэтому применим логарифмическое преобразование
df_1['log_salary'] = np.log(df_1['salary'])

sns.distplot(df_1['log_salary'])
plt.show()
sns.boxplot(df_1['log_salary'])
plt.show()
# Переучим модель на новых данных
y = df_1['log_salary']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=1)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_valid)
# Вычислим метрики
print('MSE:', mean_squared_error(y_valid, y_pred))
print('MAE:', mean_absolute_error(y_valid, y_pred))
print('MedAE:', median_absolute_error(y_valid, y_pred))
print('R2:', r2_score(y_valid, y_pred))
# Визуализируем отклонение результатов нашей модели от истинных
g = sns.pointplot(x=y_valid.index[:20], y=y_valid[:20], color='blue', label='True')
g = sns.pointplot(x=y_valid.index[:20], y=y_pred[:20], color='green', label='Prediction')
g.set_xticklabels(np.arange(20))
plt.legend()
plt.show()
df=df_1[['name', 'description']]
df['nd'] = df['name'] + ' ' + df['description']
df
bow = CountVectorizer(stop_words=stop_words_ukr_rus, min_df=5)

X = bow.fit_transform(df['nd'])
y = df_1['log_salary']

from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=1)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_valid)
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score
print('MSE:', mean_squared_error(y_valid, y_pred))
print('MAE:', mean_absolute_error(y_valid, y_pred))
print('MedAE:', median_absolute_error(y_valid, y_pred))
print('R2:', r2_score(y_valid, y_pred))
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
# using default tokenizer in TfidfVectorizer
tfidf = TfidfVectorizer(ngram_range=(1, 1))
X = tfidf.fit_transform(df['nd'])
y = df_1['log_salary']

from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=1)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_valid)
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score
print('MSE:', mean_squared_error(y_valid, y_pred))
print('MAE:', mean_absolute_error(y_valid, y_pred))
print('MedAE:', median_absolute_error(y_valid, y_pred))
print('R2:', r2_score(y_valid, y_pred))
tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words=stop_words_ukr_rus, min_df=5)
X = tfidf.fit_transform(df['nd'])
y = df_1['log_salary']

from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=1)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_valid)
print('MSE:', mean_squared_error(y_valid, y_pred))
print('MAE:', mean_absolute_error(y_valid, y_pred))
print('MedAE:', median_absolute_error(y_valid, y_pred))
print('R2:', r2_score(y_valid, y_pred))