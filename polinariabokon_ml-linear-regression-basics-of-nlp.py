import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
df = pd.read_csv('../input/dou-jobs/df_26-12-2019.csv', index_col=0)
df.head()
df_1 = df[['name', 'description', 'salary']]
df_1.head()
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
#применим логарифмическое преобразование
df_1['log_salary'] = np.log(df_1['salary'])
df_1
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
tfidf = TfidfVectorizer(stop_words=stop_words_ukr_rus,  ngram_range=(1, 2))

name_tfidf = tfidf.fit_transform(df_1['name'])
desc_tfidf = tfidf.fit_transform(df_1['description'])

print(name_tfidf.shape, desc_tfidf.shape)
# Объединим две матрицы в одну
from scipy.sparse import hstack
X = hstack([name_tfidf, desc_tfidf])

print(type(X))
print(X.shape)
print(X.count_nonzero())
# Переучим модель на новых данных
y = df_1['log_salary']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=1)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_valid)
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score
print('MSE:', mean_squared_error(y_valid, y_pred))
print('MAE:', mean_absolute_error(y_valid, y_pred))
print('MedAE:', median_absolute_error(y_valid, y_pred))
print('R2:', r2_score(y_valid, y_pred))