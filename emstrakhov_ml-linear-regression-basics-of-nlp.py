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
# Ridge
from sklearn.linear_model import Ridge
ridge = Ridge() # по умолчанию alpha=1
ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_valid)

print('MSE:', mean_squared_error(y_valid, y_pred))
print('MAE:', mean_absolute_error(y_valid, y_pred))
print('MedAE:', median_absolute_error(y_valid, y_pred))
print('R2:', r2_score(y_valid, y_pred))
# Ridge с подбором alpha
from sklearn.model_selection import GridSearchCV

alpha_grid = {'alpha': np.logspace(-4, 4, 20)} # 20 точек от 10^(-4) до 10^4
ridge_grid = GridSearchCV(ridge, alpha_grid, cv=5, scoring='neg_mean_squared_error') 
ridge_grid.fit(X_train, y_train)
# Посмотрим на наилучшие показатели
print('Best alpha:', ridge_grid.best_params_)
print('\nBest score:', ridge_grid.best_score_)
ridge_best = ridge_grid.best_estimator_
y_pred = ridge_best.predict(X_valid)
print('R2:', r2_score(y_valid, y_pred))
# Валидационная кривая
# По оси х --- значения гиперпараметров (param_alpha)
# По оси y --- значения метрики (mean_test_score)

import matplotlib.pyplot as plt
results_df = pd.DataFrame(ridge_grid.cv_results_)
plt.plot(results_df['param_alpha'], results_df['mean_test_score'])

# Подписываем оси и график
plt.xlabel('alpha')
plt.ylabel('Test accuracy')
plt.title('Validation curve')
plt.show()
# Lasso
from sklearn.linear_model import Lasso
lasso = Lasso() # по умолчанию alpha=1
lasso.fit(X_train, y_train)
y_pred = lasso.predict(X_valid)

print('MSE:', mean_squared_error(y_valid, y_pred))
print('MAE:', mean_absolute_error(y_valid, y_pred))
print('MedAE:', median_absolute_error(y_valid, y_pred))
print('R2:', r2_score(y_valid, y_pred))
eps = 1e-6
lasso_coef = lasso.coef_
print('Нулевых коэффициентов:', sum(np.abs(lasso_coef) < eps))
print('Всего коэффициентов:', lasso_coef.shape[0])
# Lasso с подбором alpha
alpha_grid = {'alpha': np.logspace(-3, 3, 10)} # 10 точек от 10^(-3) до 10^3
lasso_grid = GridSearchCV(lasso, alpha_grid, cv=5, scoring='neg_mean_squared_error') 
lasso_grid.fit(X_train, y_train)
# Посмотрим на наилучшие показатели
print('Best alpha:', lasso_grid.best_params_)
print('\nBest score:', lasso_grid.best_score_)
lasso_best = lasso_grid.best_estimator_
y_pred = lasso_best.predict(X_valid)
print('R2:', r2_score(y_valid, y_pred))
# Валидационная кривая
# По оси х --- значения гиперпараметров (param_alpha)
# По оси y --- значения метрики (mean_test_score)

results_df = pd.DataFrame(lasso_grid.cv_results_)
plt.plot(results_df['param_alpha'], results_df['mean_test_score'])

# Подписываем оси и график
plt.xlabel('alpha')
plt.ylabel('Test accuracy')
plt.title('Validation curve')
plt.show()
eps = 1e-6
lasso_coef = lasso_best.coef_
print('Нулевых коэффициентов:', sum(np.abs(lasso_coef) < eps))
print('Всего коэффициентов:', lasso_coef.shape[0])
# Снова обучим Bag of Words: на этот раз дважды, чтобы получить отдельные словари
bow1 = CountVectorizer(stop_words=stop_words_ukr_rus, min_df=5)
bow1.fit(df_1['name'])
voc1 = bow1.vocabulary_

bow2 = CountVectorizer(stop_words=stop_words_ukr_rus, min_df=5)
bow2.fit(df_1['description'])
voc2 = bow2.vocabulary_

# Объединяем в общий словарь
voc = list(voc1) + list(voc2)
print('Длина общего словаря:', len(voc))
# Посмотрим, что в словаре
print(voc[:2])
non_zero = np.abs(lasso_coef) > eps # маска: массив True/False значений
non_zero_words = np.array(voc)[non_zero] # наложили маску на словарь, получили нужные слова
print(non_zero_words[:100])
ind_sorted = np.argsort(lasso_coef)[::-1]
top_20 = ind_sorted[:20]
print(np.array(voc)[top_20])
print(lasso_coef[top_20])
zero = np.abs(lasso_coef) < eps 
zero_words = np.array(voc)[zero] # наложили маску на словарь, получили нужные слова
print(zero_words[:50])
