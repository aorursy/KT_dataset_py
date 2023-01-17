# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score
from stop_words import get_stop_words
def print_scores(y_valid, y_pred):
    '''
    Функция для быстрого вывода четырёх метрик для регрессии.
    
    y_valid --- истинные значения
    y_pred --- предсказанные моделью значения
    '''
    print('MSE:', mean_squared_error(y_valid, y_pred))
    print('MAE:', mean_absolute_error(y_valid, y_pred))
    print('MedAE:', median_absolute_error(y_valid, y_pred))
    print('R2:', r2_score(y_valid, y_pred))
df = pd.read_csv('../input/dou-jobs/df_26-12-2019.csv', index_col=0)
df.head()
df_1 = df[['name', 'description', 'salary']]
df_1['log_salary'] = np.log(df_1['salary'])
df_1.head()
stop_words_ukr = get_stop_words('ukrainian')
stop_words_rus = get_stop_words('russian')
stop_words_ukr_rus = list(set(stop_words_ukr) | set(stop_words_rus))
bow = CountVectorizer(stop_words=stop_words_ukr_rus)
lin_reg = LinearRegression()

# Составим пайплайн из "мешка слов" и линейной регрессии
model = Pipeline([('bow', bow), ('lin_reg', lin_reg)])
X_text = df_1['description']
y = df_1['log_salary']

X_train, X_valid, y_train, y_valid = train_test_split(X_text, y, test_size=0.25, random_state=1)

# Обучаем модель-пайплайн
model.fit(X_train, y_train)
# Предсказываем с помощью пайплайна
y_pred = model.predict(X_valid)
print_scores(y_valid, y_pred)
# Pipeline + GridSearchCV
param_grid = {'bow__min_df': [2, 5, 10]}
model_grid = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', 
                          n_jobs=-1) # n_jobs=-1 задействует больше процессоров
model_grid.fit(X_train, y_train)
print('Best (hyper)parameters:', model_grid.best_params_)
print('Best score:', model_grid.best_score_)
y_pred = model_grid.best_estimator_.predict(X_valid)
print_scores(y_valid, y_pred)
# Попробуем настроить ngram_range
param_grid = {'bow__ngram_range': [(1,1), (1,2), (2,2)]}
model_grid = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', 
                          n_jobs=-1) # n_jobs=-1 задействует больше процессоров
model_grid.fit(X_train, y_train)
print('Best (hyper)parameters:', model_grid.best_params_)
print('Best score:', model_grid.best_score_)
y_pred = model_grid.best_estimator_.predict(X_valid)
print_scores(y_valid, y_pred)

bow = CountVectorizer(stop_words=stop_words_ukr_rus)
lin_reg = LinearRegression()

# Составим пайплайн из "мешка слов" и линейной регрессии
model = Pipeline([('bow', bow), ('lin_reg', lin_reg)])

X_text = df_1['name']
y = df_1['log_salary']

X_train, X_valid, y_train, y_valid = train_test_split(X_text, y, test_size=0.25, random_state=1)

# Обучаем модель-пайплайн
model.fit(X_train, y_train)
# Предсказываем с помощью пайплайна
y_pred = model.predict(X_valid)

print('MSE:', mean_squared_error(y_valid, y_pred))
print('R2:', r2_score(y_valid, y_pred))
# Pipeline + GridSearchCV
param_grid = {'bow__min_df': np.arange(1, 201)}
model_grid = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', 
                          n_jobs=-1) # n_jobs=-1 задействует больше процессоров
model_grid.fit(X_train, y_train)
print('Best (hyper)parameters:', model_grid.best_params_)
print('Best score:', model_grid.best_score_)
y_pred = model_grid.best_estimator_.predict(X_valid)
print('MSE:', mean_squared_error(y_valid, y_pred))
print('R2:', r2_score(y_valid, y_pred))
tfidf = TfidfVectorizer(stop_words=stop_words_ukr_rus)
lin_reg = LinearRegression()


# Составим пайплайн из "мешка слов" и линейной регрессии
model = Pipeline([('tfidf', tfidf), ('lin_reg', lin_reg)])

X_text = df_1['name']
y = df_1['log_salary']

X_train, X_valid, y_train, y_valid = train_test_split(X_text, y, test_size=0.25, random_state=1)

# Обучаем модель-пайплайн
model.fit(X_train, y_train)

# Предсказываем с помощью пайплайна
y_pred = model.predict(X_valid)

print('MSE:', mean_squared_error(y_valid, y_pred))
print('R2:', r2_score(y_valid, y_pred))
# Pipeline + GridSearchCV
param_grid = {'tfidf__min_df': np.arange(1, 101)}
model_grid = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', 
                          n_jobs=-1) # n_jobs=-1 задействует больше процессоров
model_grid.fit(X_train, y_train)

print('Best (hyper)parameters:', model_grid.best_params_)
print('Best score:', model_grid.best_score_)

y_pred = model_grid.best_estimator_.predict(X_valid)
print('MSE:', mean_squared_error(y_valid, y_pred))
print('R2:', r2_score(y_valid, y_pred))
tfidf = TfidfVectorizer(stop_words=stop_words_ukr_rus, min_df=6)

name_tfidf = tfidf.fit_transform(df_1['name'])

X = name_tfidf
y = df_1['log_salary']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=1)


# Ridge
from sklearn.linear_model import Ridge
ridge = Ridge(random_state=1) # по умолчанию alpha=1
ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_valid)

print('MSE:', mean_squared_error(y_valid, y_pred))
print('R2:', r2_score(y_valid, y_pred))


# подбор параметров
alpha_grid = {'alpha': np.logspace(-4, 4, 40)} # 20 точек от 10^(-4) до 10^4
ridge_grid = GridSearchCV(ridge, alpha_grid, cv=5, scoring='neg_mean_squared_error') 
ridge_grid.fit(X_train, y_train)

# Посмотрим на наилучшие показатели
print('Best alpha:', ridge_grid.best_params_)
print('\nBest score:', ridge_grid.best_score_)

ridge_best = ridge_grid.best_estimator_
y_pred = ridge_best.predict(X_valid)
print('MSE:', mean_squared_error(y_valid, y_pred))
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
lasso = Lasso(random_state=1) # по умолчанию alpha=1
lasso.fit(X_train, y_train)
y_pred = lasso.predict(X_valid)

print('MSE:', mean_squared_error(y_valid, y_pred))
print('R2:', r2_score(y_valid, y_pred))

# подбор параметров
alpha_grid = {'alpha': np.logspace(-4, 4, 20)} 
lasso_grid = GridSearchCV(lasso, alpha_grid, cv=5, scoring='neg_mean_squared_error') 
lasso_grid.fit(X_train, y_train)

# Посмотрим на наилучшие показатели
print('Best alpha:', lasso_grid.best_params_)
print('\nBest score:', lasso_grid.best_score_)

lasso_best = lasso_grid.best_estimator_
y_pred = lasso_best.predict(X_valid)
print('MSE:', mean_squared_error(y_valid, y_pred))
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
# ElasticNet
from sklearn.linear_model import ElasticNet
enet = ElasticNet(random_state=1)
enet.fit(X_train, y_train)
y_pred = enet.predict(X_valid)

print('MSE:', mean_squared_error(y_valid, y_pred))
print('R2:', r2_score(y_valid, y_pred))

enet_params = {'alpha': np.logspace(-4, 4, 20)} # 20 точек от 10^(-4) до 10^4
enet_grid = GridSearchCV(enet, enet_params, cv=5, scoring='neg_mean_squared_error')
enet_grid.fit(X_train, y_train)

# Посмотрим на наилучшие показатели
print('Best alpha:', enet_grid.best_params_)
print('\nBest score:', enet_grid.best_score_)

enet_best = enet_grid.best_estimator_
y_pred = enet_best.predict(X_valid)
print('MSE:', mean_squared_error(y_valid, y_pred))
print('R2:', r2_score(y_valid, y_pred))
# Валидационная кривая
# По оси х --- значения гиперпараметров (param_alpha)
# По оси y --- значения метрики (mean_test_score)

results_df = pd.DataFrame(enet_grid.cv_results_)
plt.plot(results_df['param_alpha'], results_df['mean_test_score'])

# Подписываем оси и график
plt.xlabel('alpha')
plt.ylabel('Test accuracy')
plt.title('Validation curve')
plt.show()
enet_params = {'l1_ratio': np.linspace(0.1, 0.9, 9)}

enet_grid = GridSearchCV(enet, enet_params, cv=5, scoring='neg_mean_squared_error')
enet_grid.fit(X_train, y_train)

# Посмотрим на наилучшие показатели
print('Best l1_ratio:', enet_grid.best_params_)
print('\nBest score:', enet_grid.best_score_)

enet_best = enet_grid.best_estimator_
y_pred = enet_best.predict(X_valid)
print('MSE:', mean_squared_error(y_valid, y_pred))
print('R2:', r2_score(y_valid, y_pred))
bow = CountVectorizer(stop_words=stop_words_ukr_rus)
lin_reg = LinearRegression()

# Составим пайплайн из "мешка слов" и линейной регрессии
model = Pipeline([('bow', bow), ('lin_reg', lin_reg)])

X_text = df_1['description']
y = df_1['log_salary']

X_train, X_valid, y_train, y_valid = train_test_split(X_text, y, test_size=0.25, random_state=1)

# Обучаем модель-пайплайн
model.fit(X_train, y_train)

# Предсказываем с помощью пайплайна
y_pred = model.predict(X_valid)

print('MSE:', mean_squared_error(y_valid, y_pred))
print('R2:', r2_score(y_valid, y_pred))
# Pipeline + GridSearchCV
param_grid = {'bow__min_df': np.arange(1, 201)}
model_grid = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', 
                          n_jobs=-1) # n_jobs=-1 задействует больше процессоров
model_grid.fit(X_train, y_train)

print('Best (hyper)parameters:', model_grid.best_params_)
print('Best score:', model_grid.best_score_)

y_pred = model_grid.best_estimator_.predict(X_valid)
print('MSE:', mean_squared_error(y_valid, y_pred))
print('R2:', r2_score(y_valid, y_pred))
tfidf = TfidfVectorizer(stop_words=stop_words_ukr_rus)
lin_reg = LinearRegression()


# Составим пайплайн из "мешка слов" и линейной регрессии
model = Pipeline([('tfidf', tfidf), ('lin_reg', lin_reg)])

X_text = df_1['description']
y = df_1['log_salary']

X_train, X_valid, y_train, y_valid = train_test_split(X_text, y, test_size=0.25, random_state=1)

# Обучаем модель-пайплайн
model.fit(X_train, y_train)

# Предсказываем с помощью пайплайна
y_pred = model.predict(X_valid)

print('MSE:', mean_squared_error(y_valid, y_pred))
print('R2:', r2_score(y_valid, y_pred))
# Pipeline + GridSearchCV
param_grid = {'tfidf__min_df': np.arange(1, 201)}
model_grid = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', 
                          n_jobs=-1) # n_jobs=-1 задействует больше процессоров
model_grid.fit(X_train, y_train)

print('Best (hyper)parameters:', model_grid.best_params_)
print('Best score:', model_grid.best_score_)

y_pred = model_grid.best_estimator_.predict(X_valid)
print('MSE:', mean_squared_error(y_valid, y_pred))
print('R2:', r2_score(y_valid, y_pred))
tfidf = TfidfVectorizer(stop_words=stop_words_ukr_rus, min_df=6)
lin_reg = LinearRegression()


# Составим пайплайн из "tfidf" и линейной регрессии
model = Pipeline([('tfidf', tfidf), ('lin_reg', lin_reg)])

X_text = df_1['name']
y = df_1['log_salary']

X_train, X_valid, y_train, y_valid = train_test_split(X_text, y, test_size=0.25, random_state=1)

# Обучаем модель-пайплайн
model.fit(X_train, y_train)

# Предсказываем с помощью пайплайна
y_pred = model.predict(X_valid)

print('MSE:', mean_squared_error(y_valid, y_pred))
print('R2:', r2_score(y_valid, y_pred))
# Pipeline + GridSearchCV
param_grid = {'tfidf__ngram_range': [(1,1), (1,2), (2,2)]}
model_grid = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', 
                          n_jobs=-1) # n_jobs=-1 задействует больше процессоров
model_grid.fit(X_train, y_train)

print('Best (hyper)parameters:', model_grid.best_params_)
print('Best score:', model_grid.best_score_)

y_pred = model_grid.best_estimator_.predict(X_valid)
print('MSE:', mean_squared_error(y_valid, y_pred))
print('R2:', r2_score(y_valid, y_pred))
tfidf = TfidfVectorizer(stop_words=stop_words_ukr_rus, min_df=6, ngram_range=(1, 2))
lin_reg = LinearRegression()


# Составим пайплайн из "tfidf" и линейной регрессии
model = Pipeline([('tfidf', tfidf), ('lin_reg', lin_reg)])

X_text = df_1['name']
y = df_1['log_salary']

X_train, X_valid, y_train, y_valid = train_test_split(X_text, y, test_size=0.25, random_state=1)

# Обучаем модель-пайплайн
model.fit(X_train, y_train)

# Предсказываем с помощью пайплайна
y_pred = model.predict(X_valid)

print('MSE:', mean_squared_error(y_valid, y_pred))
print('R2:', r2_score(y_valid, y_pred))
tfidf = TfidfVectorizer(stop_words=stop_words_ukr_rus, min_df=6, ngram_range=(2, 2))
lin_reg = LinearRegression()


# Составим пайплайн из "tfidf" и линейной регрессии
model = Pipeline([('tfidf', tfidf), ('lin_reg', lin_reg)])

X_text = df_1['name']
y = df_1['log_salary']

X_train, X_valid, y_train, y_valid = train_test_split(X_text, y, test_size=0.25, random_state=1)

# Обучаем модель-пайплайн
model.fit(X_train, y_train)

# Предсказываем с помощью пайплайна
y_pred = model.predict(X_valid)

print('MSE:', mean_squared_error(y_valid, y_pred))
print('R2:', r2_score(y_valid, y_pred))
tfidf = TfidfVectorizer(stop_words=stop_words_ukr_rus, min_df=6, ngram_range=(2, 3))
lin_reg = LinearRegression()


# Составим пайплайн из "tfidf" и линейной регрессии
model = Pipeline([('tfidf', tfidf), ('lin_reg', lin_reg)])

X_text = df_1['name']
y = df_1['log_salary']

X_train, X_valid, y_train, y_valid = train_test_split(X_text, y, test_size=0.25, random_state=1)

# Обучаем модель-пайплайн
model.fit(X_train, y_train)

# Предсказываем с помощью пайплайна
y_pred = model.predict(X_valid)

print('MSE:', mean_squared_error(y_valid, y_pred))
print('R2:', r2_score(y_valid, y_pred))
tfidf = TfidfVectorizer(stop_words=stop_words_ukr_rus, min_df=6, ngram_range=(3, 3))
lin_reg = LinearRegression()


# Составим пайплайн из "tfidf" и линейной регрессии
model = Pipeline([('tfidf', tfidf), ('lin_reg', lin_reg)])

X_text = df_1['name']
y = df_1['log_salary']

X_train, X_valid, y_train, y_valid = train_test_split(X_text, y, test_size=0.25, random_state=1)

# Обучаем модель-пайплайн
model.fit(X_train, y_train)

# Предсказываем с помощью пайплайна
y_pred = model.predict(X_valid)

print('MSE:', mean_squared_error(y_valid, y_pred))
print('R2:', r2_score(y_valid, y_pred))
tfidf = TfidfVectorizer(stop_words=stop_words_ukr_rus, min_df=147)
lin_reg = LinearRegression()


# Составим пайплайн из "tfidf" и линейной регрессии
model = Pipeline([('tfidf', tfidf), ('lin_reg', lin_reg)])

X_text = df_1['name'] + ' ' + df_1['description']
y = df_1['log_salary']

X_train, X_valid, y_train, y_valid = train_test_split(X_text, y, test_size=0.25, random_state=1)

# Обучаем модель-пайплайн
model.fit(X_train, y_train)

# Предсказываем с помощью пайплайна
y_pred = model.predict(X_valid)

print('MSE:', mean_squared_error(y_valid, y_pred))
print('R2:', r2_score(y_valid, y_pred))
# Pipeline + GridSearchCV
param_grid = {'tfidf__ngram_range': [(1,1), (1,2), (2,2)]}
model_grid = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', 
                          n_jobs=-1) # n_jobs=-1 задействует больше процессоров
model_grid.fit(X_train, y_train)

print('Best (hyper)parameters:', model_grid.best_params_)
print('Best score:', model_grid.best_score_)

y_pred = model_grid.best_estimator_.predict(X_valid)
print('MSE:', mean_squared_error(y_valid, y_pred))
print('R2:', r2_score(y_valid, y_pred))
tfidf = TfidfVectorizer(stop_words=stop_words_ukr_rus, min_df=146, ngram_range=(1, 2))

name_tfidf = tfidf.fit_transform(df_1['name'])
desc_tfidf = tfidf.fit_transform(df_1['description'])

from scipy.sparse import hstack
X = hstack([name_tfidf, desc_tfidf])
y = df_1['log_salary']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=1)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_valid)

print('MSE:', mean_squared_error(y_valid, y_pred))
print('R2:', r2_score(y_valid, y_pred))
tfidf = TfidfVectorizer(stop_words=stop_words_ukr_rus, min_df=146, ngram_range=(1, 1))

name_tfidf = tfidf.fit_transform(df_1['name'])
desc_tfidf = tfidf.fit_transform(df_1['description'])

from scipy.sparse import hstack
X = hstack([name_tfidf, desc_tfidf])
y = df_1['log_salary']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=1)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_valid)

print('MSE:', mean_squared_error(y_valid, y_pred))
print('R2:', r2_score(y_valid, y_pred))
tfidf = TfidfVectorizer(stop_words=stop_words_ukr_rus, min_df=146, ngram_range=(2, 2))

name_tfidf = tfidf.fit_transform(df_1['name'])
desc_tfidf = tfidf.fit_transform(df_1['description'])

from scipy.sparse import hstack
X = hstack([name_tfidf, desc_tfidf])
y = df_1['log_salary']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=1)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_valid)

print('MSE:', mean_squared_error(y_valid, y_pred))
print('R2:', r2_score(y_valid, y_pred))
min_mse = 0.4161
i_min_mse = 30
max_r2 = -0.9898
i_max_r2 = 30
n_mse = []
n_r2 = []
n_i = []

for i in range (1, 251):

    tfidf = TfidfVectorizer(stop_words=stop_words_ukr_rus, min_df=i,ngram_range=(1, 2))

    name_tfidf = tfidf.fit_transform(df_1['name'])
    desc_tfidf = tfidf.fit_transform(df_1['description'])

    X = hstack([name_tfidf, desc_tfidf])
    y = df_1['log_salary']

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=1)

    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    y_pred = lin_reg.predict(X_valid)
    
    mse_i = mean_squared_error(y_valid, y_pred)
    r2_i = r2_score(y_valid, y_pred)
    
    n_mse.append(mse_i)
    n_r2.append(r2_i)
    n_i.append(i)
    
    if mse_i < min_mse:
        min_mse = mse_i
        i_min_mse = i
        
    if r2_i > max_r2:
        max_r2 = r2_i
        i_max_r2 = i

print('min_df: ', i_min_mse)
print('MSE:', min_mse)
print('min_df: ', i_max_r2)
print('R2:', max_r2)
import matplotlib.pyplot as plt
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(22, 10))

ax[0].plot(n_i, n_mse)
ax[0].set_xlabel('min_df')
ax[0].set_ylabel('MSE')

ax[1].plot(n_i, n_r2, linestyle='--', color='green')
ax[1].set_xlabel('min_df')
ax[1].set_ylabel('R2')

plt.show()