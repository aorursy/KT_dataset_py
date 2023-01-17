import numpy as np 
import pandas as pd 
from pandas import Series

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, RobustScaler


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
PATH_to_file = '/kaggle/input/sf-dst-scoring/'


import warnings
warnings.filterwarnings("ignore")
RANDOM_SEED = 42
!pip freeze > requirements.txt
def my_describe(df):
    """Отображение описательных статистик датафрейма в удобной форме"""
    temp = {}
    temp['Имя признака'] = list(df.columns)
    temp['Тип'] = df.dtypes
    temp['Всего значений'] = df.describe(include='all').loc['count']
    temp['Число пропусков'] = df.isnull().sum().values 
    temp['Кол-во уникальных'] = df.nunique().values
    temp['Минимум'] = df.describe(include='all').loc['min']
    temp['Максимум'] = df.describe(include='all').loc['max']
    temp['Среднее'] = df.describe(include='all').loc['mean']
    temp['Медиана'] = df.describe(include='all').loc['50%']
    temp = pd.DataFrame.from_dict(temp, orient='index')
    display(temp.T)
    return
def show_plot_boxplot(df, col_name, bins=10):
    """Построение гистограммы по столбцу и boxplot-а"""
    color_text = plt.get_cmap('PuBu')(0.95)
    plt.style.use('bmh')
    plt.rcParams['figure.figsize'] = (10, 5)
    _, axes = plt.subplots(2, 1)
    axes[0].hist(df, bins=bins)
    axes[0].set_title("Гистограмма и boxplot для признака '"+col_name+"'", color = color_text, fontsize=14)
    axes[1].boxplot(df, vert=False, showmeans = True)
    axes[1].set_title('')
    return
def show_heatmap(title, df, font_scale=1):
    """Отображение связи между признаками на тепловой карте"""
    plt.style.use('seaborn-paper')
    fig, ax = plt.subplots(figsize=(10, 10))
    color_text = plt.get_cmap('PuBu')(0.95)
    sns.set(font_scale=font_scale, style='whitegrid')
    plt.subplot(111)
    h = sns.heatmap(df.corr(), annot = True, cmap= "PuBu", center= 0, fmt='.1g')
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right", rotation_mode="anchor", fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", rotation_mode="anchor", fontsize=12)
    b, t = plt.ylim()
    plt.ylim(b+0.5, t-0.5)
    h.set_title(title,  fontsize=16, color = color_text)
    return
def all_metrics(y_true, y_pred, y_pred_prob):
    """Функция выводит в виде датафрейма значения основных метрик классификации"""
    dict_metric = {}
    P = np.sum(y_true==1)
    N = np.sum(y_true==0)
    TP = np.sum((y_true==1)&(y_pred==1))
    TN = np.sum((y_true==0)&(y_pred==0))
    FP = np.sum((y_true==0)&(y_pred==1))
    FN = np.sum((y_true==1)&(y_pred==0))
    
    dict_metric['P'] = [P,'Дефолт']
    dict_metric['N'] = [N,'БЕЗ дефолта']
    dict_metric['TP'] = [TP,'Истинно дефолтные']
    dict_metric['TN'] = [TN,'Истинно НЕ дефолтные']
    dict_metric['FP'] = [FP,'Ложно дефолтные']
    dict_metric['FN'] = [FN,'Ложно НЕ дефолтные']
    dict_metric['Accuracy'] = [accuracy_score(y_true, y_pred),'Accuracy=(TP+TN)/(P+N)']
    dict_metric['Precision'] = [precision_score(y_true, y_pred),'Точность = TP/(TP+FP)'] 
    dict_metric['Recall'] = [recall_score(y_true, y_pred),'Полнота = TP/P']
    dict_metric['F1-score'] = [f1_score(y_true, y_pred),'Среднее гармоническое Precision и Recall']
    dict_metric['ROC_AUC'] = [roc_auc_score(y_true, y_pred_prob),'ROC-AUC']    

    temp_df = pd.DataFrame.from_dict(dict_metric, orient='index', columns=['Значение', 'Описание метрики'])
    display(temp_df)   
def show_roc_curve(y_true, y_pred_prob):
    """Функция отображает ROC-кривую"""
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    plt.figure()
    plt.plot([0, 1], label='Случайный классификатор', linestyle='--')
    plt.plot(fpr, tpr, label = 'Логистическая регрессия')
    plt.title('Логистическая регрессия ROC AUC = %0.3f' % roc_auc_score(y_true, y_pred_prob))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc = 'lower right')
    plt.show()
def show_confusion_matrix(y_true, y_pred):
    """Функция отображает confusion-матрицу"""
    color_text = plt.get_cmap('PuBu')(0.95)
    class_names = ['Дефолтный', 'НЕ дефолтный']
    cm = confusion_matrix(y_true, y_pred)
    cm[0,0], cm[1,1] = cm[1,1], cm[0,0]
    df = pd.DataFrame(cm, index=class_names, columns=class_names)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), title="Матрица ошибок")
    ax.title.set_fontsize(15)
    sns.heatmap(df, square=True, annot=True, fmt="d", linewidths=1, cmap="PuBu")
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right", rotation_mode="anchor", fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", rotation_mode="anchor", fontsize=12)
    ax.set_ylabel('Предсказанные значения', fontsize=14, color = color_text)
    ax.set_xlabel('Реальные значения', fontsize=14, color = color_text)
    b, t = plt.ylim()
    plt.ylim(b+0.5, t-0.5)
    fig.tight_layout()
    plt.show()
train = pd.read_csv('/kaggle/input/sf-dst-scoring/train.csv')
test = pd.read_csv('/kaggle/input/sf-dst-scoring/test.csv')
print(f'Размерность тренировочного датасета: {train.shape[0]} записей и {train.shape[1]} признаков.')
print(f'Размерность тестового датасета: {test.shape[0]} записей и {test.shape[1]} признаков.')
# Смотрим описательные характеристики тренировочного датасета
my_describe(train)
# Смотрим описательные характеристики тестового датасета
my_describe(test)
# Сохрания ID клиентов из тестового набора для последующего формирования Submission
id_test = test.client_id
train['education'].value_counts().plot.barh()
test['education'].value_counts().plot.barh()
# заполним пропуски в education самым популярным значением (по наборам train, test)
fill_value = train['education'].value_counts().index[0]
train['education'].fillna(fill_value, inplace=True)

fill_value = test['education'].value_counts().index[0]
test['education'].fillna(fill_value, inplace=True)
# Проверим успешность заполнения пропусков для тренировочного набора
print('Всего пропусков на наборе train:', train.isna().sum().sum())
# Проверим успешность заполнения пропусков для тестового набора
print('Всего пропусков на наборе test:', test.isna().sum().sum())
# Создадим списки переменных (client_id не включаем в списки)
# временная переменная
time_cols = ['app_date']

# бинарные переменные (default - целевой признак, не включаем в список)
bin_cols = ['sex', 'car', 'car_type', 'good_work', 'foreign_passport']

# категориальные переменные 
cat_cols = ['education', 'region_rating', 'home_address', 'work_address', 'sna', 'first_time']

# числовые переменные
num_cols = ['age', 'decline_app_cnt', 'score_bki', 'bki_request_cnt', 'income']
# Посмотрим на распределение и boxplot’ы для численных переменных.
for col in num_cols:
    show_plot_boxplot(train[col], col, bins=50)
temp_list = ['age', 'decline_app_cnt', 'bki_request_cnt', 'income']
for col in temp_list:
    train[col] = np.log(train[col] + 1)
    test[col] = np.log(test[col] + 1)
# Посмотрим на распределение и boxplot’ы для численных переменных после логарифмирования
for col in num_cols:
    show_plot_boxplot(train[col], col, bins=50)
train.app_date.sample(5), test.app_date.sample(5)
# Преобразуем признак app_date в нужный формат
train.app_date = pd.to_datetime(train.app_date, format='%d%b%Y')
test.app_date = pd.to_datetime(test.app_date, format='%d%b%Y')
train.app_date.sample(5), test.app_date.sample(5),
# Определим дату самой ранней заявки в датасетах
data_start = min(train.app_date.min(), test.app_date.min())
data_start
train['delta_deys'] = (train.app_date - data_start).dt.days.astype('int')
test['delta_deys'] = (test.app_date - data_start).dt.days.astype('int')
num_cols.append('delta_deys')
train.drop(['app_date'], axis=1, inplace=True)
test.drop(['app_date'], axis=1, inplace=True)
corr_list = num_cols+['default']+['client_id']
show_heatmap('Матрица корреляции между числовыми переменными', train[corr_list])
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].scatter(x=train['client_id'], y=train['delta_deys'], marker='o')
axes[0].set_xlabel('client_id')
axes[0].set_ylabel('delta_deys')
axes[0].set_title('Набор train')
axes[1].scatter(x=test['client_id'], y=test['delta_deys'], marker='o')
axes[1].set_xlabel('client_id')
axes[1].set_ylabel('delta_deys')
axes[1].set_title('Набор test')
plt.show();
train.drop(['client_id'], axis=1, inplace=True)
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
plt.subplots_adjust(wspace = 0.5)
axes = axes.flatten()
for i in range(len(num_cols)):
    sns.boxplot(x="default", y=num_cols[i], data=train, orient = 'v', ax=axes[i], showfliers=True,  showmeans = True)
imp_num = Series(f_classif(train[num_cols], train['default'])[0], index = num_cols)
imp_num.sort_values(inplace = True)
imp_num.plot(kind = 'barh', title='Значимость непрерывных переменных')
print('Анализ бинарных признаков учебного датасета')
for col in bin_cols:
    print(f'Уникальные записи в столбце {col}:')
    print(train[col].value_counts())
print('Анализ бинарных признаков тестового датасета')
for col in bin_cols:
    print(f'Уникальные записи в столбце {col}:')
    print(train[col].value_counts())
# Для бинарных признаков мы будем использовать LabelEncoder
label_encoder = LabelEncoder()

for col in bin_cols:
    train[col] = label_encoder.fit_transform(train[col])
    test[col] = label_encoder.fit_transform(test[col])
    
# Убедимся в преобразовании    
train.head()
label_encoder = LabelEncoder()
train['education'] = label_encoder.fit_transform(train['education'])
test['education'] = label_encoder.fit_transform(test['education'])
# Определим важность признаков
imp_cat = Series(mutual_info_classif(train[bin_cols + cat_cols], train['default'],
                                     discrete_features =True), index = bin_cols + cat_cols)
imp_cat.sort_values(inplace = True)
imp_cat.plot(kind = 'barh', title = 'Значимость бин. и категор. переменных')
sns.countplot(train['default']);
# Посмотрим на данные
train.info()
# Реализуем метод  One Hot Encoding через get_dummies
X_cat = OneHotEncoder(sparse = False).fit_transform(train[cat_cols].values)
X_cat_test = OneHotEncoder(sparse = False).fit_transform(test[cat_cols].values)
# Стандартизация числовых переменных
X_num = StandardScaler().fit_transform(train[num_cols].values)
X_num_test = StandardScaler().fit_transform(test[num_cols].values)

# Объединяем
X = np.hstack([X_num, train[bin_cols].values, X_cat])
Test = np.hstack([X_num_test, test[bin_cols].values, X_cat_test])
y = train['default'].values
# Проверяем размеры наборов train и test
train.shape, test.shape, X.shape, Test.shape
# Разбиваем датасет на тренировочный и тестовый, выделив 20% данных на валидацию
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

# Обучаем модель
model_1 = LogisticRegression(random_state=RANDOM_SEED)

model_1.fit(X_train, y_train)

# Предсказываем
y_pred_prob = model_1.predict_proba(X_test)[:,1]
y_pred = model_1.predict(X_test)

# Оценка качества модели
all_metrics(y_test, y_pred, y_pred_prob)
show_roc_curve(y_test, y_pred_prob)
show_confusion_matrix(y_test, y_pred)
# RobastScaler
X_num = RobustScaler().fit_transform(train[num_cols].values)
X_num_test = RobustScaler().fit_transform(test[num_cols].values)

# Объединяем
X = np.hstack([X_num, train[bin_cols].values, X_cat])
Test = np.hstack([X_num_test, test[bin_cols].values, X_cat_test])
y = train['default'].values

# Разбиваем датасет на тренировочный и тестовый, выделив 20% данных на валидацию
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

# Обучаем модель
model_2 = LogisticRegression(random_state=RANDOM_SEED)

model_2.fit(X_train, y_train)

# Предсказываем
y_pred_prob = model_2.predict_proba(X_test)[:,1]
y_pred = model_2.predict(X_test)

# Оценка качества модели
all_metrics(y_test, y_pred, y_pred_prob)
show_roc_curve(y_test, y_pred_prob)
show_confusion_matrix(y_test, y_pred)
C = np.logspace(0, 4, 10)
iter_ = 50
epsilon_stop = 1e-3
 
hyperparameters = [
    {'penalty': ['l1'], 
     'C': C,
     'solver': ['liblinear', 'lbfgs'], 
     'class_weight':['none', 'balanced'], 
     'multi_class': ['auto','ovr'], 
     'max_iter':[iter_],
     'tol':[epsilon_stop]},
    {'penalty': ['l2'], 
     'C': C,
     'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], 
     'class_weight':['none', 'balanced'], 
     'multi_class': ['auto','ovr'], 
     'max_iter':[iter_],
     'tol':[epsilon_stop]},
    {'penalty': ['none'], 
     'C': C,
     'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'], 
     'class_weight':['none', 'balanced'], 
     'multi_class': ['auto','ovr'], 
     'max_iter':[iter_],
     'tol':[epsilon_stop]},
]

# Создаем сетку поиска с использованием 5-кратной перекрестной проверки
# указываем модель (в нашем случае лог регрессия), гиперпараметры
model = LogisticRegression(random_state=RANDOM_SEED)

# Обучаем модель
gridsearch = GridSearchCV(model, hyperparameters, scoring='f1', n_jobs=-1, cv=5)
gridsearch.fit(X_train, y_train)
# Смотрим лучшие гиперпараметры
model_3 = gridsearch.best_estimator_

# Печатаем параметры
best_parameters = model_3.get_params()
print(f'Лучшие значения параметров:') 
for param_name in sorted(best_parameters.keys()):
        print(f'  {param_name} => {best_parameters[param_name]}')

# Предсказываем
y_pred_prob = model_3.predict_proba(X_test)[:,1]
y_pred = model_3.predict(X_test)
# Оценка качества модели
all_metrics(y_test, y_pred, y_pred_prob)
show_roc_curve(y_test, y_pred_prob)
show_confusion_matrix(y_test, y_pred)
submission_pred_prob = model_3.predict_proba(Test)[:,1]
submission_predict = model_3.predict(Test)

submission = pd.DataFrame({'client_id': id_test, 
                            'default': submission_pred_prob})
submission.to_csv('submission.csv', index=False)

submission
