import pandas as pd

import seaborn as sns

from datetime import datetime, timedelta

from matplotlib import pyplot as plt

import numpy as np

from numpy import percentile

from sklearn.preprocessing import LabelEncoder

from sklearn.feature_selection import f_classif

from sklearn.feature_selection import mutual_info_classif

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

import warnings



import os



warnings.filterwarnings("ignore")
RANDOM_SEED = 42

!pip freeze > requirements.txt

# этот блок используется на kaggle

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

PATH_to_file = '/kaggle/input/sf-dst-scoring/'



# этот блок используется на локальной машине

# PATH_to_file = ''
# Предварительная загрузка и просмотр



df = pd.read_csv(PATH_to_file+'train.csv')

df.head()
# Импортируем данные



train = pd.read_csv(PATH_to_file + 'train.csv')

test= pd.read_csv(PATH_to_file + 'test.csv')



display(train.info())

display(test.info())
display("train: shape" + str(train.shape), train.columns)

display("test: shape" +str(test.shape), test.columns)



# Видим, что в тестовой выборке нет данных признака "default", что логично.
display(train.isna().sum())

display(test.isna().sum())



# Мы видим, что и в тренировочной и в тестовом примере отсутствуют значения в признаке "education".
# Делаем пометку, где тренировочная выборка, где тестовая

train['Train'] = 1

test['Train'] = 0



#Заранее сохраняем значения client_id для предсказания тестовой выборки

id_test = test['client_id']



bank = train.append(test, sort=False).reset_index(drop=True) # объединяем датафреймы

print(f'bank.shape = {bank.shape}')

display(bank.isna().sum())
bank['education'].hist()

# В датафрейме только признаке "education" есть пустые значения. 

# Количетсво пустых значений существенно меньше общего количества строк в таблице, 

# в данном признаке сильно превалирует значение "SCH", мы меняем значение на "SCH".



bank['education'].fillna('SCH', inplace=True)



# Замечание: нам не нужен признак "client_id", мы удалим его

bank.drop('client_id', axis=1, inplace=True)

print(f'bank.shape = {bank.shape}')
# Мы видим, что в отличие от всех других признаков, признак "app_date" в формате, не пригодном для дульнейшей работы.



bank['app_date'] = pd.to_datetime(bank['app_date'], format='%d%b%Y')
# Нам необходимо разделить признаки на категории.

display(bank.head(3))

display(bank.nunique())



# На основании представленной информации разделим признаки следующим образом

num_features = ['age', 'decline_app_cnt', 'score_bki', 'bki_request_cnt', 'region_rating', 'income']

cat_features = ['education', 'home_address', 'work_address', 'sna', 'first_time']

bin_features = ['sex', 'car', 'car_type', 'good_work', 'foreign_passport']

date_features = ['app_date']



# Отдельно остается поле "default" - целевая переменная, которую необходимо бужет предсказать
# Так как мы не можем работать с признаком "app_date" напрямую, создадим новый признак:

# признак 'app_days_now', которые представляет собой разницу в днях от некоторого дня, 

# например от последнего указанного дня +30 дней (это первый весомый период по учету дефолта)



max_date = bank['app_date'].max().date() + timedelta(days=30)

bank['app_days_now'] = bank['app_date'].apply(lambda x: (max_date - x.date()).days)



# Добавим данные переменные в числовые признаки

num_features.extend(['app_days_now'])
# Рассмотрим корреляцию признаков



plt.figure(figsize=(15, 10))

sns.heatmap(bank[bank['Train']==1].corr().abs(), annot=True, fmt='.2f', cmap='PuBu')



# Выводы: так как максимальная корреляция чуть превосходит 0.7, то никакие признаки удалять не будем
# Рассмотрим как распределены числовые признаки



fig, ax = plt.subplots(len(num_features), 2, figsize=(10, 40))





#Изобразим графики значений и логарима значений



for x in range(len(num_features)):

    new_series_log = np.log(bank[bank['Train']==1][num_features[x]] + 1)

    

    ax[x, 0].hist(bank[bank['Train']==1][num_features[x]], rwidth=0.9, alpha=0.7)

    ax[x, 0].set_title(num_features[x])

    

    ax[x, 1].hist(new_series_log, rwidth=0.9, alpha=0.7)

    ax[x, 1].set_title('log of ' + num_features[x])

     
# На основании графиков можно сделать следующие преобразования:



bank['age'] = np.log(bank['age'] + 1)

bank['decline_app_cnt'] = np.log(bank['decline_app_cnt'] + 1)

bank['bki_request_cnt'] = np.log(bank['bki_request_cnt'] + 1)

bank['income'] = np.log(bank['income'] + 1)
# Изобразим графики распределения категориальных и бинарных признаков



col_list = cat_features + bin_features



plt.figure()

for column in col_list:

    plt.bar(bank[bank['Train']==1][column].unique(), bank[bank['Train']==1][column].value_counts(), width=0.5, alpha=0.7)

    plt.title(column)

    plt.show()

    

# На основании графиков можно сделать следующие вывод:

# В признаке "education" сильно превалирует значение "SCH"

# В признаке "work_address" превалирует значение "2"

# В признаке "sna" превалирует значение "4"

# В признаке "first_time" превалирует значение "1"

# В бинарных признаках только "sex" распределен достаточно равномерно
# Представим таблицу описания для оценки, что количество выбросов указано корректно

display(bank[bank['Train']==1][num_features].describe())



outlier_dic = {}



for column in num_features:

    perc25 = percentile(bank[column], 25)

    perc75 = percentile(bank[column], 75)

    iqr = perc75 - perc25

    low_range = perc25 - 1.5 * iqr

    upper_range = perc75 + 1.5 * iqr

    out_count = bank[bank['Train']==1][column].apply(lambda x: None if x < low_range or x > upper_range else x).isna().sum()

    outlier_dic[column] = [round(low_range, 2), round(upper_range, 2), out_count]



print('Результаты по выбросам:\n')

for key, val in outlier_dic.items():

    print(f'{key}: ниж.граница = {val[0]}, верх.граница = {val[1]}, кол-во выбросов = {val[2]}')



# По данным можно сделать следующие выводы:

# Признак "decline_app_cnt" имеет очень много выбросов, подумаем, нужно ли удалять:

#    Так как по распределению видно, что сильно преобладает занчение 0,

#    то удаление выбросов приведет к тому, что нужно будет удалить весь столбец.

# Признаки "score_bki", "bki_request_cnt", "income"  имеют мало выбросов, удалять не будем.

# Признак "region_rating" имеет очень много выбросов, подумаем, нужно ли удалять. 
# Далее нам надо будет рассмотреть значимость числовы и бинарных переменных. 

# Для этого сделаем преобразование бинарных переменных.



label_encoder = LabelEncoder()



for column in bin_features:

    bank[column] = label_encoder.fit_transform(bank[column])



imp_num = pd.Series(f_classif(bank[bank['Train']==1][bin_features + num_features], 

                              bank[bank['Train']==1]['default'])[0], index=bin_features + num_features)

imp_num.sort_values(inplace = True)

imp_num.plot(kind = 'barh')



# Мы видим, что самым значимым является признак "scoke_bki" (сильно важнее других), 

# далее идет "decline_app_cnt", "region_rating" и т.д., последним идет "sex".

# Учитывая значимость признаков "decline_app_cnt", "region_rating" и их данные по выбросам, выбросы удалять не будем.
# Теперь рассмотрим значимость категориальных переменных



label_encoder = LabelEncoder()

bank['education'] = label_encoder.fit_transform(bank['education'])



imp_cat = pd.Series(mutual_info_classif(bank[bank['Train']==1][cat_features], 

                                        bank[bank['Train']==1]['default'], discrete_features = True), index=cat_features)

imp_cat.sort_values(inplace = True)

imp_cat.plot(kind = 'barh')



# Мы видим, что самым значимым является признак "sna", а самым незначимым "work_address".
# Делим выборку обратно на тренировочную и тестовую

bank_train = bank[bank['Train']==1]

bank_test = bank[bank['Train']==0]



# Категориальные признаки преобразовываем

X_cat_train = pd.get_dummies(bank_train[cat_features], columns=cat_features).values

X_cat_test = pd.get_dummies(bank_test[cat_features], columns=cat_features).values



# Стандартизуем числовые признаки

X_num_train = StandardScaler().fit_transform(bank_train[num_features].values)

X_num_test = StandardScaler().fit_transform(bank_test[num_features].values)



# Бинарные признаки

X_bin_train = bank_train[bin_features].values

X_bin_test = bank_test[bin_features].values





# Объединяем данные

X = np.hstack([X_cat_train, X_num_train, X_bin_train])

Y = bank_train['default'].values

test_val = np.hstack([X_cat_test, X_num_test, X_bin_test])

# Далее будем использовать параметры регуляризации для улучшения модели

# Регуляризация. Подбор параметров



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=RANDOM_SEED, shuffle = True)



model = LogisticRegression(max_iter=500)

model.fit(X_train, Y_train)



# Зададим ограничения для параметра регуляризации

C = np.logspace(0, 4, 10)



penalty = ['l1', 'l2']

hyperparameters = dict(C=C, penalty=penalty)



clf = GridSearchCV(model, hyperparameters, cv=5, verbose=0)

best_model = clf.fit(X_train, Y_train)



print('Лучший penalty:', best_model.best_estimator_.get_params()['penalty'])

print('Лучшее C:', best_model.best_estimator_.get_params()['C'])
model = LogisticRegression(penalty='l2', C=1.0, max_iter=500)

model.fit(X_train, Y_train)

Y_pred = model.predict_proba(X_test)[:,1]



fpr, tpr, thresholds = roc_curve(Y_test, Y_pred)

roc_auc_val = roc_auc_score(Y_test, Y_pred)



plt.figure()

plt.plot([0, 1], label='Baseline', linestyle='--')

plt.plot(fpr, tpr, label = 'Regression')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.title(f'Logistic Regression ROC AUC = {roc_auc_val:.4f}')

plt.legend()

plt.show()
model_submis = LogisticRegression(penalty='l2', C=1.0, max_iter=1000)

model_submis.fit(X, Y)

prob_submis = model_submis.predict_proba(test_val)[:,1]



submission = pd.DataFrame({'client_id': id_test, 'default': prob_submis})

submission.to_csv('submission.csv', index=False)



submission