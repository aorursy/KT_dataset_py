# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns 

%matplotlib inline



# Загружаем специальный удобный инструмент для разделения датасета:

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from pandas import Series

import pandas as pd

import numpy as np





import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.feature_selection import f_classif, mutual_info_classif

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression





from sklearn.metrics import confusion_matrix

from sklearn.metrics import auc, roc_auc_score, roc_curve

from sklearn.model_selection import GridSearchCV

from sklearn import metrics
RANDOM_SEED = 42
!pip freeze > requirements.txt
pd.options.mode.chained_assignment = None



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

PATH_to_file = '/kaggle/input/sf-dst-scoring/'
df_train = pd.read_csv(PATH_to_file+'train.csv')

df_test = pd.read_csv(PATH_to_file+'test.csv')

pd.set_option('display.max_columns', None)

print(df_train.shape)

display(df_train.sample(10))

print(df_test.shape)

display(df_test.sample(10))
print(df_train.info())
df_train.isna().sum()
df_test.isna().sum()
df_train.columns


def uniq_nan (df,c):

# Уникальные значения, пропущенные значения 

    column_type_numeric = False

    # тип колонки

    if df[c].dtype == 'O':

        print("Столбец:", c, "не является числовым")

    elif df[c].dtype == "int64" or df[c].dtype == "float64":

        column_type_numeric = True        

        print("Столбец:", c, "является числовым")

    else:

        print('Признак', c, 'имеет тип', df[c].dtype)

    #print("Столбец:", c)    

    print(pd.DataFrame(df[c].unique()))    

    print("Пропущено значений:", df[c].isna().sum())

    print(df[c].value_counts())

    

df_train.app_date = pd.to_datetime(df_train.app_date, format='%d%b%Y')

df_test.app_date = pd.to_datetime(df_test.app_date, format='%d%b%Y')

print(df_train.app_date.sample(5))

print(df_test.app_date.sample(5))
start_tr = df_train.app_date.min()

end_tr = df_train.app_date.max()

start_tr,end_tr
# Количество дней от старта заявки 

df_train['day_num'] = df_train.app_date - df_train.app_date.min()

df_train['day_num'] = df_train['day_num'].apply(lambda x: str(x).split()[0])

df_train['day_num'] = df_train['day_num'].astype(int)
# Количество дней от старта заявки 

df_test['day_num'] = df_test.app_date - df_test.app_date.min()

df_test['day_num'] = df_test['day_num'].apply(lambda x: str(x).split()[0])

df_test['day_num'] = df_test['day_num'].astype(int)
# Теперь столбец с датой можно удалить

df_test = df_test.drop(columns = ['app_date'])

df_train = df_train.drop(columns = ['app_date'])
# Работа с пропусками
uniq_nan(df_train, 'education')
#Решено заполнить значением "unknown"

df_train['education'] = df_train['education'].fillna('unknown')

df_test['education'] = df_test['education'].fillna('unknown')
def educ(x):

    if x == 'SCH':

        return 1

    elif x == 'GRD':

        return 2

    elif x == 'UGR':

        return 3

    elif x == 'PGR':

        return 4

    elif x == 'unknown':

        return 5

    elif x == 'ACD':

        return 6



df_train['education'] = df_train['education'].apply(educ)

df_test['education'] = df_test['education'].apply(educ)
df_train.sample(5)
# Сохрания ID клиентов из тестового набора для последующего формирования Submission

id_test = df_test.client_id
time_cols = ['app_date'] # временная переменная

bin_cols = ['sex', 'car', 'car_type', 'good_work', 'foreign_passport'] # бинарные переменные (default - целевой признак, не включаем в список)

cat_cols = ['education', 'region_rating', 'home_address', 'work_address', 'sna', 'first_time'] # категориальные переменные

num_cols = ['age','decline_app_cnt','score_bki','bki_request_cnt','income','day_num'] # числовые переменные
# Целевая переменная

sns.countplot(df_train.default)

plt.title('Histogram for default')
fig, axes = plt.subplots(2, 3, figsize=(25,12))

for col, i in zip(num_cols, range(7)):

    sns.distplot(df_train[col], kde=False, ax=axes.flat[i])
#Изобразим графики значений и логарима значений

fig, ax = plt.subplots(len(num_cols), 2, figsize=(10, 40))



for x in range(len(num_cols)):

    new_series_log = np.log(df_train[num_cols[x]] + 1)

    

    ax[x, 0].hist(df_train[num_cols[x]], rwidth=0.9, alpha=0.7)

    ax[x, 0].set_title(num_cols[x])

    

    ax[x, 1].hist(new_series_log, rwidth=0.9, alpha=0.7)

    ax[x, 1].set_title('log of ' + num_cols[x])
# На основании графиков можно сделать следующие преобразования:



df_train['age'] = np.log(df_train['age'] + 1)

df_train['decline_app_cnt'] = np.log(df_train['decline_app_cnt'] + 1)

df_train['bki_request_cnt'] = np.log(df_train['bki_request_cnt'] + 1)

df_train['income'] = np.log(df_train['income'] + 1)

df_train['day_num'] = np.log(df_train['day_num'] + 1)



df_test['age'] = np.log(df_test['age'] + 1)

df_test['decline_app_cnt'] = np.log(df_test['decline_app_cnt'] + 1)

df_test['bki_request_cnt'] = np.log(df_test['bki_request_cnt'] + 1)

df_test['income'] = np.log(df_test['income'] + 1)

df_test['day_num'] = np.log(df_test['day_num'] + 1)
imp_num = Series(f_classif(df_train[num_cols], df_train['default'])[0], index = num_cols)

imp_num.sort_values(inplace = True)

imp_num.plot(kind = 'barh')

plt.figure(figsize=(16,10))

sns.heatmap(df_train.corr().abs(), vmin=0, vmax=1, annot=True)


label_encoder = LabelEncoder()



for column in bin_cols:

    df_train[column] = label_encoder.fit_transform(df_train[column])

    df_test[column] = label_encoder.fit_transform(df_test[column])



    

# убедимся в преобразовании    

display(df_train.head())

display(df_test.head())
x_cat = OneHotEncoder(sparse = False).fit_transform(df_train[cat_cols].values)

y_cat = OneHotEncoder(sparse = False).fit_transform(df_test[cat_cols].values)



print(x_cat.shape)

print(y_cat.shape)
# Значимость признаков



imp_cat = Series(mutual_info_classif(df_train[bin_cols + cat_cols], df_train['default'],

                                     discrete_features =True), index = bin_cols + cat_cols)

imp_cat.sort_values(inplace = True)

imp_cat.plot(kind = 'barh')
# Стандартизация числовых переменных



X_num = StandardScaler().fit_transform(df_train[num_cols].values)

X_num_test = StandardScaler().fit_transform(df_test[num_cols].values)
X = np.hstack([X_num, df_train[bin_cols].values, x_cat])

Y = df_train['default'].values

Test = np.hstack([X_num_test, df_test[bin_cols].values, y_cat])



id_test = df_test.client_id
df_test.info()
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

X_test.shape
model = LogisticRegression()

model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

Y_predicted = model.predict(X_test)

print('accuracy_score:',accuracy_score(y_test,Y_predicted))

print('precision_score:',precision_score(y_test,Y_predicted))

print('recall_score:',recall_score(y_test,Y_predicted))

print('f1_score:',f1_score(y_test,Y_predicted))

print('MAE:', metrics.mean_absolute_error(y_test,Y_predicted))
probs = model.predict_proba(X_test)

probs = probs[:,1]





fpr, tpr, threshold = roc_curve(y_test, probs)

roc_auc = roc_auc_score(y_test, probs)



plt.figure()

plt.plot([0, 1], label='Baseline', linestyle='--')

plt.plot(fpr, tpr, label = 'Regression')

plt.title('Logistic Regression ROC AUC = %0.3f' % roc_auc)

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.legend(loc = 'lower right')

plt.show()
model = LogisticRegression(random_state=RANDOM_SEED)



iter_ = 50

epsilon_stop = 1e-3



param_grid = [

    {'penalty': ['l1'], 

     'solver': ['liblinear', 'saga'], 

     'class_weight':['none', 'balanced'], 

     'multi_class': ['auto','ovr'], 

     'max_iter':[iter_],

     'tol':[epsilon_stop]},

    

    {'penalty': ['none'], 

     'solver': ['newton-cg', 'saga'], 

     'class_weight':['none', 'balanced'], 

     'multi_class': ['auto','ovr'], 

     'max_iter':[iter_],

     'tol':[epsilon_stop]},

]

gridsearch = GridSearchCV(model, param_grid, scoring='f1', n_jobs=12, cv=5)

gridsearch.fit(X_train, y_train)

model = gridsearch.best_estimator_

##печатаем параметры

best_parameters = model.get_params()

for param_name in sorted(best_parameters.keys()):

        print('\t%s: %r' % (param_name, best_parameters[param_name]))

    ##печатаем метрики

preds = model.predict(X_test)

y_pred_prob = model.predict_proba(X_test)[:,1]

y_pred = model.predict(X_test)

print('Accuracy: %.4f' % accuracy_score(y_test, y_pred))

print('Precision: %.4f' % precision_score(y_test, y_pred))

print('Recall: %.4f' % recall_score(y_test, y_pred))

print('F1: %.4f' % f1_score(y_test, y_pred))
model = LogisticRegression(random_state=RANDOM_SEED, 

                           C=1, 

                           class_weight= 'balanced', 

                           dual= False, 

                           fit_intercept= True, 

                           intercept_scaling= 1, 

                           l1_ratio= None, 

                           multi_class= 'auto', 

                           n_jobs= 12, 

                           penalty= 'none', 

                           solver = 'saga', 

                           verbose= 0, 

                           warm_start= False)



model.fit(X_train, y_train)



y_pred_prob = model.predict_proba(X_test)[:,1]

y_pred = model.predict(X_test)

print('Accuracy: %.4f' % accuracy_score(y_test, y_pred))

print('Precision: %.4f' % precision_score(y_test, y_pred))

print('Recall: %.4f' % recall_score(y_test, y_pred))

print('F1: %.4f' % f1_score(y_test, y_pred))
len(y_pred_prob)

len(id_test)
probs = model.predict_proba(X_test)

probs = probs[:,1]





fpr, tpr, threshold = roc_curve(y_test, probs)

roc_auc = roc_auc_score(y_test, probs)



plt.figure()

plt.plot([0, 1], label='Baseline', linestyle='--')

plt.plot(fpr, tpr, label = 'Regression')

plt.title('Logistic Regression ROC AUC = %0.3f' % roc_auc)

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.legend(loc = 'lower right')

plt.show()


submission_pred_prob = model.predict_proba(Test)[:,1]

submission_predict = model.predict(Test)



submission = pd.DataFrame({'client_id': id_test, 

                            'default': submission_pred_prob})

submission.to_csv('submission.csv', index=False)



submission