import numpy as np 

import pandas as pd 

from pandas import Series



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.feature_selection import f_classif, mutual_info_classif

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, RobustScaler



from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LogisticRegression





from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from sklearn.metrics import auc, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
RANDOM_SEED = 42

!pip freeze > requirements.txt



path_to_data = '/kaggle/input/sf-dst-scoring/'



# Импорт и просмотр данных

df_train = pd.read_csv(path_to_data+'train.csv')

df_test = pd.read_csv(path_to_data+'test.csv')

#sample_submission = pd.read_csv(path_to_data+'/sample_submission.csv')



print('df_train: ', df_train.shape)

display(df_train.head())

print('df_test: ', df_test.shape)

display(df_test.head())
df_train.info()
df_train.isnull().sum()
df_test.info()
df_test.isnull().sum()
# Объединяем данные для обработки



df_train['Train'] = 1 # помечаем трейн

df_test['Train'] = 0 # помечаем тест



df = df_train.append(df_test, sort=False).reset_index(drop=True)
df.shape
df.info()
from datetime import datetime

current_date = pd.to_datetime('21/10/2020')
df.app_date = pd.to_datetime(df.app_date, format='%d%b%Y')
# Признак последовательности дней 

df['days'] = (df.app_date - df.app_date.min()).dt.days.astype('int')
plt.subplots(figsize=(12, 4))

sns.barplot(data=df[df['Train']==1], x=df.loc[df['Train']==1,'app_date'].dt.month, y='default', palette="rainbow")
# Введем новый признак - месяц подачи заявки



df['app_date_month'] = df.app_date.dt.month
sns.scatterplot(x='client_id',y='days',data=df)
# объединим все полученные признаки по категориям

date_cols = ['app_date']

bin_cols = ['sex', 'car', 'car_type', 'good_work', 'foreign_passport'] # default - целевая переменная

cat_cols = ['education', 'region_rating', 'home_address', 'work_address', 'sna', 'first_time', 'app_date_month']

num_cols = ['age','decline_app_cnt','score_bki','bki_request_cnt','income', 'days'] # client_id исключили из списка


for i in num_cols:

    f = plt.figure(figsize=(10, 3))

    gs = f.add_gridspec(1, 2)

    with sns.axes_style("white"):

        ax = f.add_subplot(gs[0, 0])

        sns.distplot(df[i], kde = False, rug=False, color='b')



    with sns.axes_style("white"):

        ax = f.add_subplot(gs[0, 1])

        sns.boxplot(data=df[i], palette='rainbow')



    f.tight_layout()

# пробуем логарифмировать



for i in num_cols:

    df[i] = np.log(df[i] + 4)

    f = plt.figure(figsize=(10, 3))

    gs = f.add_gridspec(1, 2)

    with sns.axes_style("white"):

        ax = f.add_subplot(gs[0, 0])

        sns.distplot(df[i], kde = False, rug=False, color='b')



    with sns.axes_style("white"):

        ax = f.add_subplot(gs[0, 1])

        sns.boxplot(data=df[i], palette='rainbow')



    f.tight_layout()

# Для облегчения анализа избавимся от пропусков и закодируем данные

# Заполнение популярным SCH.



df['education'] = df['education'].fillna('SCH')
# Для анализа закодируем 'education' по категориям

df['education'] = df.apply(lambda x: 1 if x['education']=='SCH' else 2 if x['education']=='GRD' 

                           else 3 if x['education']=='UGR' else 4 if x['education']=='PGR' 

                           else 5, axis=1)
for i in cat_cols:

    f = plt.figure(figsize=(15, 3))

    gs = f.add_gridspec(1, 4)

    with sns.axes_style("white"):

        ax = f.add_subplot(gs[0, 0])

        sns.barplot(x=i, y='default', data=df[df['Train'] ==1], estimator=len)

        

    with sns.axes_style("white"):

        ax = f.add_subplot(gs[0, 1])

        sns.barplot(x='default', y=i, data=df[df['Train'] ==1], estimator=len)



    with sns.axes_style("white"):

        ax = f.add_subplot(gs[0, 2])

        sns.boxplot(x='default', y=i, data=df[df['Train'] ==1], palette='rainbow')

        

    with sns.axes_style("white"):

        ax = f.add_subplot(gs[0, 3])

        sns.boxplot(data=df[i], palette='rainbow')



    f.tight_layout()
# Закодируем бинарные признаки

label_encoder = LabelEncoder()



for column in bin_cols:

    df[column] = label_encoder.fit_transform(df[column])
df_num = df[num_cols]

sns.heatmap(df_num.corr(), annot = True)
train_df = df[df['Train']==1]

imp_num = pd.Series(f_classif(train_df[num_cols], train_df['default'])[0], index = num_cols)

imp_num.sort_values(inplace = True)

imp_num.plot(kind = 'barh', title='Значимость числовых переменных')
imp_cat = Series(mutual_info_classif(train_df[bin_cols + cat_cols], train_df['default'],

                                     discrete_features =True), index = bin_cols + cat_cols)

imp_cat.sort_values(inplace = True)

imp_cat.plot(kind = 'barh', title = 'Значимость бинарных и категориальных признаков')
df = pd.get_dummies(df, prefix=cat_cols, columns=cat_cols)
# Поскольку в данных выбросы, воспользуемся RobustScaler

scaler = RobustScaler()



df[num_cols] = scaler.fit_transform(df[num_cols].values)
df.drop(['app_date', 'client_id'], axis=1, inplace=True)
# Возвращаем первоначальное разбиение на train и test



train_data = df[df['Train'] == 1].drop(['Train'], axis=1)

test_data = df[df['Train'] == 0].drop(['Train'], axis=1)



y = train_data.default.values           

X = train_data.drop(['default'], axis=1).values
# Из train выделяем данные на валидацию 

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

test_data.shape, train_data.shape, X.shape, X_train.shape, X_valid.shape
model = LogisticRegression(random_state=RANDOM_SEED, max_iter = 1000)



model.fit(X_train, y_train)



y_pred_prob = model.predict_proba(X_valid)[:,1]

y_pred = model.predict(X_valid)
metrics = ['accuracy', 'precision', 'recall', 'f1_score']

value = [accuracy_score(y_valid,y_pred), precision_score(y_valid,y_pred), recall_score(y_valid,y_pred), f1_score(y_valid,y_pred)]

first_metrics_df = pd.DataFrame({'Метрика': metrics, 'Значение': value}, columns=['Метрика', 'Значение'])
fpr, tpr, threshold = roc_curve(y_valid, y_pred_prob)

roc_auc = roc_auc_score(y_valid, y_pred_prob)



plt.figure()

plt.plot([0, 1], label='Baseline', linestyle='--')

plt.plot(fpr, tpr, label = 'Regression')

plt.title('Logistic Regression ROC AUC = %0.3f' % roc_auc)

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.legend(loc = 'lower right')

plt.show()
tn, fp, fn, tp = confusion_matrix(y_valid, y_pred).ravel()

print(tp, fp) 

print(fn, tn)
precision, recall, thresholds = precision_recall_curve(y_valid, y_pred_prob)
plt.figure(figsize=(8, 6))

prc_area = auc(recall, precision)

plt.plot(recall, precision, lw=3, label='площадь под PR кривой = %0.3f)' % prc_area)

    

plt.xlim([-.05, 1.0])

plt.ylim([-.05, 1.05])

plt.xlabel('Recall')

plt.ylabel('Precision')

plt.title('Precision-Recall Curve')

plt.legend(loc="upper right")

plt.show()
# Добавим метрики в наш датасет метрик для первой модели



add_metrics = pd.DataFrame({'Метрика': ['ROC_AUC', 'PRC_AUC'], 'Значение': [roc_auc, prc_area]}, columns=['Метрика', 'Значение'])



first_metrics_df = first_metrics_df.append(add_metrics, ignore_index=True)
model = LogisticRegression(random_state=RANDOM_SEED)



iter_max = 100



param_grid = [

    {'penalty': ['l1'], 

     'solver': ['liblinear', 'lbfgs'], 

     'class_weight':['none', 'balanced'], 

     'multi_class': ['auto','ovr'], 

     'max_iter':[iter_max]},

    {'penalty': ['l2'], 

     'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], 

     'class_weight':['none', 'balanced'], 

     'multi_class': ['auto','ovr'], 

     'max_iter':[iter_max]},

    {'penalty': ['none'], 

     'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'], 

     'class_weight':['none', 'balanced'], 

     'multi_class': ['auto','ovr'], 

     'max_iter':[iter_max]},

]



gridsearch = GridSearchCV(model, param_grid, scoring='f1', n_jobs=-1)

gridsearch.fit(X_train, y_train)

model = gridsearch.best_estimator_

print(model)
# Обучим модель на данных и проверим confusion_matrix



model.fit(X_train, y_train)



y_pred_prob = model.predict_proba(X_valid)[:,1]

y_pred = model.predict(X_valid)
# матрица ошибок

tn, fp, fn, tp = confusion_matrix(y_valid, y_pred).ravel()

print(tp, fp) 

print(fn, tn)
print('Accuracy: %.4f' % accuracy_score(y_valid, y_pred))

print('Precision: %.4f' % precision_score(y_valid, y_pred))

print('Recall: %.4f' % recall_score(y_valid, y_pred))

print('F1: %.4f' % f1_score(y_valid, y_pred))



precision, recall, thresholds = precision_recall_curve(y_valid, y_pred_prob)

print('ROC_AUC = ', round(roc_auc_score(y_valid, y_pred_prob), 4))

print('PRC_AUC = ', round(auc(recall, precision), 4))
# Метрики первой модели

first_metrics_df
train_data = df[df['Train'] == 1].drop(['Train'], axis=1)

test_data = df[df['Train'] == 0].drop(['Train'], axis=1)
X_train=train_data.drop(['default'], axis=1)

y_train = train_data.default.values

X_test = test_data.drop(['default'], axis=1)
test_data.shape, train_data.shape, X_train.shape, y_train.shape, X_test.shape
predict_submission = model.predict_proba(X_test)[:,1]



submit = pd.DataFrame(df_test.client_id)

submit['default']=predict_submission

submit.to_csv('submission.csv', index=False)

display(submit.head(10))