from pandas import Series
import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc, roc_auc_score, roc_curve

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegressionCV

from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer

import os

# Загружаем специальный удобный инструмент для разделения датасета:

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# зафиксируем версию пакетов, чтобы эксперименты были воспроизводимы:
!pip freeze > requirements.txt
data = pd.read_csv(
    '/Users/dariamishina/Documents/Skillfactory/skillfactory_rds/module_4/train.csv')
true_test = pd.read_csv(
    '/Users/dariamishina/Documents/Skillfactory/skillfactory_rds/module_4/test.csv')
sample_submission = pd.read_csv(
    '/Users/dariamishina/Documents/Skillfactory/skillfactory_rds/module_4/sample_submission.csv')
# посмотрим на типы данных и количество пропусков
data.info()
fig, ax = plt.subplots(figsize=(20, 12))
sns_heatmap = sns.heatmap(
    data.isnull(), yticklabels=False, cbar=False, cmap='viridis')
data.sample(5)
data.default.hist();
data.education.value_counts(dropna=False, normalize=True)
data['education'] = data['education'].fillna(data['education'].mode()[0])
# бинарные переменные
bin_cols = ['sex', 'car', 'car_type', 'good_work', 'foreign_passport']

# категориальные переменные
cat_cols = ['education', 'work_address', 'home_address', 'sna', 'first_time', 'region_rating']

# числовые переменные
num_cols = ['age', 'decline_app_cnt', 'bki_request_cnt', 'income', 'score_bki']

for i in num_cols:
    plt.figure()
    sns.distplot(data[i][data[i] > 0].dropna(), kde=False, rug=False)
    plt.title(i)
    plt.show()
data.score_bki.hist();
for i in num_cols:
    plt.figure()
    sns.distplot(data[i], kde=False, rug=False)
    plt.title(i)
    plt.show()
# числовые переменные, которые будем логарифмировать, все кроме 'score_bki', у ктр и так нормальное распределение
log_num_cols = ['age', 'decline_app_cnt', 'bki_request_cnt', 'income']

for i in log_num_cols:
    #зачем к логарифму добавляем 1 - чтобы если в df[i] был 0, то натуральный логарифм нуля - минус бесконечность
    data[i] = np.log(data[i]+1)
    plt.figure()
    sns.distplot(data[i][data[i] > 0].dropna(), kde = False, rug=False, color='b')
    plt.title(i)
    plt.show()
# числовые переменные
num_cols = ['age', 'decline_app_cnt', 'bki_request_cnt', 'income', 'score_bki']

for i in num_cols:
    plt.figure()
    sns.distplot(data[i], kde=False, rug=False)
    plt.title(i)
    plt.show()
data['decline_app_cnt'].value_counts()
data['bki_request_cnt'].value_counts()
data['age'].value_counts()
data['income'].value_counts()
bc = PowerTransformer(method='box-cox')
df1 = data['income'].values.reshape(1,-1)
bc.fit(df1)
bc.transform(df1)
bc = PowerTransformer(method='box-cox')
df4 = data['age'].values.reshape(1,-1)
bc.fit(df4)
bc.transform(df4)
yj = PowerTransformer(method='yeo-johnson')
df5 = data['age'].values.reshape(1,-1)
yj.fit(df5)
yj.transform(df5)
e = yj.transform(df5).reshape((73799, 1))
f = pd.DataFrame(e)
f.hist()
yj = PowerTransformer(method='yeo-johnson')
df6 = data['decline_app_cnt'].values.reshape(1,-1)
yj.fit(df6)
yj.transform(df6)
g = yj.transform(df6).reshape((73799, 1))
h = pd.DataFrame(g)
h.hist()
yj = PowerTransformer(method='yeo-johnson')
df7 = data['bki_request_cnt'].values.reshape(1,-1)
yj.fit(df7)
yj.transform(df7)
i = yj.transform(df7).reshape((73799, 1))
j = pd.DataFrame(i)
j.hist()
yj = PowerTransformer(method='yeo-johnson')
df2 = data['income'].values.reshape(1,-1)
yj.fit(df2)
yj.transform(df2)
a = yj.transform(df2).reshape((73799, 1))
b = pd.DataFrame(a)
b.hist()
qt = QuantileTransformer(output_distribution='normal', random_state=42)
df8 = data['age'].values.reshape(1,-1)
qt.fit(df8)
qt.transform(df8)
k = qt.transform(df8).reshape((73799, 1))
l = pd.DataFrame(k)
l.hist()
qt = QuantileTransformer(output_distribution='normal', random_state=42)
df9 = data['decline_app_cnt'].values.reshape(1,-1)
qt.fit(df9)
qt.transform(df9)
m = qt.transform(df9).reshape((73799, 1))
n = pd.DataFrame(m)
n.hist()
qt = QuantileTransformer(output_distribution='normal', random_state=42)
df10 = data['bki_request_cnt'].values.reshape(1,-1)
qt.fit(df10)
qt.transform(df10)
o = qt.transform(df10).reshape((73799, 1))
p = pd.DataFrame(o)
p.hist()
qt = QuantileTransformer(output_distribution='normal', random_state=42)
df3 = data['income'].values.reshape(1,-1)
qt.fit(df3)
qt.transform(df3)
c = qt.transform(df3).reshape((73799, 1))
d = pd.DataFrame(c)
d.hist()
sns.boxplot(data.age, color='yellow');
sns.boxplot(data.decline_app_cnt, color='yellow');
def outliers_iqr(ys):
    #находим квартили
    quartile_1, quartile_3 = np.percentile(ys, [25, 75])
    #находим межквартильное расстояние
    iqr = quartile_3 - quartile_1
    #нижняя граница коробки
    lower_bound = quartile_1 - (iqr * 1.5)
    #верхняя граница коробки
    upper_bound = quartile_3 + (iqr * 1.5)
    #возращаем только те значения и их индексы, ктр больше upper_bound и меньше lower_bound
    return ys[((ys > upper_bound) | (ys < lower_bound))]
#применяем функцию к колонке data.decline_app_cnt
out = outliers_iqr(data.decline_app_cnt)
out
len(out)/len(data)
sns.boxplot(data.bki_request_cnt, color='yellow');
#применяем функцию к колонке data.bki_request_cnt
out1 = outliers_iqr(data.bki_request_cnt)
out1
len(out1)/len(data)
#находим квартили
quartile_1, quartile_3 = np.percentile(data.bki_request_cnt, [25, 75])
#находим межквартильное расстояние
iqr = quartile_3 - quartile_1
#нижняя граница коробки
lower_bound = quartile_1 - (iqr * 1.5)
#верхняя граница коробки
upper_bound = quartile_3 + (iqr * 1.5)

#Этот способ позволил нам отобрать экстремально низкие и экстремально высокие оценки. Отфильтруем данные:
data = data.loc[data.bki_request_cnt.between(lower_bound, upper_bound)]
sns.boxplot(data.bki_request_cnt, color='yellow');
sns.boxplot(data.income, color='yellow');
#применяем функцию к колонке data.income
out2 = outliers_iqr(data.income)
out2
len(out2)/len(data)
#находим квартили
quartile_1, quartile_3 = np.percentile(data.income, [25, 75])
#находим межквартильное расстояние
iqr = quartile_3 - quartile_1
#нижняя граница коробки
lower_bound = quartile_1 - (iqr * 1.5)
#верхняя граница коробки
upper_bound = quartile_3 + (iqr * 1.5)

#Этот способ позволил нам отобрать экстремально низкие и экстремально высокие оценки. Отфильтруем данные:
data = data.loc[data.income.between(lower_bound, upper_bound)]
sns.boxplot(data.income, color='yellow');
#применяем функцию к колонке data.income 
out3 = outliers_iqr(data.income)
out3
len(out3)/len(data)
#находим квартили
quartile_1, quartile_3 = np.percentile(data.income, [25, 75])
#находим межквартильное расстояние
iqr = quartile_3 - quartile_1
#нижняя граница коробки
lower_bound = quartile_1 - (iqr * 1.5)
#верхняя граница коробки
upper_bound = quartile_3 + (iqr * 1.5)

#Этот способ позволил нам отобрать экстремально низкие и экстремально высокие оценки. Отфильтруем данные:
data = data.loc[data.income.between(lower_bound, upper_bound)]
sns.boxplot(data.income, color='yellow');
sns.boxplot(data.score_bki, color='yellow');
#применяем функцию к колонке data.score_bki
out4 = outliers_iqr(data.score_bki)
out4
len(out4)/len(data)
#находим квартили
quartile_1, quartile_3 = np.percentile(data.score_bki, [25, 75])
#находим межквартильное расстояние
iqr = quartile_3 - quartile_1
#нижняя граница коробки
lower_bound = quartile_1 - (iqr * 1.5)
#верхняя граница коробки
upper_bound = quartile_3 + (iqr * 1.5)

#Этот способ позволил нам отобрать экстремально низкие и экстремально высокие оценки. Отфильтруем данные:
data = data.loc[data.score_bki.between(lower_bound, upper_bound)]
sns.boxplot(data.score_bki, color='yellow');
#применяем функцию к колонке data.score_bki еще раз
out5 = outliers_iqr(data.score_bki)
out5
len(out5)/len(data)
#находим квартили
quartile_1, quartile_3 = np.percentile(data.score_bki, [25, 75])
#находим межквартильное расстояние
iqr = quartile_3 - quartile_1
#нижняя граница коробки
lower_bound = quartile_1 - (iqr * 1.5)
#верхняя граница коробки
upper_bound = quartile_3 + (iqr * 1.5)

#Этот способ позволил нам отобрать экстремально низкие и экстремально высокие оценки. Отфильтруем данные:
data = data.loc[data.score_bki.between(lower_bound, upper_bound)]
sns.boxplot(data.score_bki, color='yellow')
len(data)
sns.heatmap(data[num_cols].corr().abs(), vmin=0, vmax=1)
#annot=True этот параметр отвечает за вывод значения на карту
#sns.heatmap(stud_for_corr.corr(), square=True, annot=True, fmt=".3f", linewidths=0.1, cmap="RdBu")
data[num_cols].corr()
#df[num_cols] - это признаки
#df['default'] - это таргет 
#index = num_cols - а так присваиваются названия признакам, но это именно для сериз
#указываем индекс [0], чтобы выводились значения
imp_num = Series(f_classif(data[num_cols], data['default'])[0], index = num_cols)
#сортировка по убыванию
imp_num.sort_values(inplace = True)
#barh - для переворачивания гистограммы
imp_num.plot(kind = 'barh')
# Для бинарных признаков мы будем использовать LabelEncoder

label_encoder = LabelEncoder()

for column in bin_cols:
    data[column] = label_encoder.fit_transform(data[column])
    
# убедимся в преобразовании    
data.head()
cat_cols = ['education', 'work_address', 'home_address', 'sna', 'first_time', 'region_rating']
# Для категориальных признаков мы будем использовать OneHotEncoder
#sparse bool, default=True
#Will return sparse matrix if set True else will return an array- в нашем случае возвращает массив
X_cat = OneHotEncoder(sparse = False).fit_transform(data[cat_cols].values)
#переводим признак education в численный формат
data['education'] = label_encoder.fit_transform(data['education'])
#df[bin_cols + cat_cols] - это признаки
#df['default'] - это таргет
#discrete_features =True If bool, then determines whether to consider all features discrete or continuous.
#index = bin_cols + cat_cols - это чтобы выводились названия колонок
imp_cat = Series(mutual_info_classif(data[bin_cols + cat_cols], data['default'],
                                     discrete_features =True), index = bin_cols + cat_cols)
imp_cat.sort_values(inplace = True)
imp_cat.plot(kind = 'barh')
data[num_cols].head()
# числовые переменные
num_cols = ['decline_app_cnt', 'bki_request_cnt', 'score_bki']
scaler = RobustScaler() 
X_num = scaler.fit_transform(data[num_cols].values)
bin_cols = ['foreign_passport']
# Объединяем
#Функция hstack() соединяет массивы по горизонтали. По своей сути такое соединение эквивалентно соединению массивов 
#вдоль второй (или первой, если считать от 0) оси. Одномерные массивы просто соединяются.
X = np.hstack([X_num, data[bin_cols].values, X_cat])
Y = data['default'].values
# Воспользуемся специальной функцие train_test_split для разбивки тестовых данных
# выделим 20% данных на валидацию (параметр test_size)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

#predict_proba - Probability estimates.
probs = model.predict_proba(X_test)
probs = probs[:,1]


fpr, tpr, threshold = roc_curve(y_test, probs)
roc_auc = roc_auc_score(y_test, probs)

plt.figure()
# [0, 1] чтобы график был в пространстве от 0 до 1, по типу диагонали
plt.plot([0, 1], label='Baseline', linestyle='--')
plt.plot(fpr, tpr, label = 'Regression')
plt.title('Logistic Regression ROC AUC = %0.3f' % roc_auc)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc = 'lower right')
plt.show()
# Добавим типы регуляризации
penalty = ['l1', 'l2', 'elasticnet']

# Зададим ограничения для параметра регуляризации
C = np.logspace(0, 4, 10)


# Зададим значения для солвера - не получилось сделать перебор, комп считал 4 часа и умер
solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']

# Зададим значения до кол-ва максимальных итераций
max_iter = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

class_weight = ['balanced', None]

multi_class = ['auto', 'ovr', 'multinomial']

# Создадим гиперпараметры
hyperparameters = dict(C=C, penalty=penalty, solver=solver, max_iter=max_iter, class_weight=class_weight, multi_class=multi_class)
#hyperparameters = dict(C=C, penalty=penalty, max_iter=max_iter)
#hyperparameters = dict(C=C, penalty=penalty, solver=solver)

model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Создаем сетку поиска с использованием 5-кратной перекрестной проверки
#указываем модель (в нашем случае лог регрессия), гиперпараметры
clf = GridSearchCV(model, hyperparameters, cv=5, verbose=0)

best_model = clf.fit(X_train, y_train)

# View best hyperparameters
print('Лучшее Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Лучшее C:', best_model.best_estimator_.get_params()['C'])
print('Лучшее solver:', best_model.best_estimator_.get_params()['solver'])
print('Лучшее max_iter:', best_model.best_estimator_.get_params()['max_iter'])
print('Лучшее class_weight:', best_model.best_estimator_.get_params()['class_weight'])
print('Лучшее multi_class:', best_model.best_estimator_.get_params()['multi_class'])
model = LogisticRegression(penalty='l2', C=7.742636826811269, solver ='saga', class_weight='balanced', max_iter=100, random_state=42)
model.fit(X_train, y_train)

#predict_proba - Probability estimates.
probs = model.predict_proba(X_test)
probs = probs[:,1]


fpr, tpr, threshold = roc_curve(y_test, probs)
roc_auc = roc_auc_score(y_test, probs)

plt.figure()
# [0, 1] чтобы график был в пространстве от 0 до 1, по типу диагонали
plt.plot([0, 1], label='Baseline', linestyle='--')
plt.plot(fpr, tpr, label = 'Regression')
plt.title('Logistic Regression ROC AUC = %0.3f' % roc_auc)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc = 'lower right')
plt.show()
clf = LogisticRegressionCV(cv=5, random_state=42).fit(X_train, y_train)
#predict_proba - Probability estimates.
probs = clf.predict_proba(X_test)
probs = probs[:,1]


fpr, tpr, threshold = roc_curve(y_test, probs)
roc_auc = roc_auc_score(y_test, probs)

plt.figure()
# [0, 1] чтобы график был в пространстве от 0 до 1, по типу диагонали
plt.plot([0, 1], label='Baseline', linestyle='--')
plt.plot(fpr, tpr, label = 'Regression')
plt.title('Logistic Regression ROC AUC = %0.3f' % roc_auc)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc = 'lower right')
plt.show()
true_test.drop(['app_date'], axis = 1, inplace=True)
#data = pd.read_csv(
    #'C:/Users/DariaMishina/skillfactory_rds/module_4/train.csv')
true_test = pd.read_csv(
    'C:/Users/DariaMishina/skillfactory_rds/module_4/test.csv')
#sample_submission = pd.read_csv(
    #'C:/Users/DariaMishina/skillfactory_rds/module_4/sample_submission.csv')
true_test.sample(5)
true_test.info()
true_test['education'] = true_test['education'].fillna(true_test['education'].mode()[0])
X_num_test = scaler.transform(true_test[num_cols].values)
X_cat_test = OneHotEncoder(sparse = False).fit_transform(true_test[cat_cols].values)
# Для бинарных признаков мы будем использовать LabelEncoder

label_encoder = LabelEncoder()

for column in bin_cols:
    true_test[column] = label_encoder.fit_transform(true_test[column])
    
# убедимся в преобразовании    
true_test.head()
upd_true_test = np.hstack([X_num_test, true_test[bin_cols].values, X_cat_test])
sample_submission
predict_submission = model.predict(upd_true_test)
#для LogisticRegressionCV
#predict_submission = clf.predict(upd_true_test)
true_test['default'] = predict_submission
submission = true_test[['client_id', 'default']]
submission.to_csv('submission.csv', index=False)
submission.head(10)
