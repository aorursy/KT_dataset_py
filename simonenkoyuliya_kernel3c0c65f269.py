import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sb



from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LogisticRegressionCV

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier



from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score



%matplotlib inline
import sys



if not sys.warnoptions:

    import warnings

    warnings.simplefilter("ignore")
print(pd.__version__)

print(np.__version__)
%time

df = pd.read_csv('../input/wine-quality/wineQualityWhites.csv' )

print('Кол-во строк: ', df.shape[0])

print('Кол-во столбцов: ', df.shape[1])
#Удалили порядковый номер (1 столбец) так как он не несет смысловой нагрузки

df = df.drop(columns = ['Unnamed: 0'])
#Проверим типы данных, а также наличие пропущенных значений

df.info()
#Описательная статистика

df.describe()
#Oтобразим первые 5 строк

df.head()
plt.hist(df['fixed.acidity'], bins=np.arange(3, df['fixed.acidity'].max()+1, 1))

plt.xlabel('Fixed acidity')

plt.ylabel('Counts')

plt.title('Histogram of fixed acidity')
plt.hist(df['total.sulfur.dioxide'], bins=np.arange(0, df['total.sulfur.dioxide'].max()+11, 15))

plt.xlabel('Sulfur dioxide')

plt.ylabel('Counts')

plt.title('Histogram of total sulfur dioxide')
#Pairplot

sb.pairplot(df)
#Вывод
#Heatmap

# вводим новую переменную cor для матрицы корреляций признаков. функция corr считаем корреляцию между признаками.можно описать корреляции,   строим heatmap ,

#fig ax - Обозначили размер картинки

cor = df.corr()

fig, ax = plt.subplots(figsize = (10,10))

sb.heatmap(cor, annot = True, cmap="YlGnBu")
#Вывод
#Построим ящик с усами (boxplot)

sb.boxplot(x = df['quality'], y = df['alcohol'])
#Вывод: для вин более высокого качества характерно повышенное содержание алкоголя
# отбираем количественные признаки (выбрали только : - выбрали все строки, запятая, потом какие столбцы нужны - выбираем минус 1, iloc - функция помогает показать какой фрагмент данных нужно взять)

x = df.iloc[:,1:12]
y = df[['quality']]
x.head()
y
#Нормализация данных

scaler = MinMaxScaler(feature_range=(0, 1))

x = scaler.fit_transform(x)
#Разделим выборку на обучающую и тестовую в отношении 80 и 20 %

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
#Аргументы функции:

# - multi_class = 'multinomial' - уточняем, что в отличии от стандартной бинарной логистической регрессии в данном случае целевая переменная может принимать более 2 значений

# - cv = 3- столько раз мы разделим данные на обучающую и тестовую выборку, обучим и протестируем модель



best_clf_LR = LogisticRegressionCV(multi_class = 'multinomial', cv = 3)



#Обучаем модель

best_clf_LR.fit(x_train, y_train)
#Получаем прогнозируемые значения

y_pred_LR = best_clf_LR.predict(x_test)
#Получим точность модели

print(accuracy_score(y_pred_LR, y_test))
#Confusion matrix

cm_LR = confusion_matrix(y_pred_LR, y_test)

cm_LR
#Вывод
# Создаем случайный набор гиперпараметров для модели

param_grid = {

    'max_depth': [80, 90, 100, 110],

    'max_features': [10, 11],

    'min_samples_leaf': [10, 20, 50],

    'min_samples_split': [8, 10, 12],

    'n_estimators': [100, 200, 300, 1000]

}
#Обучаем модель

best_clf_RF = RandomForestClassifier()

grid_search_RF = GridSearchCV(best_clf_RF, param_grid = param_grid, cv = 3)

grid_search_RF.fit(x_train, y_train)
#Получим точность модели

np.mean(grid_search_RF.cv_results_['mean_test_score'])
#Вывод
# Создаем случайный набор гиперпараметров для модели

param_grid = {

    'max_depth': [80, 90, 100, 110],

    'n_estimators': [100, 200, 300, 1000]

}
#Обучаем модель

best_clf_XG = XGBClassifier()

grid_search_XG = GridSearchCV(best_clf_XG, param_grid = param_grid, cv = 3)

grid_search_XG.fit(x_train, y_train)
#Получим точность модели

np.mean(grid_search_XG.cv_results_['mean_test_score'])
#Вывод