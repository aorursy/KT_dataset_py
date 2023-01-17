# ШАГ 1
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# ШАГ 2
# Импорт библиотек для анализа и отображения его результатов
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import missingno as msno
import plotly.express as px

sns.set(style="darkgrid")

# Импортируем датасет
df = pd.read_csv("/kaggle/input/cannabis-strains-features/Cannabis_Strains_Features.csv")

# Делаем копию начальной таблицы. Понадобится в дальнейшем
df_copy = df.copy(deep=True)

# Выводим последнюю строку файла, смотрим количество строк
df.tail(1)
# ШАГ 3
# Посмотрим сводную информацию о наших данных
df.info()
# ШАГ 4
# Убеждаемся, что нет повторяющихся штаммов
len(df['Strain'].unique())
# Просто тестовый поиск
df[df['Flavor'].str.contains("Earthy")]
# Просто тестовый поиск
df[df['Effects'].str.contains("Relaxed")]
# Просто тестовый поиск
df[df['Flavor'].str.contains("Earthy") & df['Effects'].str.contains("Relaxed")]
# Разделяем колонку "Flavors" по запятой, посмотреть функционал
df_copy['Flavor'].apply(lambda x: pd.Series(str(x).split(",")))
# Разделяем колонку "Effects" по запятой, посмотреть функционал
df_copy['Effects'].apply(lambda x: pd.Series(str(x).split(",")))
# ШАГ 5
# Разделяем колонку "Effects" на 5 колонок (для описания всех возможных эффектов по отдельности)
df_copy[['First_Effect','Second_Effect','Third_Effect','Fourth_Effect','Fifth_Effect']] = df.Effects.str.split(",",expand=True)

# Разделяем колонку "Flavor" на 4 колонки (для описания всех возможных эффектов по отдельности)
df_copy[['First_Flavor','Second_Flavor','Third_Flavor','Fourth_Flavor']] = df.Flavor.str.split(",",expand=True)
# ШАГ 6
# Удаляем лишние колонки (оставляем все эффекты и запахи)
to_drop = ['Type', 'Rating', 'Effects', 'Flavor', 'Description','Fourth_Flavor']
df_copy.drop(columns=to_drop, inplace = True)

# Дублируем колонку "Strain", т.к. мы все стринговые значения будем преобразовывать в числовые. Оставим возможность посмореть стригновое значение по новой колонке "Strain_string"
df_copy['Strain_string'] = df_copy['Strain'].values

df_copy.info()
# Удаляем лишние колонки (оставляем 2 атрибута)
to_drop = ['Type', 'Rating', 'Effects', 'Flavor', 'Description','Second_Effect','Third_Effect','Fourth_Effect','Fifth_Effect','Second_Flavor','Third_Flavor','Fourth_Flavor']
df_copy.drop(columns=to_drop, inplace = True)

# Дублируем колонку "Strain", т.к. мы все стринговые значения будем преобразовывать в числовые. Оставим возможность посмореть стригновое значение по новой колонке "Strain_string"
df_copy['Strain_string'] = df_copy['Strain'].values

df_copy.info()
# ШАГ 7
# Смотрим количество пустых записей по столбам в процентном соотношении
missing_values = df_copy.isnull()
missing_percentage = (missing_values.sum()*100)/df.shape[0]
missing_percentage
# ШАГ 8
# Визуализируем наличие пустых данных с помощью либы missingno
msno.matrix(df_copy)
# ШАГ 9
# Удаляем строки с нулевыми параметрами
df_copy.dropna(inplace=True)

df_copy.info()
# ШАГ 10
# Cмотрим уникальные эффекты
column_values_effects = df_copy[['First_Effect']].values.ravel()
unique_values_effects =  pd.unique(column_values_effects)
dfEffects = pd.DataFrame(list(unique_values_effects), columns = ['Effects'] )
dfEffects
# ШАГ 11
# Смотрим уникальные запахи
column_values_flavor = df_copy[['First_Flavor']].values.ravel()
unique_values_flavor =  pd.unique(column_values_flavor)
dfFlavor = pd.DataFrame(list(unique_values_flavor), columns = ['Flavor'] )
dfFlavor
# ШАГ 12
# Заменяем невалидные значения на их аналоги
df_copy = df_copy.replace(['\nRelaxed'],'Relaxed')
# Подход был заимпрувлен, можно игнориовать
# Заменяем невалидные значения на их аналоги
df_copy = df_copy.replace(['\nRelaxed'],'Relaxed')
df_copy = df_copy.replace(['Happy\n'],'Happy')
df_copy = df_copy.replace(['Sleepy\n'],'Sleepy')
df_copy = df_copy.replace(['Uplifted\n'],'Uplifted')
df_copy = df_copy.replace(['Hungry\n'],'Hungry')
df_copy = df_copy.replace(['Energentic\n'],'Energentic')
df_copy = df_copy.replace(['Citrus\n'],'Citrus')
df_copy = df_copy.replace(['Vanilla\n'],'Vanilla')
df_copy = df_copy.replace(['Bubblegum\n'],'Bubblegum')
df_copy = df_copy.replace(['Diesel\n'],'Diesel')
df_copy = df_copy.replace(['Earthy\n'],'Earthy')
df_copy = df_copy.replace(['Berry\n'],'Berry')
df_copy = df_copy.replace(['Sweet\n'],'Sweet')
df_copy = df_copy.replace(['Euphoric\n'],'Euphoric')
# ШАГ 13
# Выдвигаем гипотезу, что первый упомянутый запах является наиболее часто распознаваемым для штамма.
# Создадим гистограмму, которая бы отображала кол-во штаммов у которых в первую очередь распознается тот или иной запах.
fig = px.histogram(df_copy, x='First_Flavor', color='First_Flavor')
fig.update_layout(title_text='Flavor occurencies', title_x=0.5)
fig.show()
# ШАГ 14
# Выдвигаем гипотезу, что первый упомянутый эффекс является наиболее часто распознаваемым для штамма.
# Создадим гистограмму, которая бы отображала кол-во штаммов у которых в первую очередь распознается тот или иной эффек.
fig = px.histogram(df_copy, x='First_Effect', color='First_Effect')
fig.update_layout(title_text='Effect occurencies', title_x=0.5)
fig.show()
# ШАГ 15
# Версия для всех параметров
# Переводим все стринговые значения в числовые для обработки выбранным алгоритом
df_copy['Strain'] = pd.factorize(df_copy.Strain)[0] + 1
df_copy['First_Effect'] = pd.factorize(df_copy.First_Effect)[0] + 1
df_copy['Second_Effect'] = pd.factorize(df_copy.Second_Effect)[0] + 1
df_copy['Third_Effect'] = pd.factorize(df_copy.Third_Effect)[0] + 1
df_copy['Fourth_Effect'] = pd.factorize(df_copy.Fourth_Effect)[0] + 1
df_copy['Fifth_Effect'] = pd.factorize(df_copy.Fifth_Effect)[0] + 1
df_copy['First_Flavor'] = pd.factorize(df_copy.First_Flavor)[0] + 1
df_copy['Second_Flavor'] = pd.factorize(df_copy.Second_Flavor)[0] + 1
df_copy['Third_Flavor'] = pd.factorize(df_copy.Third_Flavor)[0] + 1
df_copy
# Версия для двух первых параметров
# Переводим все стринговые значения в числовые
df_copy['Strain'] = pd.factorize(df_copy.Strain)[0] + 1
df_copy['First_Effect'] = pd.factorize(df_copy.First_Effect)[0] + 1
df_copy['First_Flavor'] = pd.factorize(df_copy.First_Flavor)[0] + 1
df_copy
# ШАГ 16
# Убеждаемся, что все необходимые для анализа данные стали числовыми
df_copy.info()
# ШАГ 17
# Пример без тестовой выборки
# В качестве алгоритма ML выбираем Random Forest
# Разделяем колонки по признаку зависимости (эффекты и запахи зависят от мтамма)
X=df_copy[['First_Effect', 'Second_Effect', 'Third_Effect', 'Fourth_Effect', 'Fifth_Effect', 'First_Flavor', 'Second_Flavor', 'Third_Flavor']]
y=df_copy['Strain']
# Пример с тестовой выборкой
# В качестве алгоритма ML выбираем Random Forest
# Разделяем колонки по признаку зависимости (эффекты и запахи зависят от мтамма). На этот раз возьмем меньшее колчиество параметров для увеличения точности
X=df_copy[['First_Effect','First_Flavor']]
y=df_copy['Strain']

# Формирование тетсовой выборки: 70% данных для обучения и 30% для тестирования
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# ШАГ 18
# Без тестовой выборки
# Создаем классификатор Гуасса с сотней оценщиков
clf=RandomForestClassifier(n_estimators=100)

# Запускаем обучение модели 
clf.fit(X,y)
# C тестовой выборкой
# Создаем классификатор Гуасса с сотней оценщиков
clf=RandomForestClassifier(n_estimators=100)

# Запускаем обучение модели
clf.fit(X_train,y_train)
# C тестовой выборкой
y_pred=clf.predict(X_test)

# Проверим точность модели с помощью sklearn metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# В результате 
# ШАГ 19
# Можно еще раз вывести себе таблицу с эффектами, чтобы было удобнее при поиске
dfEffects
# ШАГ 20
# Можно еще раз вывести себе таблицу с запахами, чтобы было удобнее при поиске
dfFlavor
# ШАГ 21
#Вводим параметры для поиска
First_Effect = 3
Second_Effect = 5
Third_Effect = 2
Fourth_Effect = 7
Fifth_Effect = 6
First_Flavor = 1
Second_Flavor = 11
Third_Flavor = 9

# Получаем предсказание по комбинации эффектов / запахов
predict_result = clf.predict([[First_Effect,Second_Effect,Third_Effect,Fourth_Effect,Fifth_Effect,First_Flavor,Second_Flavor,Third_Flavor]])
df_copy.iloc[predict_result-1 , : ]

# Вводим параметры для поиска
First_Effect = 5
First_Flavor = 9


# Получаем предсказание по комбинации эффектов / запахов)
predict_result = clf.predict([[First_Effect,First_Flavor]])
df_copy.iloc[predict_result-1 , : ]

# ШАГ 22
# Оценим важность каждого атрибута при определении штамма
feature_imp = pd.Series(clf.feature_importances_,index=X.columns).sort_values(ascending=False)
feature_imp
# ШАГ 23
# Созадим график важности атрибутов штамма
sns.barplot(x=feature_imp, y=feature_imp.index)

# Добавим лейблы для понимания графика
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()