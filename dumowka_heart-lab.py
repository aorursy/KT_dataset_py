# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from statsmodels.stats.weightstats import _tconfint_generic



from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn import tree

import seaborn as sns



from sklearn.cluster import KMeans



import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (28,30)



import graphviz as gv



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Считываем данные

df = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")

df
# Типы переменных

df.info()
# Поиск null значений

df.isnull().sum()
# Смотрим уникальные значения в каждом параметре

for column in df.columns:

    print('==============================')

    print(f"{column} : {df[column].unique()}")
# Стоим диаграммы кол-во людей с заболеваниями и без относительно переменных



for column in df.columns[:-1]:

    if df[column].nunique() > 5:

        f, ax = plt.subplots(figsize=(8, 6))

        sns.boxplot(x="target", y=column, data=df)

    else:

        f, ax = plt.subplots(figsize=(8, 6))

        ax = sns.countplot(x= column, hue="target", data=df)

plt.show()
# Подсчет мужчин и женщин

df.groupby('sex')['target'].value_counts()
# Подсчет людей с разными болями в груди

df.groupby('cp')['target'].value_counts()
# Подсчет людей по параметру кол-во сахара в крови

df.groupby('fbs')['target'].value_counts()
# Подсчет людей по параметру ЭКГ в покое

df.groupby('restecg')['target'].value_counts()
# Подсчет людей по наличию стенокардии после физической нагрузки

df.groupby('exang')['target'].value_counts()
# Подсчет людей по наклону пика упражнений сегмента ST

df.groupby('slope')['target'].value_counts()
# Подсчет людей по количеству окрашенных крупных сосудов

df.groupby('ca')['target'].value_counts()
# Подсчет людей по уровню гемоглобина в крови

df.groupby('thal')['target'].value_counts()
# Стоим матрицу корреляции

df.corr().abs().style.background_gradient(cmap='coolwarm').set_precision(2)
# Корреляция с target

df.corr()['target'].sort_values(ascending=False)
# Описательная статистика 

df.describe()
# Описательная статистика для людей без заболеваний

df[df["target"] == 0].describe()

# Описательная статистика для людей с заболеванием

df[df["target"] == 1].describe()
# Избавляемся от аномалии в thal

df.loc[df['thal'] == 0, 'thal'] = 2
# Удаляем незначимые переменные

df = df.drop(['fbs', 'chol'], axis = 1)
Y = df["target"] 

X = df.drop("target", axis=1, inplace=False)
# Стандартизируем данные

from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

scaled_X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, Y, test_size=0.33, random_state=42)
# Логистическая регрессия

model = LogisticRegression()

model.fit(X_train, y_train)
model.score(X_test, y_test)
# Дерево решений

tree = DecisionTreeClassifier(random_state=42)

tree.fit(X_train, y_train)
tree.score(X_test, y_test)
# Оценка информативности признаков

f_imp = pd.DataFrame({"feature": X.columns, "importance": tree.feature_importances_})

f_imp = f_imp[f_imp["importance"]>0.04].sort_values("importance", ascending=False)

f_imp