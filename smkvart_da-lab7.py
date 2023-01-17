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
df = pd.read_excel("../input/Credit.xlsx")

df
df.columns
df = df.drop(['Количество', 'Дата кредитования'], axis=1) #значение параметра всегда одно и то же, а дата не влияет
df = df.replace({"Да":1, "Нет":0}) #меняем бинарные категории на числовые
fct = {}

for col in [

            "Пол",

            "Расположение",

            "Должность",

            "Специализация",

           ]:

    df[col], fct[col] = pd.factorize(df[col])
df["Образование"] = df["Образование"].map({"среднее":0, "специальное": 1, "высшее":2})

df["Машина"] = df["Машина"].map({"Нет автомобиля":0, "отечественная": 1, "импортная":2})

df["Класс предприятия"] = df["Класс предприятия"].map({"малое":0, "среднее": 1, "крупное":2})
from sklearn.preprocessing import LabelEncoder #кодировщик, который каждой категории сопоставляет целое число



le = LabelEncoder()

for col in ["Цель кредитования", "Способ приобретения собств.", "Отрасль предприятия", "Основное направление расходов"]:

    le.fit(df[col])

    df[col] = le.transform(df[col])

df
df.dtypes #видим, что все переменные числовые
from sklearn.model_selection import train_test_split



Y = df["Давать кредит"]

X = df.drop("Давать кредит", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=40) #разделяем выборку на обучающую и тестовую
from sklearn.tree import DecisionTreeClassifier



dtc = DecisionTreeClassifier()

dtc.fit(X_train, y_train) #обучаем модель
from sklearn.model_selection import GridSearchCV



params = {

    "criterion" : ["gini", "entropy"],

    "max_depth" : [None, 1, 2, 3, 10]

}



#Поиск наилучших параметров делаем по точности(precision, полноте(recall) среднему гармоническому 

#precision и recall(f1), последний из которых является результирующим в подборе параметров

#Почему они? Precision и recall не зависят, в отличие от accuracy, от соотношения классов и потому применимы в условиях несбалансированных выборок

#А f1 как гармноческое соотношение между ними, очевидно, хороший параметр, который следует улучшать

GridSCV = GridSearchCV(dtc, params, scoring= ['f1', 'precision', 'recall'], refit="f1", cv=10)

GridSCV.fit(X, Y)
GridSCV.best_score_
f_imp = pd.DataFrame({"Признак": X.columns, "Информативность": GridSCV.best_estimator_.feature_importances_})

f_imp = f_imp[f_imp["Информативность"]>0.00].sort_values("Информативность", ascending=False)

f_imp
df_n = df.drop(f_imp['Признак'][f_imp['Информативность'] < 0.05], axis=1) #выбрасывем шумовые признаки
Y = df_n["Давать кредит"]

X = df_n.drop("Давать кредит", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=40) #разделяем выборку на обучающую и тестовую



dtc = DecisionTreeClassifier()

dtc.fit(X_train, y_train) #обучаем модель



GridSCV_n = GridSearchCV(dtc, params, scoring= ['f1', 'precision', 'recall'], refit="f1", cv=10)

GridSCV_n.fit(X, Y)
GridSCV_n.best_score_ - GridSCV.best_score_  #Качество классификации улучшилось после исключения шумовых признаков, но не значительно