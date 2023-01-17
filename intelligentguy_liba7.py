# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.tree import DecisionTreeClassifier

from sklearn import tree



import graphviz as gv







import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (28,30)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_excel("/kaggle/input/Credit.xlsx")

df
df.columns
for col in df.columns:

    if df[col].dtype not in (np.int32, np.int64, np.float32, np.float64, np.int16, np.uint8):

        print(df[col].value_counts(dropna=False))

        print("--------------------------------------")
df.drop(

    [

        "Дата кредитования", # Не интересно, равзве что день недели извлеч, но тоже сомнительно

        "Количество", # У всех событий только одно значение =1

        "Специализация" # 

    ], axis=1, inplace=True)
df.replace({"Да":1, "Нет":0}, inplace=True)

factors = {}

for col in [

            "Пол",

            "Расположение",

            "Должность"

           ]:

    df[col], factors[col] = pd.factorize(df[col])
df["Образование"] = df["Образование"].map({"среднее":0, "специальное": 1, "высшее":2})

df["Машина"] = df["Машина"].map({"Нет автомобиля":0, "отечественная": 1, "импортная":2}) # Не совсем правильно

df["Класс предприятия"] = df["Класс предприятия"].map({"малое":0, "среднее": 1, "крупное":2})
for col in ["Цель кредитования", "Способ приобретения собств.", "Отрасль предприятия", "Основное направление расходов"]:

    vc = df[col].value_counts()

    vc = vc[vc<7].index

    df[col].replace(vc, np.nan, inplace=True)

df = pd.get_dummies(df)
df.dtypes
Y = df["Давать кредит"] # отделяем признаки от классов

X = df.drop("Давать кредит", axis=1, inplace=False)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=40)

dtc = DecisionTreeClassifier(random_state=1,max_depth=7, min_samples_split=0.05, min_samples_leaf=2)#, class_weight="balanced")

dtc.fit(X_train, y_train)

tree.plot_tree(dtc,

               feature_names = X.columns, 

               class_names=["Нет", "Да"],

               filled = True);
dtc.score(X_test, y_test)
f_imp = pd.DataFrame({"feature": X.columns, "importance": dtc.feature_importances_})

f_imp = f_imp[f_imp["importance"]>0.04].sort_values("importance", ascending=False)

f_imp
parameters = dict(max_depth=np.arange(3, 10),

                  min_samples_split=np.arange(3, 15),

                  min_samples_leaf=np.arange(1, 6))

dtc = DecisionTreeClassifier(random_state=1)

gs = GridSearchCV(dtc, parameters, scoring=['accuracy', 'f1', 'precision', 'recall', 'roc_auc'], cv=7, return_train_score=True, refit="accuracy")

gs.fit(X, Y)

results = pd.concat(

    [pd.DataFrame(gs.cv_results_["params"])] + 

        [

            pd.DataFrame(gs.cv_results_["mean_test_" + metric], columns=[metric])

                for metric in ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']

        ] + 

        [

            pd.DataFrame(gs.cv_results_["mean_train_" + metric], columns=["train " + metric])

                for metric in ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']

        ],

    axis=1)

results
gs.best_params_, gs.best_score_
dupl = results.duplicated(subset=["accuracy", "f1", "precision", "recall", "roc_auc",

                                  "train accuracy", "train f1", "train precision", "train recall", "train roc_auc"], keep="first")

results = results[~dupl]

results.sort_values("accuracy", ascending=False).head(10)
X = X[['Сумма кредита', 'Расположение', 'Срок кредита', 'Среднемес. доход','Среднемес. расход']]
gs.fit(X, Y)

results = pd.concat(

    [pd.DataFrame(gs.cv_results_["params"])] + 

        [

            pd.DataFrame(gs.cv_results_["mean_test_" + metric], columns=[metric])

                for metric in ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']

        ] + 

        [

            pd.DataFrame(gs.cv_results_["mean_train_" + metric], columns=["train " + metric])

                for metric in ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']

        ] +

        [pd.DataFrame(gs.cv_results_['std_test_accuracy'], columns=["std_test_accuracy"])],

    axis=1)

dupl = results.duplicated(subset=["accuracy", "f1", "precision", "recall", "roc_auc", "train accuracy",], keep="first")

results = results[~dupl]

results.sort_values("accuracy", ascending=False).head(10)
f_imp = pd.DataFrame({"feature": X.columns, "importance": gs.best_estimator_.feature_importances_})

f_imp = f_imp[f_imp["importance"]>0.00].sort_values("importance", ascending=False)

f_imp
tree.plot_tree(gs.best_estimator_,

               feature_names = X.columns,

               class_names=["Нет", "Да"],

               filled = True)

gv.Source(tree.export_graphviz(gs.best_estimator_,

               feature_names = X.columns,

               class_names=["Нет", "Да"],

               filled = True), format='png').render(filename="credit_tree", cleanup=True)