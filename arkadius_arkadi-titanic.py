# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in



import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn import tree, ensemble

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score





data = pd.read_csv("../input/train.csv")

data_test = pd.read_csv("../input/test.csv")



dates = [data, data_test]



# нулевые значения

print("Train data null colums : \n", data.isnull().sum(), "\n", "-"*20)

print("Test data null colums : \n", data_test.isnull().sum(), "\n", "-"*20)



for d in dates:

    # Женщина - 1 : Мужчина - 0

    d["gender"] = d["Sex"].apply(lambda x: int(x == "female"))



    # Заполняем пропуски медианными значениями

    d["Age"].fillna(d["Age"].median(), inplace=True)

    d["Fare"].fillna(d["Fare"].median(), inplace=True)



    # тут пропуски заполним модой

    d["Embarked"].fillna(d["Embarked"].mode(), inplace=True)



    # Отбрасываем ненужное

    d.drop(["Name", "Ticket", "Cabin", "Sex"], axis=1, inplace=True)
# Графики

def paint_hist(col):

    plt.hist(x=[data[data["Survived"] == 1][col], data[data["Survived"] == 0][col]],

             stacked=True, label=["Survived", "Dead"])

    plt.title(col + ' Histogram by Survival')

    plt.xlabel(col)

    plt.ylabel('# of Passengers')

    plt.legend()





block_graf = False

# видна явная зависимость выживания от пола

# графики возраста, parch и sibsp сильно похожи ( имеется в виду для выжили/не выжили)

# у Fare много значений лежит правее от основого графика

plt.figure(figsize=(16, 12))

id_gr = 230

for str_col in ["gender", "Age", "Fare", "Parch", "SibSp", "Pclass"]:

    id_gr += 1

    plt.subplot(id_gr)

    paint_hist(str_col)

plt.show(block=block_graf)



# Практически никакой корреляции между выживаемостью и age, sibsp, Parch

print(data[["Survived", "gender", "Age", "Fare", "Parch", "SibSp", "Pclass"]].corr())



# у Fare все значения больше 150 сделаем равными 150

for d in dates:

    d.loc[d["Fare"] > 150, "Fare"] = 150

paint_hist("Fare")

plt.show(block=block_graf)
# Для обучения и предсказания оставляем только эти колонки

column_for_predict = ["Pclass", "gender", "Fare"]



train = data[column_for_predict + ["Survived"]]

X = train.drop("Survived", axis=1)

y = train["Survived"]



X_train, X_test, y_train, y_test = train_test_split(X, y)





def fitting(alg):

    alg.fit(X_train, y_train)

    return alg





# Список классификаторов

classifiers = pd.DataFrame([

                ["DecisionTreeClassifier", fitting(tree.DecisionTreeClassifier(max_depth=3))],

                ["RandomForestClassifier", fitting(ensemble.RandomForestClassifier())],

                ["BaggingClassifier", fitting(ensemble.BaggingClassifier())],

                ["GradientBoostingClassifier", fitting(ensemble.GradientBoostingClassifier())]

            ],

            columns=["Name", "cf"]

            )



classifiers["train_acc"] = classifiers["cf"].apply(lambda x: accuracy_score(y_train, x.predict(X_train)))

classifiers["test_acc"] = classifiers["cf"].apply(lambda x: accuracy_score(y_test, x.predict(X_test)))

classifiers["delta_acc"] = classifiers.apply(lambda x: x["train_acc"] - x["test_acc"], axis=1)



print(classifiers[["Name", "train_acc", "test_acc", "delta_acc"]])

classifiers.plot(kind="barh", x=classifiers["Name"])

plt.xlim(0.7, 0.95)

plt.show(block=block_graf)



# Для тестовой выборки берем классификатор с наименьшей разность между точностями

classifier = classifiers.iloc[classifiers["delta_acc"].idxmin()]

print("Chosen : " + classifier["Name"])



# Результат

data_test["Survived"] = classifier["cf"].predict(data_test[column_for_predict])



submit = data_test[["PassengerId", "Survived"]]

# submit.to_csv("submit.csv", index=False)