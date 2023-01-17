import pandas as pd

import numpy as np



from sklearn import model_selection

from sklearn import tree

from sklearn import ensemble

from sklearn.model_selection import GridSearchCV



import matplotlib.pyplot as plt

%matplotlib inline



#изменить данные исходного датасета, чтобы 

def transform_dataset(ds):

    transformed_dataset = ds.copy()



    #Заполнить пропуски в столбцах Age и Fare средними значениями заполненных строк

    transformed_dataset['Age'].fillna(transformed_dataset['Age'].median(), inplace=True)

    transformed_dataset['Fare'].fillna(transformed_dataset['Fare'].median(), inplace=True)



    #заменить столбец Sex на перечислимое значение

    transformed_dataset['Sex'] = pd.factorize(transformed_dataset['Sex'])[0]

    #разделить столбец Embarked

    #get_dummies, добавляет столбцы, соответствующие всем уникальным значениям столбца. Таким образом, если  возможны значения столбца — Q, C и S, метод get_dummies создает три различных столбца и назначает им значения 0 или 1 в зависимости от соответствия значения этому столбцу.

    transformed_dataset=pd.concat([transformed_dataset, pd.get_dummies(transformed_dataset['Embarked'],  prefix="Embarked")], axis=1)



    #Разделить столбец Parch на два столбца Childs и Parents (пытаясь основываться на законы логики)

    transformed_dataset['Childs'] = transformed_dataset.apply (lambda row: get_childs(row), axis=1)

    transformed_dataset['Parents'] = transformed_dataset.apply (lambda row: get_parents(row), axis=1)



    transformed_dataset = transformed_dataset.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked', 'Parch'], axis=1)

    return transformed_dataset



#Если число родителей/детей больше двух, и возраст больше 18

#будем считать, что это родитель и вернем для сроки количество детей, иначе - 0

def get_childs(row):

    if (row.Parch>2 or row.Age>=18):

        return  row.Parch

    return 0



#Если число родителей/детей не больше двух, и возраст меньше 18

#будем считать, что это ребенок и вернем для строки количество родителей, иначе - 0

def get_parents(row):

    if (row.Parch <= 2 and row.Age<18):

        return row.Parch

    return 0
#считать датасет из файла, вывести информацию о датасете

dataset = pd.read_csv('../input/titanic/train.csv')

dataset.info()
#категоризировать датасет (вынес в отдельный файл transform.py, поскольку встречается несколько раз)

#на выходе получаем набор с числовыми значениями

transformed_dataset = transform_dataset(dataset)

transformed_dataset.info()
#Указываем зависимости. Y(выживаемость пассажира) зависит от всех других параметров

X = transformed_dataset.drop(['Survived'], axis=1)

Y = dataset['Survived']
#разбиваем датасет на обучающий и тестовый

#test_size - указывает процент выборки, который станет тестовым(20%) 

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size = 0.2, random_state = 21)
#определить модель решения через деревья решений

#настроить параметры модели

#n_estimators - число деревьев

#max_depth - максимальная глубина деревьев

#max_features - число признаков для выбора расщепления

param_grid =[ {'n_estimators' : [10, 15, 20, 25, 30, 35, 40], 'max_depth' : [5,10,15, 20]},]

rf = ensemble.RandomForestClassifier(random_state=21, max_features= 3)

model = GridSearchCV(rf,param_grid, cv = 5)
#тренируем модель на обучающей выборке X_train,Y_train

model.fit(X_train,Y_train)

#выводим результат точности(accuracy) на обучающей тестовой выборке

print('train score = ', model.score(X_train,Y_train), '\n test score = ', model.score(X_test,Y_test), '\n ', model.best_params_)
#считываем второй датасет из файла (который используется для проверки алгоритма, без столбца о том, выжил ли пассажир)

test = pd.read_csv('../input/titanic/test.csv')

test.info()
#запоминаем столбец PassengerId

passenger_id = test['PassengerId']
#приводим к виду первого датасета, на котором производилось обучение

transformed_test = transform_dataset(test)

transformed_test.info()
#предсказываем значение целевого параметра(выживаемости) по зависимым параметрам

Y_predict = model.predict(transformed_test)

#и изменяем полученный результат в именованный столбец

Y_p = pd.DataFrame(Y_predict, columns=['Survived'])

Y_p
#объединяем столбец-предсказанный результат со столбцос Id пассожиров

#axis=1 - конкатенация по горизонтали

res = pd.concat([passenger_id, Y_p], axis=1)

res
#записываем результат в файл

res.to_csv('res.csv', index=None)