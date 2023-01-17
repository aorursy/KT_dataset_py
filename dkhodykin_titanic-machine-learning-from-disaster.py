# Импортируем необходимые для работы библиотеки

import pandas as pd

from sklearn.naive_bayes import GaussianNB

from imblearn.under_sampling import RandomUnderSampler

from sklearn.model_selection import train_test_split

from sklearn import metrics
# Импортируем данные

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")
# Как правильно обращаться к наименованиям столбцов

train.columns
# Выясним, в каких столбцах не хватает данных

train.isna().sum()
test.isna().sum()
# Недостающие данные по возрасту заполним средним арифметическим

mean_age = train['Age'].mean() # Определим среднее значение

print ('Средний возраст пассажира: ', round(mean_age, 2))

train['Age'].fillna(mean_age , inplace=True) # Заполним пустые значения

test['Age'].fillna(mean_age , inplace=True)
# Посмотрим, из какого порта отправилось большинство пассажиров

# Будет логично предположить, что 2 пассажира с неизвестным портом отправления сели там же

print(train['Embarked'].value_counts())
# Больше всего пассажиров отправились в путь на Титанике из порта S - Southampton

# Назначим это значение пассажирам, чей порт посадки не известен

train['Embarked'].fillna('S',inplace=True)
# Недостающие данные по тарифу заполним средними значениями

mean_fare = train['Fare'].mean() # здесь вычисляем

print ('Средняя такса: ', round(mean_fare, 2))

test['Fare'].fillna(mean_fare , inplace=True) # здесь заполняем
# Создаем словари

dict_Sex = {'male': 1, 'female': 0}

dict_Embarked = {'S': 0, 'C': 1, 'Q':2}



# Преобразуем данные при помощи map

train['Sex'] = train['Sex'].map(dict_Sex)

train['Embarked'] = train['Embarked'].map(dict_Embarked)

test['Sex'] = test['Sex'].map(dict_Sex)

test['Embarked'] = test['Embarked'].map(dict_Embarked)
# Создадим датасеты, готовые к дальнейшей обработке

test_clean = test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1)

train_clean = train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1)
# Посмотрим что получилось после предобработки

train_clean.sample(5)
len(train_clean)
# Разделим тренировочный датасет на две части: для обучения модели и ее тренировки

dfx = train_clean.drop('Survived', axis = 1)

dfy = train_clean[['Survived']] 



X_train, X_test, y_train, y_test = train_test_split(dfx, dfy, test_size=0.2, random_state=17)
# Создадим модель классификатора

mnb = GaussianNB()
# Обучим модель

mnb.fit(X_train, y_train.values.ravel())
y_pred = mnb.predict(X_test)

y_pred
# Рассмотрим качество работы модели

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

cnf_matrix
TN = cnf_matrix[0,0] # True Negative

TP = cnf_matrix[1,1] # True Positive

FN = cnf_matrix[1,0] # False Negative

FP = cnf_matrix[0,1] # False Positive

    

Ac = mnb.score(X_test, y_test)

Sens = TP/(TP+FN) 

Sp = TN/(TN+FP)

P = TP/(TP+FP)

typeI = FP/(FP+TN)

typeII = FN/(FN+TP)

    

print('Accuracy: ', Ac)

print('Sensitivity: ', Sens)

print('Specificity: ', Sp)

print('Pricision: ', P)

print('Type I error rate: ', typeI)

print('Type II error rate: ', typeII)
# Подсчитаем количество выживших в тестовой выборке

share = y_train['Survived'].value_counts()
# Создадим новую выборку данных, определим ее параметры

sampler = RandomUnderSampler(ratio={1: share[1], 0: share[1]})
# Unpersampling выполняется здесь:

X_train_under_np, y_train_under_np = sampler.fit_sample(X_train, y_train)
# Преобразуем выборки в DataFrame, чтобы передать их модели

X_train_under = pd.DataFrame(X_train_under_np)

y_train_under = pd.DataFrame(y_train_under_np)
mnb_under = GaussianNB()
# Обучим модель

mnb_under.fit(X_train_under, y_train_under.values.ravel())
# Сделаем предсказания на тестовой выборке

y_pred_under = mnb_under.predict(X_test)

y_pred_under
# Смотрим на confusion-матрицу

cnf_matrix_under = metrics.confusion_matrix(y_test, y_pred_under)

cnf_matrix_under
TN = cnf_matrix_under[0,0] # True Negative

TP = cnf_matrix_under[1,1] # True Positive

FN = cnf_matrix_under[1,0] # False Negative

FP = cnf_matrix_under[0,1] # False Positive

    

Ac = mnb_under.score(X_test, y_test)

Sens = TP/(TP+FN) 

Sp = TN/(TN+FP)

P = TP/(TP+FP)

typeI = FP/(FP+TN)

typeII = FN/(FN+TP)

    

print('Accuracy: ', Ac)

print('Sensitivity: ', Sens)

print('Specificity: ', Sp)

print('Pricision: ', P)

print('Type I error rate: ', typeI)

print('Type II error rate: ', typeII)
X_result = test_clean
len(X_result)
# Сделаем предсказание для тестовой (результирующей) выборки

y_result = mnb_under.predict(X_result)

y_result
result = pd.DataFrame()
result['PassengerId'] = test['PassengerId']

result['Survived'] = pd.Series(y_result)
result.info()
result.isna().sum()
result.to_csv("result.csv", index = False)