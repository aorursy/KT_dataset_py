# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#импортируем библиотеки и процедуры

import seaborn as sb

import numpy as np



import pandas as pd

from pandas import Series 

from pandas import DataFrame



import matplotlib.pyplot as plt

from matplotlib import rcParams



import sklearn

from sklearn.preprocessing import scale 

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_validate

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn import preprocessing

from sklearn.metrics import classification_report





def prepareTable(table):   

    #убираем дубликаты

    drop_duplicates = table.drop_duplicates()



    #заполняем пустые ячейки нулями

    training_data_set = drop_duplicates

    

    training_data_set["Age"]      = training_data_set["Age"].fillna(training_data_set["Age"].median())

    training_data_set["Fare"]     = training_data_set["Fare"].fillna(training_data_set["Fare"].median())

    training_data_set["Embarked"] = training_data_set["Embarked"].fillna("S")  

    training_data_set             = training_data_set.fillna(0)



    #перекодируем некоторые солбцы таблицы

    training_data_set.Sex.replace(['male', 'female'], [1, 0], inplace=True)

    training_data_set.Embarked.replace(['C', 'Q', 'S'], [0, 1, 2], inplace=True)



    cabins = training_data_set.Cabin

    cabins = cabins.drop_duplicates()



    indexes = Series(np.arange(len(cabins)))

    indexes.name = "Cabin_Code"



    df = DataFrame(cabins)

    df.index = np.arange(len(df))



    result_cabins = DataFrame.join(df, indexes)



    training_data_set_result = pd.merge(training_data_set, result_cabins, how='left')

    training_data_set_result = training_data_set_result.drop(columns = 'Cabin')



    tickets = training_data_set.Ticket

    tickets = tickets.drop_duplicates()



    indexes = Series(np.arange(len(tickets)))

    indexes.name = "Ticket_Code"



    dff = DataFrame(tickets)

    dff.index = np.arange(len(dff))



    result_tickets = DataFrame.join(dff, indexes)



    training_data_set_result = pd.merge(training_data_set_result, result_tickets, how = 'left')

    training_data_set_result = training_data_set_result.drop(columns = 'Ticket')



    #убираем столбцы, которые мы посчтитали неинформативными

    training_data_set_result = training_data_set_result.drop(columns = 'PassengerId')

    training_data_set_result = training_data_set_result.drop(columns = 'Name')

    

    return training_data_set_result





#считываем данные

address = '../input/train.csv'

table = pd.read_csv(address)

#print(table)



#заполняем пустые ячейки и нормализуем

training_data_set_encode = prepareTable(table)

training_data_set_encode = training_data_set_encode.drop(columns = 'Cabin_Code')

training_data_set_encode = training_data_set_encode.drop(columns = 'Ticket_Code')

#print(training_data_set_encode)



#готовим данные для ML

y = training_data_set_encode.Survived

X = training_data_set_encode.drop(columns = 'Survived')

X = scale(X)



#обучаем алгоритм LogisticRegression

LogReg = LogisticRegression(C = 0.005, random_state=1)

LogReg.fit(X,y)



#сравниваем результаты обучения с исходными показателями

y_pred = LogReg.predict(X)

y_pred = Series(y_pred)

print(y_pred.value_counts())



#обучаем алгоритм RandomForestClassifier

RFC = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=4, min_samples_leaf=2)

RFC.fit(X,y)



#сравниваем результаты обучения с исходными показателями

y_pred = RFC.predict(X)

y_pred = Series(y_pred)

print(y_pred.value_counts())



#считываем данные для теста

address_test = '../input/test.csv'

table_test = pd.read_csv(address_test)



PassengerId = table_test.PassengerId

test_data_set_encode = prepareTable(table_test)

test_data_set_encode = test_data_set_encode.drop(columns = 'Cabin_Code')

test_data_set_encode = test_data_set_encode.drop(columns = 'Ticket_Code')



X = scale(test_data_set_encode)

y_pred = RFC.predict(X)



#выводим результат в файл

s = Series(y_pred)

s.name = 'Survived'



result = DataFrame.join(DataFrame(PassengerId), s)

result.to_csv('result.csv', index=False)