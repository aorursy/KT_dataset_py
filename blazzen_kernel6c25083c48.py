import numpy as np

import pandas as pd

import warnings

warnings.filterwarnings("ignore")
# переводит N-й день (1..1099) в соответствующий день недели (1..7)

def get_weekday(day):

    weekday = day % 7

    if weekday == 0:

        weekday = 7

    return weekday



# переводит N-й день (1..1099) в соответствующий номер недели (0..156)

def get_week(day):

    return (day - 1) // 7



# возвращает словарь со счётчиками уникальных значений в массиве

def get_counts_dict(arr):

    return dict(zip(*np.unique(arr, return_counts=True)))
# создание представлений для более удобного построения признаков

train = pd.read_csv('../input/train.csv')

train['visits'] = train['visits'].apply(

    lambda s: np.array(sorted([int(x) for x in s.strip().split(" ")]))

)



# all_days[i] == 1 значит, что был визит в день i+1

all_days = np.zeros((300000,  1099), dtype=int)

# all_weeks[i] - день недели первого визита на (i+1)-ой неделе

all_weeks = np.zeros((300000,  157), dtype=int)



for i in range(0, 300000):

    for day in train['visits'][i]:

        all_days[i][day - 1] = 1

        

        week = get_week(day)

        if all_weeks[i][week] == 0:

            all_weeks[i][week] = get_weekday(day)
X_train = np.zeros((all_days.shape[0], 18), dtype=float)

X = np.zeros((all_days.shape[0], 18), dtype=float)

y = np.zeros((all_days.shape[0],), dtype=int)



for i in range(0, 300000):

    # отрезаем у каждого посетителя все последние недели без посещений

    days = all_days[i]

    while np.count_nonzero(days[-7:]) == 0:

        days = days[:-7]

    

    # первые 7 признаков

    # для X - доли посещений по определённым дням недели от общего количества посещений

    # для X_train - аналогично, только без учёта последней недели

    # y - день недели первого посещения на последней посещённой неделе

    flag = False

    for j in range(0, 7):

        if days[-7:][j] == 1 and flag == False:

            y[i] = j + 1

            flag = True 

        X[i, j] = np.sum(days[[x for x in range(j, days.shape[0], 7)]])

        X_train[i, j] = np.sum(days[[x for x in range(j, days.shape[0] - 7, 7)]])

    X[i][:7] = np.divide(X[i][:7], np.sum(X[i][:7]))

    X_train[i][:7] = np.divide(X_train[i][:7], np.sum(X_train[i][:7]))

    

    # следующие 7 признаков - доли дней недели первых посещений от общего количества первых посещений

    weeks = all_weeks[i]

    counts = get_counts_dict(weeks)

    del counts[0]

    for weekday, count in counts.items():

        X[i, 6 + weekday]  = count

        X_train[i, 6 + weekday] = count

    last_weekday = weeks[get_week(days.shape[0])]

    X_train[i, 6 + last_weekday] -= 1

    X[i][7:14] = np.divide(X[i][7:14], np.sum(X[i][7:14]))

    X_train[i][7:14] = np.divide(X_train[i][7:14], np.sum(X_train[i][7:14]))

    

    # ещё 4 признака:

    # 15-й - разница в днях между двумя последними визитами

    # 16-18 - дни недели трёх последних визитов

    visits = train['visits'][i]

    X[i, 14] = visits[-1] - visits[-2]

    X[i, 15] = get_weekday(visits[-1])

    X[i, 16] = get_weekday(visits[-2])

    X[i, 17] = get_weekday(visits[-3])

    last_week = get_week(visits[-1])

    while get_week(visits[-1]) == last_week:

        visits = visits[:-1]

    X_train[i, 14] = visits[-1] - visits[-2]

    X_train[i, 15] = get_weekday(visits[-1])

    X_train[i, 16] = get_weekday(visits[-2])

    X_train[i, 17] = get_weekday(visits[-3])
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(solver='sag', multi_class='multinomial', max_iter=100)

logreg.fit(X_train, y)

res = logreg.predict(X)

pd.DataFrame({'id': np.arange(1, 300001), 'nextvisit':[' ' + str(x) for x in res]}).to_csv('./solution.csv', index=False)