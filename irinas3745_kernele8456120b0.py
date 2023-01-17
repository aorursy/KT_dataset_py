import pandas as pd

import os



train = pd.read_csv('../input/train.csv', header = 0)

test  = pd.read_csv('../input/test.csv' , header = 0)

full = [train, test]

print('количество пассажиров', train['Name'].count())

print('средний возраст пассажиров', test['Age'].mean()) #средний возраст

print('минимальный возраст пассажиров',test['Age'].min()) #мин возраст

print (train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).count()) #кол-во выживших мужчин и женщин

print('человек в возрасте от 50 до 60 лет:',train.loc['50':'60', 'Age'].count()) #76 человек в возрасте от 50 до 60 лет

print(train[['Ticket','Pclass']].groupby(['Pclass'],as_index=False).count()) #распределение билетов по классам

print (train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()) #кол-во билетов по классам

print (train[['Ticket', 'Survived']].groupby(['Ticket'], as_index=False))

#test[test.Age > 15][['Name']] #вывод имен пассажиров старше 15 лет

test[train.Survived==1][['Pclass','Name']] #выжившие в различных классах
