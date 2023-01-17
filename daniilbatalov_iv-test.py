# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/titanic/train.csv')
train_t = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')
gs=pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
ob = pd.concat([train, test], ignore_index=True, sort = False)
#Анализ данный в тренировочной выборке
display(ob)
print(ob.describe())
print(ob.info())
ob.isnull().sum()

# Находим процент выживших пассажиров
print(ob.groupby("Survived")["PassengerId"].count())
print(ob.Survived.count())
survived=342/891*100
print(survived)
#Выжило 38%, погибло 62%
# Заменяем значение male - 1, female - 0
for i in range(1309):
    if ob.Sex[i]=='female':
        ob.Sex[i]=0
    elif ob.Sex[i]=='male':
        ob.Sex[i]=1 
for i in range(891):
    if train.Sex[i]=='female':
        train.Sex[i]=0
    elif train.Sex[i]=='male':
        train.Sex[i]=1 
for i in range(418):
    if test.Sex[i]=='female':
        test.Sex[i]=0
    elif test.Sex[i]=='male':
        test.Sex[i]=1
#Анализ выживаемости пассажиров 1-3 класссов
print(ob.groupby("Pclass")["PassengerId"].count())
sur1,sur2,sur3=0,0,0
for i in range(891):
    if ob.Pclass[i]==1:
        if train.Survived[i]==1:
            sur1+=1
    elif ob.Pclass[i]==2:
        if train.Survived[i]==1:
            sur2+=1
    elif ob.Pclass[i]==3:
        if train.Survived[i]==1:
            sur3+=1
print("Процент выживших из первого класса:",(sur1/216*100))
print("Процент выживших из второго класса:",(sur2/184*100))
print("Процент выживших из третьего класса:",(sur3/491*100))
#Чем элитнее класс, тем выше процент выживания
# Статистика по выжившим мужчинам и женщинам на борту
male=female=sur_m=sur_f=0
for i in range(891):
    if ob.Sex[i]==1:
        male+=1
        if train.Survived[i]==1:
            sur_m+=1
    else:
        female+=1
        if ob.Survived[i]==1:
            sur_f+=1
print("Всего мужчин на борту:",male)
print("Всего женщин на борту:",female)

print("Процент выживших мужчин:",(sur_m/male*100))
print("Процент выживших женщин:",(sur_f/female*100))
#Заполним значения по возрасту средним по тренровочной выборке
train.fillna(train.mean(),inplace=True)
#Статистика по выжившим по возрасту:
_10=_20=_30=_40=_50=_60=_70=_80=0
s_10=s_20=s_30=s_40=s_50=s_60=s_70=s_80=0
for i in range(891):
    if 0<ob.Age[i]<=10:
        _10+=1
        if ob.Survived[i]==1:
            s_10+=1
    elif 10<ob.Age[i]<=20:
        _20+=1
        if ob.Survived[i]==1:
            s_20+=1
    elif 20<ob.Age[i]<=30:
        _30+=1
        if ob.Survived[i]==1:
            s_30+=1
    elif 30<ob.Age[i]<=40:
        _40+=1
        if ob.Survived[i]==1:
            s_40+=1
    elif 40<ob.Age[i]<=50:
        _50+=1
        if ob.Survived[i]==1:
            s_50+=1
    elif 50<ob.Age[i]<=60:
        _60+=1
        if ob.Survived[i]==1:
            s_60+=1
    elif 60<ob.Age[i]<=70:
        _70+=1
        if ob.Survived[i]==1:
            s_70+=1
    elif 70<ob.Age[i]<=80:
        _80+=1
        if ob.Survived[i]==1:
            s_80+=1
print("Количество человек от 0 до 10 лет:",_10,"Выжило:",s_10)
print("Количество человек от 10 до 20 лет:",_20,"Выжило:",s_20)
print("Количество человек от 20 до 30 лет:",_30,"Выжило:",s_30)
print("Количество человек от 30 до 40 лет:",_40,"Выжило:",s_40)
print("Количество человек от 40 до 50 лет:",_50,"Выжило:",s_50)
print("Количество человек от 50 до 60 лет:",_60,"Выжило:",s_60)
print("Количество человек от 60 до 70 лет:",_70,"Выжило:",s_70)
print("Количество человек от 70 до 80 лет:",_80,"Выжило:",s_80)
#Статистика влияния братьев/сестёр и родителей на выживание
ob['Family'] = ob.Parch + ob.SibSp
train['Family'] = train.Parch + train.SibSp
test['Family'] = test.Parch + test.SibSp
alone=sur_a=no_alone=sur_na=0
for i in range(891):
    if train.Family[i]==0:
        alone+=1
        if train.Survived[i]==1:
            sur_a+=1
    elif train.Family[i]>0:
        no_alone+=1
        if train.Survived[i]==1:
            sur_na+=1
for i in range(418):
    if test.Family[i]==0:
        alone+=1
    elif test.Family[i]>0:
        no_alone+=1
print("Количество человек без семьи:",alone,"Процент выживших:",(sur_a/alone*100))
print("Количество человек c семьёй:",no_alone,"Процент выживших:",(sur_na/no_alone*100))
    
#Удалим из выборки стобцы Fare, Embarked, Cabin и Ticket
train.drop("Fare",axis=1, inplace=True)
train.drop("Ticket",axis=1, inplace=True)
train.drop("Cabin",axis=1, inplace=True)
train.drop("Embarked",axis=1, inplace=True)
train.drop("SibSp",axis=1, inplace=True)
train.drop("Parch",axis=1, inplace=True)
train.drop("Name",axis=1, inplace=True)
train.drop("Survived",axis=1, inplace=True)
train.Age.fillna(train.Age.mean(), inplace=True)
test.drop("Fare",axis=1, inplace=True)
test.drop("Ticket",axis=1, inplace=True)
test.drop("Cabin",axis=1, inplace=True)
test.drop("Embarked",axis=1, inplace=True)
test.drop("SibSp",axis=1, inplace=True)
test.drop("Parch",axis=1, inplace=True)
test.drop("Name",axis=1, inplace=True)
test.Age.fillna(train.Age.mean(), inplace=True)
display(test)
#Модель

clf = DecisionTreeClassifier()
x_train, x_test, y_train, y_test = train_test_split(train, train_t.Survived, test_size=0.3)
clf.fit(x_train, y_train)
print("RF Accuracy: "+repr(round(clf.score(x_test, y_test) * 100, 2)) + "%")
result_rf=cross_val_score(clf,x_train,y_train,cv=10,scoring='accuracy')
print('The cross validated score for Random forest is:',round(result_rf.mean()*100,2))

#Оптимизация значений
parameteres = {'max_depth': list(range(1,30)), 
               'max_features': list(range(1,30)),
               'max_leaf_nodes': list(range(1,30))
              }
clf_grid = GridSearchCV(clf, param_grid=parameteres, cv=5, n_jobs=-1)
clf_grid.fit(x_train, y_train)
clf_grid.best_params_
clf_grid.score(x_train, y_train)
res=clf_grid.predict(test)
print(res)
print(len(res))
testik=test["PassengerId"]
dff=pd.DataFrame({"PassengerId": testik, "Survived": res})
display(dff)
dff.to_csv("submission.csv", index=False)