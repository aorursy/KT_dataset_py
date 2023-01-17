#Импорт библиотек

import pandas as pd

import numpy as np



#Библиотеки визуализаций

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



#Методы ML

from sklearn import svm

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier



#Грузим датасеты и смотрим на данные

dataset = pd.read_csv("../input/titanic/train.csv")

train_x = dataset[['Pclass', 'Sex', 'Age', 'SibSp','Parch', 'Fare', 'Embarked']]

train_y = dataset['Survived'].values 



testdata= pd.read_csv("../input/titanic/test.csv")

test_x= testdata[train_x.columns.values] #makes it same size as train_x column wise



print(dataset.shape, train_x.shape, train_y.shape, test_x.shape)

train_x.head(10) #train_x.head()



test_x.head(5)



#Проверяем наличие отсутствующих значений

print('train описание данных:')

dataset.info(),

print('\n','test описание данных:')

test_x.info()



#Смотрим ключевые метрики на датасете

dataset.describe()



#Строим график выживаемости по возрастам. Видим что только 38% выжило, при этом большинство - подростки

figure, axis = plt.subplots(1,1,figsize=(20,3))

age = dataset[["Age", "Survived"]].groupby(['Age'],as_index=False).mean()

sns.barplot(x='Age', y='Survived', data= age)



figure = plt.figure(figsize=(13,8))

plt.hist([train_x[dataset['Survived']==1]['Age'].dropna(),train_x[dataset['Survived']==0]['Age'].dropna()], stacked=True, color = ['r','b'], bins = 30,label = ['Survived','Dead'])

plt.xlabel('Age')

plt.ylabel('Passengers Count')

plt.legend()



#заполняем пустые значения в возрастах медианным возрастом

age_fill = [train_x['Age'].median(), test_x['Age'].median()]

train_x['Age'].fillna(age_fill[0], inplace=True)

test_x['Age'].fillna(age_fill[1], inplace=True)



#Загоняем оба датасета в единый датафрейм для удобства манипуляций с данными

print (train_x.shape, train_y.shape)

combined = train_x.append(test_x) #[featureset, test_x]

print (type(combined), combined.shape)



survived_peeps = train_x[dataset['Survived']==1]['Sex'].value_counts()

dead_peeps = train_x[dataset['Survived']==0]['Sex'].value_counts()

df = pd.DataFrame([survived_peeps,dead_peeps])

df.index = ['Survived','Dead']

print (df)

df.plot(kind='bar',stacked=True, figsize=(12,6))



#На построенном выше графике видим корреляцию между выживаемостью и полом-женщины составляют большинство выживших

#Заменяем качественную переменную пола количественной

fixSex = combined['Sex'].copy().values

fixSex[fixSex == 'male'] = 0  

fixSex[fixSex == 'female'] = 1 

fixSex.shape



grid = sns.FacetGrid(dataset, col ='Sex', size = 3.2, aspect =1.7)

grid.map(sns.barplot, 'Embarked','Survived', alpha= 0.6, ci = None)



combined['Embarked'].fillna(combined['Embarked'].mode()[0], inplace=True) #filling Null vals in Embarked column

fixEmb = combined['Embarked'].copy().values



#Выживших женщин существенно больше вне зависимости от порта посадки



fixEmb[fixEmb =='S'] = 0 #Меняем опять качественную переменную на количественную

fixEmb[fixEmb =='C'] = 1

fixEmb[fixEmb =='Q'] = 2

fixEmb.shape



#Проверка на пустоты и заполнение

combined['Fare'].fillna(combined['Fare'].median(), inplace=True)

combined.info() #везде нет отсутствующих значений



#Начинаем процедуру замены не числовых значений на числовые

print (combined.shape)

combined.head(15) #датасет до трансформации в числовой





allnum_dataset = combined.copy()

allnum_dataset.loc[:,'Sex'] = fixSex 

allnum_dataset.loc[:,'Embarked'] = fixEmb

allnum_dataset.head(15) #датасет после трансформации в числовой



#Разделяем тренировочный и тестовые датасеты из датафрейма

X = allnum_dataset.copy()[:891]

test_x=  allnum_dataset.copy()[891:]

X.shape, test_x.shape #разделенный датафрейм на два датасета





#Тренировочные и проверочные данные разделяем для точной оценки модели SVM

XtrainV, XtestV, ytrainV, ytestV = train_test_split(X,train_y, test_size = 0.30)

XtrainV.shape, ytrainV.shape, XtestV.shape, ytestV.shape  # X = XtrainV(70%) + XtestV(30%), train_y = ytrainV(70%) +ytestV(30%)





cla_sv =svm.SVC()# SVM- rbf kernel классификатор

cla_sv





print ("размер проверочных тренировочных данных:","X inputs->", XtrainV.shape,", y targets-> ", ytrainV.shape)

cla_sv.fit(XtrainV, ytrainV)

print ("\n","ожидаемая точность:",cla_sv.score(XtrainV, ytrainV))









print ("размер проверочных тестовых данных:","X inputs->", XtestV.shape,", y targets-> ", ytestV.shape)

ypred = cla_sv.predict(XtestV)

print ("\n","ожидаемая точность прогноза:", metrics.accuracy_score(ytestV, ypred))





#Применяем RBF kernel SVM на полных тренировочных данных

print ("размер всего тренингового набора данных:","X inputs->", X.shape,", y targets-> ", train_y.shape)

cla_sv.fit(X, train_y)

print ("\n","ожидаемая точность прогноза:", cla_sv.score(X, train_y))





print ("размер всего тестового набора данных:","X inputs->", test_x.shape,  "y targets->", test_x.shape[0])

target_sv = cla_sv.predict(test_x)

submission_sv= pd.DataFrame({'PassengerId':testdata['PassengerId'].values, 'Survived': target_sv})





submission_sv.shape

submission_sv.head(10)

submission_sv.to_csv('submission_sv.csv', index = False)