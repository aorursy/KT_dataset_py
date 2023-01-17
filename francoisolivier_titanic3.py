import numpy as np

import pandas as pd

import re



#Print you can execute arbitrary python code

train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, index_col='PassengerId')

test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )



#Print to standard output, and see the results in the "log" section below after running your script

print("\n\nTop of the training data:")

print(train.head())



#print("\n\nSummary statistics of training data")

#print(train.describe())



#Any files you save will be available in the output tab below

#train.to_csv('copy_of_the_training_data.csv', index=False)



print('1.')

print(train['Sex'].value_counts())



survived_counts = train['Survived'].value_counts()



print('2.')

print('Survived share:')

print('method 1')

print(round(survived_counts/sum(survived_counts) * 100,2) )

print('method 2')

print(round(train.groupby('Survived').agg('size').apply(lambda x: x/train['Survived'].count() ) * 100, 2))



print('3.')

print('First class share:')

print(round( (train[ train['Pclass'] == 1]['Pclass'].count() / train['Pclass'].count()) * 100, 2))



print('4.')

print('Average and median:')

print(train.mean(0,skipna=True))

print(train.median(0,skipna=True))



print('5.')

print('Correlation coeff between братьев/сестер/супругов and числом родителей/детей')

print(np.corrcoef(train[['SibSp','Parch']], rowvar=0))



print('6.')

#print(train['Name'].apply( lambda rawname: re.split(',', rawname)[0]).describe() )



print(train[ train['Sex'] == 'female' ]['Name'].apply( lambda rawname: re.split('\.', rawname)[1]).describe())



print(train.describe())



## cleansing

# Оставьте в выборке четыре признака: класс пассажира (Pclass), цену билета (Fare), возраст пассажира (Age) и его пол (Sex).

#tr = ~np.isnan( train['Pclass'] ) & ~np.isnan( train['Fare'] ) & ~np.isnan( train['Age'] ) & ~np.isnan( train['Sex'] )

#tr = train.dropna()



print(np.isnan(train['Age']).value_counts())



tr = train[~np.isnan(train['Age'])]



#from sklearn import preprocessing

#le = preprocessing.LabelEncoder()

#le.fit(tr['Sex'])

#tr.loc[:, 'S'] = le.transform(tr['Sex'])

#tr.insert(0, 'S', le.transform(tr['Sex']))

#tr.loc[:, 4] = le.transform(tr['Sex'])





tr.insert(0, 'Gender', train['Sex'].map({'female': 0, 'male': 1}).astype(int))





print(tr.describe)



X = tr[ ['Pclass', 'Fare', 'Age', 'Gender'] ]

Y = tr[ ['Survived'] ]

# Обратите внимание, что признак Sex имеет строковые значения.

#Выделите целевую переменную — она записана в столбце Survived.

#В данных есть пропущенные значения — например, для некоторых пассажиров неизвестен их возраст. Такие записи при чтении их в pandas принимают значение nan. Найдите все объекты, у которых есть пропущенные признаки, и удалите их из выборки.



#Обучите решающее дерево с параметром random_state=241 и остальными параметрами по умолчанию (речь идет о параметрах конструктора DecisionTreeСlassifier).



#Вычислите важности признаков и найдите два признака с наибольшей важностью. Их названия будут ответами для данной задачи (в качестве ответа укажите названия признаков через запятую или пробел, порядок не важен).





## decision tree

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=241)

clf.fit(X, Y)



importances = clf.feature_importances_



print(importances)



indices = np.argsort(importances)[::-1]



print(X.columns[indices])



print(X.describe())