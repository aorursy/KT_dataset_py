# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print("\""+str(os.path.join(dirname, filename))+"\", ")



# Any results you write to the current directory are saved as output.
import xlrd



files = ["/kaggle/input/divorces.xlsx",  

"/kaggle/input/nonlocal_laborers.xlsx", 

"/kaggle/input/housing_per_1000.xlsx.xlsx", 

"/kaggle/input/loc_laborers.xlsx", 

"/kaggle/input/birth.xlsx", 

#"/kaggle/input/demography.xlsx", 

"/kaggle/input/marriages.xlsx"] #список путей к файлам



df = pd.DataFrame({"Unnamed: 0": []})

for filename in files:

    df1=pd.read_excel(filename, header=1) #читаем каждый файл по отдельности

    df1 = df1.dropna() #удаляем пропущенные значения

    del df1["Unnamed: 1"] #удаляем коды субъектов федерации

    for col in df1.columns:

        if col != "Unnamed: 0":

            df1[col+filename.replace("/kaggle/input/","")[:-5]] = df1[col] #переименоваем названия столбцов

            del df1[col]

    df = df.merge(df1, on="Unnamed: 0", how="right") #обединяем все данные в единый файл по субъектам федерации

df["Subject"] = df["Unnamed: 0"]

del df["Unnamed: 0"]

print(df) #выводим результат
years = [] #здесь фильтруем полученные значения в субъектах федерации по годам

for x in df.columns:

    if x != "Subject":

        year = x[:4]

        if not (year in years):

            years.append(year)

print(years)

    
dfs = dict()

for year in years:

    columns = df.columns

    dfs[year]=pd.DataFrame()

    dfs[year]["Subject"] = df["Subject"]

    for x in columns:

        if year in x:

            if x != "housing_per_1000":

                dfs[year][x]=df[x]/(df[year+" г.nonlocal_laborers"]+df[year+" г.loc_laborers"])

                #делаем здесь пересчет на количество работающих в субъекте федерации

            else:

                dfs[year][x]=df[x]

    dfs[year][year+" labor"] = df[year+" г.nonlocal_laborers"]+df[year+" г.loc_laborers"]

                #добавляем параметр количества работающих людей в субъекте федерации

                

    
from matplotlib import pyplot as plt

fig, ax = plt.subplots()

loc_df = dfs["2018"].copy()

del loc_df["Subject"]

matrix = loc_df.corr()

plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")

ax.set_xticks(np.arange(len(loc_df.columns)))

ax.set_yticks(np.arange(len(loc_df.columns)))

ax.set_xticklabels(loc_df.columns)

ax.set_yticklabels(loc_df.columns)

plt.imshow(matrix,cmap='coolwarm',interpolation='nearest')

plt.show()

print(matrix)



#здесь мы строим матрицу корреляции (зависмоси параметров друг от друга

#видим, то есть корреляция между количеством сданных в эксплуатацию квадратных метров домов

#и количества новорожденных
print(loc_df.columns)
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

#исследуем линейную регрессионную модель для объяснения количества сданных квадратных метров

#при помощи демографичеких показателей.

#коэффициент детерминации (объясненной дисперссии) для линейной модели около 18%.

# наиболее значимыми для объяснения количества сданного жилья с положительной корреляцией оказывается число

# рабочих работающих там же, где проживают, и обратно пропорционаально количеству рабочих, которую работают

# не там, где зарегистрированы.



X=dfs["2018"].copy()

y=dfs["2018"]["2018 г.housing_per_1000.xlsx"]

del X["Subject"]

del X["2018 г.housing_per_1000.xlsx"]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

# здесь выборка разбивается на тренировочную и тестовую

model = LinearRegression() #объявляем модель

model.fit(X,y) # обучаем модель

print(model.score(X,y)) # выводим коэффициент детерминации

print(model.coef_) # выводим коэффициенты модели

print(X.columns) # выводим названия переменных
#делаем то же самое, что и в предыдущем блоке, но для регресии с помощью алгоритма случайного леса.

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split



score = 0

X=dfs["2018"].copy()

y=dfs["2018"]["2018 г.housing_per_1000.xlsx"]

del X["Subject"]

del X["2018 г.housing_per_1000.xlsx"]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=7)

model = RandomForestRegressor(random_state=22)

model.fit(X_train,y_train)

print(model.score(X_test,y_test))

print(model.feature_importances_)

print(X.columns)



#коэффициент объясненной дисперсии теперь 81.7%, наиболе значимо

#влияющим на количество сданного жилья в пересчете кв.м. на 1000 человек 

#оказался параметр количества работающих людей. Из 81.7% он объяснеяет 71.3% дисперсии.
from sklearn.linear_model import SGDRegressor

from sklearn.model_selection import train_test_split



X=dfs["2018"].copy()

y=dfs["2018"][["2018 г.housing_per_1000.xlsx"]]

del X["Subject"]

del X["2018 г.housing_per_1000.xlsx"]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

model = SGDRegressor()

model.fit(X_train,y_train)

print(model.score(X_test,y_test))

print(model.coef_)

print(X.columns)

#стохастический градиентный регрессор показывает в данном случае несостоятельные оценки предсказания
#Пока делаем прозаический вывод, что чем абсолютно больше трудящихся в регионе, тем лучше

#там сдается в эксплутацию жилье даже в пересчете в кв.м. на 1000 человек
from sklearn.linear_model import LinearRegression

#здесь линейной регрессией будем проверять, насколько количество сданного жилья в различных годах

#влияет на количество новорожденных в настоящее время.



X=pd.read_excel("/kaggle/input/housing_per_1000.xlsx.xlsx", header=1)

X = X.dropna()

del X["Unnamed: 1"]

X["Subject"] = X["Unnamed: 0"]

del X["Unnamed: 0"]

y=dfs["2018"][["Subject", "2018 г.birth"]]

X=X.merge(y, on="Subject", how="inner")

del X["Subject"]

y = X["2018 г.birth"]

del X["2018 г.birth"]

for i in range(len(X.columns)-1):

    X[X.columns[i+1]+"/"+X.columns[i]] = X[X.columns[i+1]]/X[X.columns[i]]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

model = LinearRegression()

model.fit(X,y)

print(model.score(X,y))

print(model.coef_)

print(X.columns)