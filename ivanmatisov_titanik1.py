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

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/titanic/train.csv")
data.head()
data.describe() #предварительный анализ (числовые столбцы)
data["Sex"].value_counts() #посчитать значение

total = data.shape[0]

print(total) #Общее количество человек на Титанике
n_alive, alive = data["Survived"].value_counts() #не выжил, выжил (если мы знаем порядок)

p_alive = alive*100/total #вычисляем процент выживших на общее число человек

print(n_alive, alive) # первое - не выжили, вторые - выжили

print(np.round(p_alive)) #процент 
fc_pass = data["Pclass"].value_counts()[1]

p_fcclass = fc_pass*100/total

print(np.round(p_fcclass, 2)) #округление до 2х знаков
data["Pclass"].hist(); #гистограмма, в скобках пишем дополнительные параметры для оформления
print(data["Age"].mean())

print(data["Age"].median()) #посчитать 1.среднее значение и 2.медиану
data["SibSp"].corr(data["Parch"]) #корреляция Пирсона (то есть зависимость/связь)
numeric = data[["PassengerId","Survived","Pclass","Age","SibSp","Parch", "Fare"]].corr()
import seaborn as sns #дополнительная бибилиотека

%matplotlib inline
sns.heatmap(numeric, annot=True); #тепловая карта, нас интересует

                    #только диапазон цвета(чем светлее тем боле скоррелированы, а значит есть связь)
numeric[numeric>0.2] #это фильтр
names = data[data["Sex"]=="female"]["Name"]
def find_name(name):

    if "Miss." in name:

        lst = name.split(" ")

        idx = lst.index("Miss.")

        return lst [idx+1]

    if "(" in name:

        idx = name.find("(")

        return name[idx+1:-1].split(" ")[0]
names.apply(find_name).value_counts()