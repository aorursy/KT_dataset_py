# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from pandas import Series,DataFrame

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

#print(check_output(["ls", "../input/train.csv"]).decode("utf8"))

data_train = pd.read_csv("../input/train.csv")

data_train.describe()



fig = plt.figure()

fig.set(alpha=0.2)  # 设定图表颜色alpha参数



plt.subplot2grid((2,3),(0,0))             # 在一张大图里分列几个小图

data_train.Survived.value_counts().plot(kind='bar')# 柱状图 

plt.title(u"Survived?(1 for yes)") # 标题

plt.ylabel(u"Sum(person)")  



plt.subplot2grid((2,3),(0,1))

data_train.Pclass.value_counts().plot(kind="bar")

plt.ylabel(u"Sum(person)")

plt.title(u"Passenger Level")



plt.subplot2grid((2,3),(0,2))

plt.scatter(data_train.Survived, data_train.Age)

plt.ylabel(u"Age")                         # 设定纵坐标名称

plt.grid(b=True, which='major', axis='y') 

plt.title(u"Age on survive-stastic(1 for Survive)")





plt.subplot2grid((2,3),(1,0), colspan=2)

data_train.Age[data_train.Pclass == 1].plot(kind='kde')   

data_train.Age[data_train.Pclass == 2].plot(kind='kde')

data_train.Age[data_train.Pclass == 3].plot(kind='kde')

plt.xlabel(u"Age")# plots an axis lable

plt.ylabel(u"midu") 

plt.title(u"Age on Class")

plt.legend((u'1st Class', u'2nd Class',u'3rd Class'),loc='best') # sets our legend for our graph.





plt.subplot2grid((2,3),(1,2))

data_train.Embarked.value_counts().plot(kind='bar')

plt.title(u"BoardToShip")

plt.ylabel(u"Sum(person)")  

plt.show()

# Any results you write to the current directory are saved as output.