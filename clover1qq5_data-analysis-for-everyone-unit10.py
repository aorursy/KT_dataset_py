# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import csv

import matplotlib.pyplot as plt



f = open('../input/seoul-gender/gender.csv', encoding='cp949')

data = csv.reader(f)

m = []

f = []
name = input('지역 이름: ')

for row in data:

    if name in row[0]:

        for i in range(3, 104):

            m.append(int(row[i]))

            f.append(int(row[i+103]))

        break



plt.plot(m, label='Male')

plt.plot(f, label='Female')

plt.legend()

plt.show()
f = open('../input/seoul-gender/gender.csv', encoding='cp949')

data = csv.reader(f)

result =[]



name = input('지역 이름: ')

for row in data:

    if name in row[0]:

        for i in range(3, 104):

            result.append(int(row[i]) - int(row[i+103])) 

        break



plt.bar(range(101), result)

plt.show()
plt.scatter([1,2,3,4], [10,20,30,40])

plt.show()
plt.scatter([1,2,3,4], [10,20,30,40], s=[100,200,250,300])

plt.show()
plt.scatter([1,2,3,4], [10,50,30,40], s=[30,60,90,120], c=['red', 'blue', 'green', 'gold'])

plt.show()
plt.scatter([1,2,3,4], [10,50,30,40], s=[30,60,90,120], c=range(4))

plt.colorbar()

plt.show()
plt.scatter([1,2,3,4], [10,50,30,40], s=[30,60,90,120], c=range(4), cmap='jet')

plt.colorbar()

plt.show()
import random



x=[]

y=[]

size = []



for i in range(100):

    x.append(random.randint(50,100))

    y.append(random.randint(50,100))

    size.append(random.randint(10,100))

plt.scatter(x,y, s=size)

plt.show()
import random



x=[]

y=[]

size = []



for i in range(100):

    x.append(random.randint(50,100))

    y.append(random.randint(50,100))

    size.append(random.randint(10,100))

plt.scatter(x,y, s=size, c=size, cmap='jet')

plt.colorbar()

plt.show()
import random



x=[]

y=[]

size = []



for i in range(100):

    x.append(random.randint(50,100))

    y.append(random.randint(50,100))

    size.append(random.randint(10,100))

plt.scatter(x,y, s=size, c=size, cmap='jet', alpha=0.7)

plt.colorbar()

plt.show()
f = open('../input/seoul-gender/gender.csv', encoding='cp949')

data = csv.reader(f)



m=[]

f=[]



name=input('지역 이름: ')

for row in data:

    if name in row[0]:

        for i in range(3,104):

            m.append(int(row[i]))

            f.append(int(row[i+103]))

        break

plt.scatter(m, f)

plt.show()
f = open('../input/seoul-gender/gender.csv', encoding='cp949')

data = csv.reader(f)



m=[]

f=[]



name=input('지역 이름: ')

for row in data:

    if name in row[0]:

        for i in range(3,104):

            m.append(int(row[i]))

            f.append(int(row[i+103]))

        break

plt.scatter(m, f, c=range(101), alpha=0.5, cmap='jet')

plt.colorbar()

plt.plot(range(max(m)), range(max(m)), 'g')

plt.show()
import csv

import matplotlib.pyplot as plt

import math



f = open('../input/seoul-gender/gender.csv', encoding='cp949')

data = csv.reader(f)



m=[]

f=[]

size=[]





name=input('지역 이름: ')

for row in data:

    if name in row[0]:

        for i in range(3,104):

            m.append(int(row[i]))

            f.append(int(row[i+103]))

            size.append(math.sqrt(int(row[i])+int(row[i+103])))

        break

plt.style.use('ggplot')

plt.rc('font', family='Malgun Gothic')

plt.figure(figsize=(10,5), dpi=100)

plt.title(name+' 성별 인구 그래프')

plt.scatter(m, f, c=range(101), alpha=0.5, cmap='jet')

plt.colorbar()

plt.plot(range(max(m)), range(max(m)), 'g')

plt.xlabel('남성 인구수')

plt.ylabel('여성 인구수')

plt.show()