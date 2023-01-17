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



f=open('../input/seoul-gender/gender.csv', encoding='cp949')

data = csv.reader(f)



m = []

f = []

name = input('지역 이름: ')



for row in data:

    if name in row[0]:

        for i in row[3:104]:

            m.append(-int(i)) 

        for i in row[106:]:

            f.append(int(i))



plt.style.use('ggplot')

plt.figure(figsize=(10,5), dpi=100)

plt.rc('font', family='Malgun Gothic')

plt.rcParams['axes.unicode_minus'] =False

plt.title(name+ ' 남녀 성별 인구 분포')

plt.barh(range(101), m, color='skyblue', label='남성')

plt.barh(range(101), f, color= 'pink', label='여성')

plt.legend()

plt.show()       
import csv

import matplotlib.pyplot as plt



f=open('../input/seoul-gender/gender.csv', encoding='cp949')

data = csv.reader(f)



m = []

f = []

name = input('지역 이름: ')



for row in data:

    if name in row[0]:

        for i in row[3:104]:

            m.append(-int(i)) 

        for i in row[106:]:

            f.append(int(i))

        break



plt.style.use('ggplot')

plt.figure(figsize=(10,5), dpi=100)

plt.rc('font', family='Malgun Gothic')

plt.rcParams['axes.unicode_minus'] =False

plt.title(name+ ' 남녀 성별 인구 분포')

plt.barh(range(101), m, color='skyblue', label='남성')

plt.barh(range(101), f, color= 'pink', label='여성')

plt.legend()

plt.show()       
plt.pie([10,20])

plt.show()
size = [1, 2, 3, 4]

plt.axis('equal')

plt.pie(size)

plt.show()
plt.rc('font', family='Malgun Gothic')

size = [1, 2, 3, 4]

label = ['A형', 'B형', 'O형', 'AB형']

plt.axis('equal')

plt.pie(size, labels=label)

plt.show()
plt.rc('font', family='Malgun Gothic')

size = [1, 2, 3, 4]

label = ['A형', 'B형', 'O형', 'AB형']

plt.axis('equal')

plt.pie(size, labels=label, autopct='%.1f%%')

plt.legend()

plt.show()
plt.rc('font', family='Malgun Gothic')

size = [1, 2, 3, 4]

label = ['A형', 'B형', 'O형', 'AB형']

color = ['darkmagenta', 'deeppink', 'hotpink', 'pink']

plt.axis('equal')

plt.pie(size, labels=label, autopct='%.1f%%', colors=color, explode=(0,0,0.4,0))

plt.legend()

plt.show()
f = open('../input/seoul-gender/gender.csv', encoding='cp949')

data = csv.reader(f)

size=[]

name=input('지역 이름: ')



for row in data:

    if name in row[0]:

        m = 0

        f = 0

        for i in range(101):

            m += int(row[i+3])

            f += int(row[i+106])

        break



size.append(m)

size.append(f)

print(size)
plt.rc('font', family = 'Malgun Gothic')

color = ['crimson', 'darkcyan']

plt.axis('equal')

plt.pie(size, labels=['남', '여'], autopct = '%.1f%%', colors=color, startangle=90)

plt.title(name+' 지역의 남녀 성별 비율')

plt.show()
import csv



f = open('../input/seoul-gender/gender.csv', encoding='cp949')

data = csv.reader(f)

size=[]

name=input('지역 이름: ')



for row in data:

    if name in row[0]:

        m = 0

        f = 0

        for i in range(101):

            m += int(row[i+3])

            f += int(row[i+106])

        break



size.append(m)

size.append(f)



plt.rc('font', family = 'Malgun Gothic')

color = ['lightblue', 'hotpink']

plt.axis('equal')



plt.pie(size, labels=['남', '여'], autopct = '%.1f%%', colors=color, startangle=90)

plt.title(name+' 지역의 남녀 성별 비율')

plt.show()