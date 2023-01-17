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
import matplotlib.pyplot as plt

plt.hist([1,1,2,3,4,5,6,7,8,10,6])
import random

print(random.randint(1, 6))
import random

dice = []

for i in range(5):

    dice.append(random.randint(1,6))

print(dice)
import random

dice = []

for i in range(5):

    dice.append(random.randint(1,6))

plt.hist(dice, bins=6)

plt.show()
import random

dice = []

for i in range(1000000):

    dice.append(random.randint(1,6))

plt.hist(dice, bins=6)

plt.show()
import csv

f = open('../input/seoul-weather/seoul.csv', encoding='cp949')

data = csv.reader(f)

next(data)

result=[]



for row in data:

    if row[-1]!= '':

        result.append(float(row[-1]))

        

plt.hist(result, bins=100, color='r')

plt.show()


f = open('../input/seoul-weather/seoul.csv', encoding='cp949')

data = csv.reader(f)

next(data)

aug=[]



for row in data:

    month = row[0].split('-')[1]

    if row[-1] != '':

        if month == '08':

            aug.append(float(row[-1]))

    

    

plt.hist(aug, bins=100, color='r')

plt.show()


f = open('../input/seoul-weather/seoul.csv', encoding='cp949')

data = csv.reader(f)

next(data)

aug=[]

jan=[]



for row in data:

    month = row[0].split('-')[1]

    if row[-1] != '':

        if month == '08':

            aug.append(float(row[-1]))

        if month == '01':

            jan.append(float(row[-1]))

    

    

plt.hist(aug, bins=100, color='r', label='aug')

plt.hist(jan, bins=100, color='blue', label='jan')

plt.legend()

plt.show()
result = []

for i in range(13):

    result.append(random.randint(1, 1000))

print(sorted(result))



plt.boxplot(result)

plt.show()
result = []

for i in range(13):

    result.append(random.randint(1, 1000))

print(result)



plt.boxplot(result)

plt.show()
f = open('../input/seoul-weather/seoul.csv', encoding = 'cp949')

data = csv.reader(f)

next(data)

result = []



for row in data:

    if row[-1]!= '':

        result.append(float(row[-1]))

        

plt.boxplot(result)

plt.show()
f = open('../input/seoul-weather/seoul.csv', encoding = 'cp949')

data = csv.reader(f)

next(data)

aug=[]

jan=[]



for row in data:

    month = row[0].split('-')[1]

    if row[-1]!= '':

        if month == '08':

            aug.append(float(row[-1]))

        if month == '01':

            jan.append(float(row[-1]))

        

plt.boxplot(aug)

plt.boxplot(jan)

plt.show()
plt.boxplot([aug,jan])
f = open('../input/seoul-weather/seoul.csv', encoding='cp949')

data = csv.reader(f)

next(data)

month = [[],[], [], [], [], [], [], [], [], [], [], []]



for row in data:

    if row[-1] != '':

        month[int(row[0].split('-')[1])-1].append(float(row[-1]))

plt.boxplot(month)

plt.show()
f = open('../input/seoul-weather/seoul.csv', encoding='cp949')

data = csv.reader(f)

next(data)

day = []

for i in range(31):

    day.append([])



for row in data:

    if row[-1] != '':

        if row[0].split('-')[1] =='08':

            day[int(row[0].split('-')[2])-1].append(float(row[-1]))



        

plt.style.use('ggplot')

plt.figure(figsize=(10,5), dpi=300)

plt.boxplot(day, showfliers=False)



plt.show()