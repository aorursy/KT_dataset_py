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
f = open('../input/subwayfee/subwayfee.csv')
data = csv.reader(f)

for row in data:
    print(row)
import csv 
f = open('../input/subwayfee/subwayfee.csv')
data = csv.reader(f)
next(data)

for row in data:
    print(row)
import csv 

f = open('../input/subwayfee/subwayfee.csv')
data = csv.reader(f)
next(data)

mx=0
rate=0
mx_station = ''

for row in data:
    for i in range(4,8):
        row[i] = int(row[i])
    if rate[6] != 0 and (row[4]+row[6]) >100000:
        rate = row[4] / (row[4]+row[6])
        if rate > mx:
            mx=rate
            mx_station = row[3] + '' + row[1]
            
print(mx_station, round(mx*100, 2))
import csv
f = open('../input/subwayfee/subwayfee.csv')
data = csv.reader(f)
next(data)
mx = [0] * 4
mx_station = [''] * 4
label = ['유임승차','유임하차','무임승차','무임하차']
for row in data :
    for i in range(4,8) :
        row[i] = int(row[i])
        if row[i] > mx[i-4] :
            mx[i-4] = row[i]
            mx_station[i-4] = row[3] + ' ' + row[1]
for i in range(4) :
    print(label[i] + ' : ' + mx_station[i], mx[i])
import csv
import matplotlib.pyplot as plt
f = open('../input/subwayfee/subwayfee.csv')
data = csv.reader(f)
next(data)
label = ['유임승차','유임하차','무임승차','무임하차']
for row in data :
    for i in range(4,8) :
        row[i] = int(row[i])
    #plt.figure(dpi = 300)
    plt.pie(row[4:8])
    plt.axis('equal')
    plt.show()
import csv
import matplotlib.pyplot as plt
f = open('../input/subwayfee/subwayfee.csv')
data = csv.reader(f)
next(data)
label = ['유임승차','유임하차','무임승차','무임하차']
c = ['#14CCC0', '#389993', '#FF1C6A', '#CC14AF']
plt.rc('font', family = 'Malgun Gothic')
for row in data :
    for i in range(4,8) :
        row[i] = int(row[i])
    plt.figure(dpi = 300)
    plt.title(row[3] + ' ' + row[1])
    plt.pie(row[4:8], labels = label, colors = c, autopct = '%1.f%%')
    plt.axis('equal')
    plt.show()
import csv
import matplotlib.pyplot as plt
f = open('../input/subwayfee/subwayfee.csv')
data = csv.reader(f)
next(data)
label = ['유임승차','유임하차','무임승차','무임하차']
c = ['#14CCC0', '#389993', '#FF1C6A', '#CC14AF']
plt.rc('font', family = 'Malgun Gothic')
for row in data :
    for i in range(4,8) :
        row[i] = int(row[i])
    plt.figure(dpi = 300)
    plt.title(row[3] + ' ' + row[1])
    plt.pie(row[4:8], labels = label, colors = c, autopct = '%1.f%%')
    plt.axis('equal')
    plt.savefig(row[3] + ' ' + row[1] + '.png')
    plt.show()