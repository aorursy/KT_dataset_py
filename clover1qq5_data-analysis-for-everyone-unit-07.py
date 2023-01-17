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

f = open('../input/seoul-age/age.csv', encoding = 'cp949')

data = csv.reader(f)



for row in data:

    print(row)
f = open('../input/seoul-age/age.csv', encoding = 'cp949')

data = csv.reader(f)







for row in data:

    if '서울특별시 서초구 양재1동(1165065100)' ==row[0]:

        print(row)
f = open('../input/seoul-age/age.csv', encoding = 'cp949')

data = csv.reader(f)







for row in data:

    if '양재1동'in row[0]:

        print(row)
f = open('../input/seoul-age/age.csv', encoding = 'cp949')

data = csv.reader(f)







for row in data:

    if '양재1동' in row[0]:

        for i in row[3:]:

            print(i)
f = open('../input/seoul-age/age.csv', encoding = 'cp949')

data = csv.reader(f)





Q

for row in data:

    if '양재1동' in row[0]:

            print(len(row[3:]))
f = open('../input/seoul-age/age.csv', encoding = 'cp949')

data = csv.reader(f)

result=[]





for row in data:

    if '양재1동' in row[0]:

        for i in row[3:]:

            result.append(int(i))

print(result)
import matplotlib.pyplot as plt

plt.style.use('ggplot')

plt.plot(result)

plt.show