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

f = open('../input/seoul-weather/seoul.csv', 'r', encoding='cp949')

data = csv.reader(f)

header = next(data)

for row in data:

    row[-1] = float(row[-1])

    print(row)

f.close()
import csv

f = open('../input/seoul-weather/seoul.csv', 'r', encoding='cp949')

data = csv.reader(f)

header = next(data)

max_temp = -999

max_date = ''

for row in data:

    if row[-1] =='':

        row[-1] = -999

    row[-1] = float(row[-1])

    if max_temp <  row[-1]:

        max_date = row[0]

        max_tmemp = row[-1]

f.close()

print('기상 관측 이래 서울의 최고 기온이 가장 높았던 날은', max_date+'로,', max_temp, '도 였습니다')