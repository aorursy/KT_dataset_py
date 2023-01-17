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
import pandas as pd
data = pd.read_csv('/kaggle/input/globses/GLOB.SES.csv', encoding = 'ISO-8859-1')

print('FIRST 5 AND LAST \n',data.head(),data.tail(), '\n') # первые 5 и последние 5 значений

print('STATISTIC \n',data.describe(), '\n') # статистика

print('SOCIAL ECONOMIC SCORE \n' ,data['SES'], '\n') # social economic score

print('LAST SCORES > 24 \n', data[data['SES']>24].tail(), '\n')

print('LAST 5 OF gdppc **2 \n', data['gdppc'].apply(lambda d: d**2).head(), '\n') # последние 5 значений из gdppc, помноженные на себя

print('MAXIMUM SCORE IN SES \n', data['SES'].max(), '\n') # максимальное значение из SES

print('SLICE 5:9 \n', data[5:9], '\n') # срез 5:9

print('SORTED \n', data.sort_values(by = 'SES', ascending = True, inplace = False).tail()) # последние 5 значений из отсортированного SES
print('FIRST 5 NaNs IN yrseduc \n', data[data['yrseduc'].isnull()].head(), '\n') # первые 5 NaN значений из yrseduc
print('FIRST 5 SCORES IN yrseduc WITHOUT NaNs \n', data[data['yrseduc'].notnull()].head(), '\n') # первые 5 значений из yrseduc, в котором убрали NaN
print('FIRST 5 SCORES IN yrseduc WITHOUT NaNs (USING dropna())\n', data.dropna(how = 'any', subset = ['yrseduc']).head(), '\n') # то же самое
print('FIRST 5 SCORES IN yrseduc WITH FILLED NaNs \n',data.fillna(value = {'yrseduc': 0}).head()) # заполнить NaNы нулевыми значениями, вывести первые 5 из полученной выборки
print('FIRST 5 GROUPED DATA BY SES \n',data[['country','SES']].groupby('country').mean().head())
print('FILTRED DATA \n', pd.pivot_table(data[data['year']>2007], values = 'SES', index = ['country'], columns = ['year']).head(), '\n')
print('USING ix OPERATOR \n', pd.pivot_table(data[data['year']>2000], values = 'SES', index = ['country'], columns = ['year']).loc[['Norway', 'New Zealand'],[2010]])#в новой версии pandas ix удалили, поскольку на протяжении всего его существования использование осуждалось
print('RANGED \n',pd.pivot_table(data[data['year']>1980], values = 'SES', index = ['country'], columns = ['year']).rank(ascending = False, method = 'first').head(), '\n') # ранжирование
print('RANGED WITH TOTAL SUM \n', pd.pivot_table(data[data['year']>1980], values = 'SES', index = ['country'], columns = ['year']).sum(axis = 1).rank(ascending = False, method = 'dense').head())
pd.pivot_table(data[data['SES']>90], values = 'SES', index = ['country'], columns = ['year']).sum(axis = 1).sort_values(ascending = False).plot(kind = 'bar', style = 'b', alpha = 0.4, title = 'TOTAL VALUES FOR COUNTRY (BY SES)')
my_colors = ['b', 'g', 'r', 'y', 'm', 'c']
pd.pivot_table(data[data['SES']>90], values = 'SES', index = ['country'], columns = ['year']).sum(axis = 1).sort_values(ascending = False).plot(kind = 'barh', stacked = True, color = my_colors)