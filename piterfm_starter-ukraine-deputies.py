import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # for plotting
import matplotlib.pyplot as plt # for plotting
import datetime

import os
print(os.listdir("../input"))
data = pd.read_csv('../input/ukraine_deputies.csv')
data.head()
data.columns
data.shape
data.info()
data['Education'] = (data['Alma mater'].fillna('')+data['Освіта'].fillna('')).replace(r'', np.nan)
data.drop(['Alma mater', 'Освіта'], axis=1, inplace=True)
data['Start Work'].unique()
data['End Work'].unique()
today = datetime.datetime.now().strftime('%d.%m.%Y')
data['End Work'].fillna(today, inplace=True)
data['End Work'] = data['End Work'].str.replace(r'\[\d*\]| \S*', '')
today
data['WorkEnd'] = data[data['Rada']==8].apply(lambda x: '-'.join(x['End Work'].split('.')[::-1]), axis=1)
data['WorkEnd'] = data['WorkEnd'].fillna(data[data['Rada']<8]['End Work'])
data['WorkEnd'] = pd.to_datetime(data['WorkEnd']).dt.date
data.drop(['End Work'], axis=1, inplace=True)
data['WorkStart'] = data[data['Rada']==8].apply(lambda x: '-'.join(x['Start Work'].split('.')[::-1]), axis=1)
data['WorkStart'] = data['WorkStart'].fillna(data[data['Rada']<8]['Start Work'])
data['WorkStart'] = pd.to_datetime(data['WorkStart']).dt.date
data.drop(['Start Work'], axis=1, inplace=True)
data['WorkPeriod'] = (data['WorkEnd']-data['WorkStart']).dt.days
data.info()

