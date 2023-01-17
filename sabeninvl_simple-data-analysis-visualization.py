import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
data = pd.read_csv('../input/big_data.csv', index_col = 0)

data.head()
#Разбивка каждой строки из Model на множество, перевод в целочисленное значение и распределение по новым столбцам

data['L1'] = data['Model'].apply(lambda x: int(x.split('/')[0]))

data['L2'] = data['Model'].apply(lambda x: int(x.split('/')[1]))

data['L3'] = data['Model'].apply(lambda x: int(x.split('/')[2]))

data['Batch_Size'] = data['Model'].apply(lambda x: int(x.split('/')[3].split('B_S')[-1]))

data['Epochs'] =data['Model'].apply(lambda x: int(x.split('/')[4].split('Ep')[-1]))
data.head()
#Группировка. Мультииндекс. Сортировка по Score от большего.

data.groupby(['L1','L2','L3','Batch_Size','Epochs']).mean()#.sort_values(by = ['Score'], ascending = False)
#Визуализация зависимостей

data.groupby(['L1','L2','L3','Batch_Size','Epochs']).mean().Score.unstack().plot(kind='line',figsize=(20,15), table=True, marker='o', linestyle='-')

plt.show()
#Визуализация зависимостей

data.groupby(['L1','L2','L3','Epochs','Batch_Size']).mean().val_loss.unstack().plot(kind='line',figsize=(20,15), table=True, marker='o', linestyle='-')

plt.show()