 #import pandas

import pandas as pd

import os

#print plot

import matplotlib.pyplot as plt

import seaborn as sns

 
#all files

print(os.listdir('../input'))

#filename 

my_filepath = "../input/steam.csv"

#read this file

steamTable = pd.read_csv(my_filepath)

 

 

#steamTable.dataframeName = 'output.csv'

nRow, nCol = steamTable.shape

print(f'В таблице {nRow} строк и {nCol} колонок')

 

#grouping by  average_playtime (variant 9)

argregateTable = steamTable.groupby(['average_playtime','name']).count()

#show

argregateTable

#print plow

plt.figure(figsize=(12,6))

sns.lineplot(data=steamTable['average_playtime'])



 

#Создаем csv файл

argregateTable.to_csv("output.csv", index=True)