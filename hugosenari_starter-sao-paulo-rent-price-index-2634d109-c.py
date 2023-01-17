import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

def plotMonthsAfter(df, min_year):
    fig, ax = plt.subplots(figsize=(12,6))
    ax.set(xlabel='Mês', ylabel='ZAP FIPE',
       title='Ano a Ano do Índice ZAP FIPE')
    for year, data in df.groupby(by='Ano', axis=0):
        if year > min_year:
            data.plot(kind="line", x='Mês', y='Variação Mensal', ax=ax, label=year)
    plt.show()


nRowsRead = 1000 # specify 'None' if want to read whole file
df1 = pd.read_csv('../input/arquivo-fipezap-SAOPAULO-TodoPeriodo-locacao-quartos2-0-20180917.csv', delimiter=',', nrows = nRowsRead)
df1.dataframeName = 'arquivo-fipezap-SAOPAULO-TodoPeriodo-locacao-quartos2-0-20180917.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')
plotMonthsAfter(df1, 2013)