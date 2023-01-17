import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



%matplotlib inline
df = pd.read_csv('../input/contrib-ativos-uf.csv',  encoding = "ISO-8859-1")



df.shape
df.head(5)
df.tail(5)
df = df.drop(df[df['Ano'] == '-'].index)



print(df.shape)
anos = np.array(df['Ano'])

anos = np.unique(anos)

anos = anos.astype(int)
df_index = df.columns[2]

contribuintes = []



for ano in anos:

    valores = [int(df.loc[l][df_index]) 

                 for l in range(0, df.shape[0]) 

                 if int(df.loc[l]['Ano']) == ano]

    

    contribuintes.append(np.sum(valores))
fig, ax = plt.subplots()



ax.plot(anos, contribuintes)



ax.set(xlabel='Ano', ylabel='Quantidade (milhões)',

       title='Contribuintes empregados por ano')



ax.grid()



fig.set_size_inches(18.5, 10.5)

plt.show()