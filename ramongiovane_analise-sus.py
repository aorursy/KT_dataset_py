# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



print(10)

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/LibreOffice Writer.csv')

df

df.columns = ['dataHora', 'nome', 'p1', 'p2', 'p3','p4','p5','p6', 'p7','p8','p9','p10']

def limparCelula(x):

    try:

        return int(x[0])

    except:

        return x
def limparTabela():

    

    for x in range(df.shape[1] - 2):

        i = 'p' + str(x + 1)

        print(i)

        df[i] = df[i].apply(limparCelula)
limparTabela()

df
def estatisticas(col):

    return col.min(),  col.max(),col.mean(),  col.std()
    

df['total'] = df.apply(calcula_total, axis=1)

df


for x in df.columns[2 :]:

    print (x)

    print(estatisticas(df[x]))

    print()