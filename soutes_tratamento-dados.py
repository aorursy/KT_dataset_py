# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dataset = pd.read_csv("/kaggle/input/callcenter-mkt/callcenter_marketing.csv")
dataset.head()
dataset.shape
dataset.isnull().sum()
#Aqui visualizamos como estão distribuídos os dados originais:

dataset['emprestimo_pessoal'].value_counts()
df_ep_mode = dataset['emprestimo_pessoal']
df_ep_porcentagem = dataset['emprestimo_pessoal']
df_ep_del = dataset['emprestimo_pessoal']
dataset['emprestimo_pessoal'].value_counts()/(dataset.shape[0]-990)
def define_pessoal(valor):

    numero = np.random.randint(1,100)    

    if pd.isnull(valor):

        if numero<=85:

            return 'nao'

        else:

            return 'sim'

    else:

        return valor

#Chamamos o resultado da função e aplicamos no dataset, confirmando a alteração.

df_ep_porcentagem = df_ep_porcentagem.apply(define_pessoal)
df_ep_porcentagem.value_counts()
df_ep_porcentagem.isnull().sum()
df_ep_mode.fillna(df_ep_mode.mode()[0], inplace=True)

df_ep_mode.value_counts()
df_ep_mode.isnull().sum()
df_ep_del.dropna(inplace=True)
df_ep_del.isnull().sum()
df_ec = dataset['estado_civil']
df_ec.value_counts()
df_ec.isnull().sum()
df_ec.fillna('não definido', inplace=True)
df_ec.value_counts()