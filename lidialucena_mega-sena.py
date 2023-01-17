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
Mega = pd.read_csv('/kaggle/input/sorteiosmegasena/sorteios.csv')



Mega.head(5)
print('Mega-Sena:', Mega.shape)
Mega.info()
Mega = Mega.drop('Unnamed: 22',axis=1)

print(Mega.head(5))
Mega.head(5)
Mega = Mega.drop('Id',axis=1)

Mega.head(5)
Mega[Mega.duplicated()]
Mega.drop_duplicates(inplace=True)



Mega.info()
Mega['data_convertida'] = Mega.iloc[:,1]

Mega.data_convertida = pd.to_datetime(Mega.data_convertida)



Mega.head(5)
Mega['Dia']   = Mega.data_convertida.dt.day

Mega['Mês'] = Mega.data_convertida.dt.month 

Mega['Ano']  = Mega.data_convertida.dt.year



Mega.head(5)



Mega.set_index('Ano')
Formatar=['Arrecadacao_Total','Rateio_Sena','Rateio_Quina','Rateio_Quadra','Valor_Acumulado']



for col in Formatar:

    Mega[col] = Mega[col].str.replace(".", "",regex=True).str.replace(",", ".")

    

Mega[Formatar] = Mega[Formatar].fillna(0).astype(float)

assert Mega[Formatar].dtypes.all() == np.float64

Mega.dtypes
Mega.groupby('Mês')['Ganhadores_Sena'].describe()
Mega.nlargest(1, 'Ganhadores_Sena')
Mega.nsmallest(11, 'Arrecadacao_Total')
import seaborn as sns

import matplotlib.pyplot as plt
plt.figure(figsize=(15,5))

sns.boxplot(Mega['Ano'], Mega['Ganhadores_Sena'])

plt.title('Média de Ganhadores')

plt.xticks(rotation=90)

plt.locator_params(axis='y', nbins=20)

plt.show()
plt.figure(figsize=(15,5))

sns.pointplot(x='Valor_Acumulado', y='Ano', data=Mega, color='green')

plt.title('Rateio da Mega-Sena por ano')

plt.grid(True, color='grey')
sns.pairplot(x_vars='Ano', y_vars='Arrecadacao_Total', data=Mega, size=10, hue='Mês')
sns.stripplot(x='UF', y='Ganhadores_Sena', data=Mega)

plt.xticks(rotation=90)
Dezenas = pd.DataFrame(Mega['1ª Dezena'].tolist() + Mega['2ª Dezena'].tolist() + Mega['3ª Dezena'].tolist() + Mega['4ª Dezena'].tolist() + Mega['5ª Dezena'].tolist() + Mega['6ª Dezena'].tolist(), columns=['numeros'])

Dezenas['numeros'].value_counts().sort_values(ascending=False).head(10).plot(kind='barh', title='Dezenas mais sorteadas', figsize=(10,5), fontsize=12, legend=True, color='gray')