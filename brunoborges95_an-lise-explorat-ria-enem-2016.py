# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



#importando as bibliotecas que deverão ser utilizadas

import pandas as pd #manipulação de dados 

import numpy as np #manipulação de arrays

import matplotlib.pyplot as plt #visualização

import seaborn as sns #viualozação

from sklearn.ensemble import RandomForestRegressor #Random Forest para detectar importância de features

from sklearn.preprocessing import LabelEncoder #codifocação de dados categóricos em dados inteiros



# Input data files are available in the "../input/" directory.

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory





# Any results you write to the current directory are saved as output.
colunas = ['SG_UF_RESIDENCIA','TP_SEXO','TP_COR_RACA','TP_ESCOLA','Q001','Q002', 'Q006', 'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 

          'NU_NOTA_MT', 'NU_NOTA_REDACAO']

base = pd.read_csv('/kaggle/input/enem-2016/microdados_enem_2016_coma.csv', encoding='latin-1', 

                 sep=',', usecols=colunas)
total_null = base.isnull().sum().sort_values(ascending = False)

base.info()



# Elimina as linhas com nota média NaN

base.dropna(inplace=True)



print(len(base))
# Cria nota média

base['NOTA_MEDIA'] = (

    base['NU_NOTA_CN'] + base['NU_NOTA_CH'] + base['NU_NOTA_LC'] + base['NU_NOTA_MT'] + 

    base['NU_NOTA_REDACAO']) / 5.0
#nota de corte para os melhores alunos

nota_corte = base.NOTA_MEDIA.nlargest(round(len(base)*0.05)).min()

base_melhores_notas = base[base.NOTA_MEDIA >= nota_corte]

print(base_melhores_notas)
notas_medias={'ciências da natureza':np.mean(base['NU_NOTA_CN']),

                                               'Ciências Humanas':np.mean(base['NU_NOTA_CH']),

                                               'Linguagens, Códigos':np.mean(base['NU_NOTA_LC']),

                                               'Matemática':np.mean(base['NU_NOTA_MT']),

                                               'Redação':np.mean(base['NU_NOTA_REDACAO'])}

lists = sorted(notas_medias.items()) 



x1, y1 = zip(*lists) 

plt.figure(figsize=(10,6))



plt.bar(x1, y1)

plt.title('Nota média por Disciplina')

plt.xlabel('Disciplina')

plt.ylabel('Nota')

plt.show()



notas_medias_melhores = {'Ciências da natureza':np.mean(base_melhores_notas['NU_NOTA_CN']),

                                               'Ciências Humanas':np.mean(base_melhores_notas['NU_NOTA_CH']),

                                               'Linguagens, Códigos':np.mean(base_melhores_notas['NU_NOTA_LC']),

                                               'Matemática':np.mean(base_melhores_notas['NU_NOTA_MT']),

                                               'Redação':np.mean(base_melhores_notas['NU_NOTA_REDACAO'])}



lists = sorted(notas_medias_melhores.items())



x2, y2 = zip(*lists)

plt.figure(figsize=(10,6))



plt.bar(x2, y2)

plt.title('Melhores Alunos - Nota média por Disciplina')

plt.xlabel('Disciplina')

plt.ylabel('Nota')

plt.show()
# Histograma das médias



plt.axes(sns.distplot(base.NOTA_MEDIA))

plt.title('Histograma de média das notas')

plt.xlabel('Nota Média')
#numero de alunos por estado

base.SG_UF_RESIDENCIA.value_counts().plot.bar(color='green')

plt.title('Número de alunos por estado')

plt.xlabel('Estado')

plt.ylabel('Número de alunos')

plt.show()



 #Melhores alunos - numero de alunos por estado

base_melhores_notas.SG_UF_RESIDENCIA.value_counts().plot.bar(color='green')

plt.title('Melhores Alunos - Número de alunos por estado')

plt.xlabel('Estado')

plt.ylabel('Número de alunos')

plt.show()
plt.figure(figsize=(12,8))



sns.barplot(

    x="SG_UF_RESIDENCIA",

    y="NOTA_MEDIA",

    data=base, 

    )

plt.title("Distribuição de notas por estado")

plt.xlabel('Estado')

plt.ylabel('Nota Média')

plt.show()
#numero por etnia

base_TP_COR_RACA = pd.Series(base.TP_COR_RACA, dtype='category').cat.rename_categories({3:'Parda', 2:'Preta', 1:'Branca', 0:'Não Declarado', 4:'Amarela', 5:'Indígena', 6:'Não dispor'})

base_TP_COR_RACA.value_counts().plot.bar(color='blue')

plt.title('Numero de alunos por etnia')

plt.xlabel('Etnia')

plt.ylabel('Número de alunos')

plt.show()



#Melhores alunos - numero por etnia

base_TP_COR_RACA_m = pd.Series(base_melhores_notas.TP_COR_RACA, dtype='category').cat.rename_categories({3:'Parda', 2:'Preta', 1:'Branca', 0:'Não Declarado', 4:'Amarela', 5:'Indígena', 6:'Não dispor'})

base_TP_COR_RACA_m.value_counts().plot.bar(color='blue')

plt.title('Numero de alunos por etnia')

plt.xlabel('Etnia')

plt.ylabel('Número de alunos')

plt.show()
#notas por Etnia

base['TP_COR_RACA_c'] = base_TP_COR_RACA

sns.boxplot(x = 'TP_COR_RACA_c', y = 'NOTA_MEDIA', data = base)

plt.title("Nota média por raça")

plt.xlabel('Etnia')

plt.ylabel('Nota Média')
 # distribuição do número de alunos por nível de renda

base.Q006.value_counts().plot.bar(color='blue')

sns.set_style("whitegrid") 

plt.title('Número de alunos por renda')

plt.xlabel('Classe social')

plt.ylabel('Número de alunos')

plt.show()



 # Melhores alunos - distribuição do número de alunos por nível de renda

base_melhores_notas.Q006.value_counts().plot.bar(color='blue')

sns.set_style("whitegrid") 

plt.title('Melhores Alunos - Número de alunos por renda')

plt.xlabel('Classe social')

plt.ylabel('Número de alunos')

plt.show()
#nota média por renda

import string

sns.set_style("whitegrid") 

sns.boxplot(x = 'Q006', y = 'NOTA_MEDIA', data = base,

           order=list(string.ascii_uppercase[:17]))

plt.title('Nota média por renda')

plt.xlabel('Classe social')

plt.ylabel('Nota Média')

plt.show()

 #numero de alunos por sexo

base.TP_SEXO.value_counts().plot.bar(color='green')

plt.title('Número de alunos por gênero')

plt.xlabel('Estado')

plt.ylabel('Número de alunos')

plt.show()



 #Melhores alunos - numero de alunos por sexo

base_melhores_notas.TP_SEXO.value_counts().plot.bar(color='green')

plt.title('Melhores Alunos - Número de alunos por gênero')

plt.xlabel('Estado')

plt.ylabel('Número de alunos')

plt.show()
#notas por sexo

sns.boxplot(x = 'TP_SEXO', y = 'NOTA_MEDIA', data = base)

plt.title("Nota média por sexo")

plt.xlabel('Sexo')

plt.ylabel('Nota Média')
base_mulher = base.where(base['TP_SEXO'] == 'F')

base_homem = base.where(base['TP_SEXO'] == 'M')

notas_medias_mulheres = {'ciências da natureza':np.mean(base_mulher['NU_NOTA_CN']),

                                               'Ciências Humanas':np.mean(base_mulher['NU_NOTA_CH']),

                                               'Linguagens, Códigos':np.mean(base_mulher['NU_NOTA_LC']),

                                               'Matemática':np.mean(base_mulher['NU_NOTA_MT']),

                                               'Redação':np.mean(base_mulher['NU_NOTA_REDACAO'])}



lists = sorted(notas_medias_mulheres.items()) 



x1, y1 = zip(*lists) 

plt.figure(figsize=(10,6))



plt.bar(x1, y1)

plt.title('Mulheres - Nota média por Disciplina')

plt.xlabel('Disciplina')

plt.ylabel('Nota')

plt.show()





notas_medias_homens = {'ciências da natureza':np.mean(base_homem['NU_NOTA_CN']),

                                               'Ciências Humanas':np.mean(base_homem['NU_NOTA_CH']),

                                               'Linguagens, Códigos':np.mean(base_homem['NU_NOTA_LC']),

                                               'Matemática':np.mean(base_homem['NU_NOTA_MT']),

                                               'Redação':np.mean(base_homem['NU_NOTA_REDACAO'])}



lists = sorted(notas_medias_homens.items()) 



x1, y1 = zip(*lists) 

plt.figure(figsize=(10,6))



plt.bar(x1, y1)

plt.title('Homens - Nota média por Disciplina')

plt.xlabel('Disciplina')

plt.ylabel('Nota')

plt.show()
#random forest com 1600 árvores.

rfr = RandomForestRegressor(n_estimators = 1600,

 min_samples_split = 2,

 min_samples_leaf = 1,

 max_features = 'sqrt',

 max_depth = 100,

 bootstrap = False)

from sklearn.preprocessing import LabelEncoder

LE = LabelEncoder()

#como a base é muito grande, para acelerar o processamento e evitar falta de memória, vamos amostrar 1% da base,

base2 = base.sample(n=round(len(base)*0.01), random_state=42)

base2['SG_UF_RESIDENCIA'] = LE.fit_transform(base2['SG_UF_RESIDENCIA'])

base2['TP_SEXO'] = LE.fit_transform(base2['TP_SEXO'])

base2['Q001'] = LE.fit_transform(base2['Q001'])

base2['Q002'] = LE.fit_transform(base2['Q002'])

base2['Q006'] = LE.fit_transform(base2['Q006'])

rfr.fit(base2.drop(columns=['NOTA_MEDIA','NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 

          'NU_NOTA_MT', 'NU_NOTA_REDACAO','TP_COR_RACA_c']),base2.NOTA_MEDIA)



features = ['SG_UF_RESIDENCIA','TP_SEXO','TP_COR_RACA','TP_ESCOLA','Q001','Q002', 'Q006']

tmp = pd.DataFrame({'Feature': features, 'Importância da feature': rfr.feature_importances_})

tmp = tmp.sort_values(by='Importância da feature',ascending=False)

plt.figure(figsize = (7,4))

plt.title('Importância da feature - Random Forest',fontsize=14)

s = sns.barplot(x='Feature',y='Importância da feature',data=tmp)

s.set_xticklabels(s.get_xticklabels(),rotation=90)

plt.show() 