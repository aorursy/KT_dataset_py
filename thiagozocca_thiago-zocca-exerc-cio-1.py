nRowsRead = None 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # plotting

import seaborn as sns





df = pd.read_csv('/kaggle/input/dataviz-facens-20182-aula-1-exerccio-2/anv.csv', delimiter=',', nrows = nRowsRead)



df.head(10) 

resposta = [["aeronave_pais_registro", "Qualitativa Nominal"],["aeronave_tipo_veiculo","Qualitativa Nominal"],["aeronave_fabricante","Qualitativa Nominal"],["aeronave_fase_operacao","Qualitativa Nominal"],["total_fatalidades","Quantitativa Discreta"],["aeronave_ano_fabricacao","Quantitativa Discreta"],["aeronave_nivel_dano","Qualitativa Nominal"]] #variáveis relacionadas a tempo são contínuas, mas podem ser discretas pois não há perdas -- (discretização)

resposta = pd.DataFrame(resposta, columns=["Variavel", "Classificação"])

resposta
display(df['aeronave_tipo_veiculo'].value_counts())
display(df['aeronave_fabricante'].value_counts())
display(df['aeronave_fase_operacao'].value_counts())
display(df['aeronave_nivel_dano'].value_counts())
display(df['aeronave_pais_registro'].value_counts())
soma_fatalidade_veic =  df[['aeronave_tipo_veiculo','total_fatalidades']]

soma_fatalidade_veic = soma_fatalidade_veic.loc[soma_fatalidade_veic['total_fatalidades'] > 0]

soma_fatalidade_veic = soma_fatalidade_veic.groupby(['aeronave_tipo_veiculo']).count() #somando a quantidade de fatalidades por tipo de veículo

soma_fatalidade_veic = soma_fatalidade_veic.sort_values(by=['total_fatalidades'], ascending=False) #ordenando pela somatória

soma_fatalidade_veic

plt.figure(figsize=(20,5))

plt.title("Quantidade de Fatalidades por Tipo de Veículo")

sns.barplot(x=soma_fatalidade_veic.index, y=soma_fatalidade_veic.total_fatalidades, palette = 'bright')
cont_fatalidade_ano_fab =  df[['aeronave_ano_fabricacao','total_fatalidades']]



#criando as colunas novas

cont_fatalidade_ano_fab['sem_mortes'] = 0

cont_fatalidade_ano_fab['com_mortes'] = 0



#preenchendo as colunas novas

for i, l in cont_fatalidade_ano_fab.iterrows():

    if l['total_fatalidades'] > 0:

        cont_fatalidade_ano_fab.at[i,'com_mortes'] = 1

    else:

        cont_fatalidade_ano_fab.at[i,'sem_mortes'] = 1



#somando as colunas novas

cont_fatalidade_ano_fab = cont_fatalidade_ano_fab.groupby(['aeronave_ano_fabricacao']).sum()



#recriando a coluna total

cont_fatalidade_ano_fab['total'] = cont_fatalidade_ano_fab['com_mortes'] + cont_fatalidade_ano_fab['sem_mortes']



#tirando a coluna total

cont_fatalidade_ano_fab = cont_fatalidade_ano_fab.drop(columns=['total_fatalidades'])



#criando as colunas de %

cont_fatalidade_ano_fab['sem_mortes_p'] = ''

cont_fatalidade_ano_fab['com_mortes_p'] = ''



#preenchendo as colunas de %

for i, l in cont_fatalidade_ano_fab.iterrows():

    cont_fatalidade_ano_fab.at[i,'sem_mortes_p'] = l['sem_mortes']/l['total']

    cont_fatalidade_ano_fab.at[i,'com_mortes_p'] = l['com_mortes']/l['total']

    

cont_fatalidade_ano_fab = cont_fatalidade_ano_fab['com_mortes_p']

cont_fatalidade_ano_fab = cont_fatalidade_ano_fab.sort_index()#ordenando pelo ano

ax = cont_fatalidade_ano_fab.plot.line(title='Percentual de Mortes por Ano de Fabricação',subplots=False, figsize=(10,5))

ax.set_xlim(1979,2017)
cont_fatalidade_faseop = df[['aeronave_fase_operacao','total_fatalidades']]

cont_fatalidade_faseop = cont_fatalidade_faseop.loc[cont_fatalidade_faseop['total_fatalidades'] > 0]

cont_fatalidade_faseop = cont_fatalidade_faseop.groupby(['aeronave_fase_operacao']).count() #somando a quantidade de fatalidades por fase de operação

cont_fatalidade_faseop = cont_fatalidade_faseop.sort_values(by=['total_fatalidades'], ascending=False) #ordenando pela somatória

cont_fatalidade_faseop

plt.figure(figsize=(20,5))

plt.title("Quantidade de Fatalidades por Fase de Operação")

sns.barplot(y=cont_fatalidade_faseop.index, x=cont_fatalidade_faseop.total_fatalidades, palette = 'bright')
cont_fatalidade_fabricante =  df[['aeronave_fabricante','total_fatalidades','aeronave_tipo_veiculo']]

cont_fatalidade_fabricante

#criando as colunas novas

cont_fatalidade_fabricante['sem_mortes'] = 0

cont_fatalidade_fabricante['com_mortes'] = 0



#preenchendo as colunas novas

for i, l in cont_fatalidade_fabricante.iterrows():

    if l['total_fatalidades'] > 0:

        cont_fatalidade_fabricante.at[i,'com_mortes'] = 1

    else:

        cont_fatalidade_fabricante.at[i,'sem_mortes'] = 1



#somando as colunas novas

cont_fatalidade_fabricante = cont_fatalidade_fabricante.groupby(['aeronave_fabricante']).sum()



#recriando a coluna total

cont_fatalidade_fabricante['total'] = cont_fatalidade_fabricante['com_mortes'] + cont_fatalidade_fabricante['sem_mortes']



#tirando a coluna total

cont_fatalidade_fabricante = cont_fatalidade_fabricante.drop(columns=['total_fatalidades'])



#criando as colunas de %

cont_fatalidade_fabricante['sem_mortes_p'] = ''

cont_fatalidade_fabricante['com_mortes_p'] = ''



#preenchendo as colunas de %

for i, l in cont_fatalidade_fabricante.iterrows():

    cont_fatalidade_fabricante.at[i,'sem_mortes_p'] = l['sem_mortes']/l['total']

    cont_fatalidade_fabricante.at[i,'com_mortes_p'] = l['com_mortes']/l['total']

    

cont_fatalidade_fabricante

cont_fatalidade_fabricante = cont_fatalidade_fabricante.sort_values(by=['com_mortes_p'], ascending=False) #ordenando pela somatória

cont_fatalidade_fabricante = cont_fatalidade_fabricante[['com_mortes_p']].head(50)

cont_fatalidade_fabricante.plot(kind='bar', title='Percentual de Mortes por tipo de fabricante',subplots=False, figsize=(10,5))
soma_fatalidade_dano =  df[['aeronave_nivel_dano','total_fatalidades']]

soma_fatalidade_dano = soma_fatalidade_dano.loc[soma_fatalidade_dano['total_fatalidades'] > 0]

soma_fatalidade_dano = soma_fatalidade_dano.groupby(['aeronave_nivel_dano']).count() #somando a quantidade de fatalidades por nível de dano

soma_fatalidade_dano = soma_fatalidade_dano.sort_values(by=['total_fatalidades'], ascending=False) #ordenando pela somatória

soma_fatalidade_dano

plt.figure(figsize=(20,5))

plt.title("Quantidade de Fatalidades por Tipo de Veículo")

sns.barplot(x=soma_fatalidade_dano.index, y=soma_fatalidade_dano.total_fatalidades, palette = 'bright')
soma_fatalidade_dano =  df[['aeronave_pais_registro','total_fatalidades']]

soma_fatalidade_dano = soma_fatalidade_dano.loc[soma_fatalidade_dano['total_fatalidades'] > 0]

soma_fatalidade_dano = soma_fatalidade_dano.groupby(['aeronave_pais_registro']).count() #somando a quantidade de fatalidades por país de registro

soma_fatalidade_dano = soma_fatalidade_dano.sort_values(by=['total_fatalidades'], ascending=False) #ordenando pela somatória

soma_fatalidade_dano

plt.figure(figsize=(20,5))

plt.title("Quantidade de Fatalidades por País de Registro")

sns.barplot(x=soma_fatalidade_dano.index, y=soma_fatalidade_dano.total_fatalidades, palette = 'bright')