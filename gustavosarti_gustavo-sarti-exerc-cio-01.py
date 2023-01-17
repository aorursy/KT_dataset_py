import pandas as pd



df = pd.read_csv('../input/dataviz-facens-20182-aula-1-exerccio-2/BR_eleitorado_2016_municipio.csv', delimiter=',')

df.head()
resposta = [

    ["uf", "Categórica Discreta"],

    ["total_eleitores", "Numérica Discreta"],

    ['f_16', 'Numérica Discreta'],

    ['f_17', 'Numérica Discreta'],

    ['gen_feminino', 'Numérica Discreta'],

    ['gen_masculino', 'Numérica Discreta'],

    ['gen_nao_informado', 'Numérica Discreta']

] #variáveis relacionadas a tempo são contínuas, mas podem ser discretas pois não há perdas -- (discretização)



resposta = pd.DataFrame(resposta, columns=["Variavel", "Classificação"])

resposta
print("Tabela de frequência de Estados:")

df.uf.value_counts()
import matplotlib.pyplot as plt

import numpy as np



uf = df.uf.value_counts()



plt.figure(figsize=(20,5))

uf.plot.bar(color = 'cornflowerblue')

plt.title('Municipios por UF')
eleitores_estado = df.groupby(['uf']).sum()['total_eleitores']

ufs = df.groupby(['uf']).sum().index



novo_index_uf = np.argsort(eleitores_estado)[::-1]

eleitores_estado = eleitores_estado[novo_index_uf]

ufs = ufs[novo_index_uf]





plt.figure(figsize=(20,5))

plt.bar(ufs, eleitores_estado, color = 'cornflowerblue')

plt.title('Quantidade de eleitores por unidade da federação Brasileira')

plt.xticks(rotation='vertical')



plt.show()
faixas = ['f_16','f_17','f_18_20','f_21_24','f_25_34','f_35_44','f_45_59','f_60_69','f_70_79','f_sup_79']

totais = df[faixas]



totais = np.array([df[i].sum() for i in faixas]) # realizo a soma de todos os valores das colunas

totais



faixas = ['16','17','18-20','21-24','25-34','35-44','45-59','60-69','70-79','79+']



plt.figure(figsize=(10,5))



plt.bar(faixas, totais, color = 'cornflowerblue')

plt.title('Nº de eleitores por faixa etária')





plt.show()

h = df['gen_masculino'].sum()

m = df['gen_feminino'].sum()

ni = df['gen_nao_informado'].sum()



# create data

sexo='Homem', 'Mulher', 'Não Informado'

size=[h,m,ni]

 

# Create a circle for the center of the plot

my_circle=plt.Circle( (0,0), 0.7, color='white')



from palettable.colorbrewer.qualitative import Pastel1_7

plt.pie(size, labels=sexo, autopct='%1.1f%%', colors=Pastel1_7.hex_colors)

p=plt.gcf()

p.gca().add_artist(my_circle)

plt.show()

df = pd.read_csv('../input/anv.csv', delimiter=',')

df.head(1)