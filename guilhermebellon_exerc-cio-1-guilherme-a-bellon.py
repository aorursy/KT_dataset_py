import pandas as pd

df= pd.read_csv('../input/dataviz-facens-20182-aula-1-exerccio-2/BR_eleitorado_2016_municipio.csv', delimiter=',')
df.head(10)

resposta = [['uf', 'Qualitativa Nominal'],
            ['nome_municipio', 'Qualitativa Nominal'],
            ["total_eleitores", "Quantitativa Discreta"],
            ['f_16', 'Quantitativa Discreta'],
            ['f_18_20', 'Quantitativa Discreta'],
            ['f_25_34', 'Quantitativa Discreta'],
            ['f_45_59', 'Quantitativa Discreta'],
            ['f_70_79', 'Quantitativa Discreta'],
            ['f_sup_79', 'Quantitativa Discreta'],
            ['gen_feminino', 'Quantitativa Discreta'],
            ['gen_masculino', 'Quantitativa Discreta'],
            ['gen_nao_informado', 'Quantitativa Discreta']]

resposta = pd.DataFrame(resposta, columns=["Variavel", "Classificação"])
resposta

tabela= df['uf'].value_counts()
print("Frequência de Estados")
tabela

import numpy as np
import matplotlib.pyplot as plt

idade =  np.array(['f_16', 'f_18_20', 'f_25_34', 'f_45_59', 'f_70_79', 'f_sup_79'])

tot_idade = np.array([df[i].sum() for i in idade])

label_idade = np.array(['16', '18-20', '25-34', '45-59', '70-79', '> 79'])

plt.figure(figsize=(10,10))

plt.bar(label_idade, tot_idade)
plt.title('Eleitores por faixa etária')

plt.show()

df_ord_eleitores = pd.DataFrame(df.sort_values(by=['total_eleitores'], axis=0, ascending=False, inplace=False))
df_ord_eleitores = df_ord_eleitores.head(10)


cidade = df_ord_eleitores['nome_municipio']
total_cidade = df_ord_eleitores['total_eleitores']

plt.figure(figsize=(10,10))

plt.barh(cidade,total_cidade, color=['green'],edgecolor =['black'])
plt.title('Quantidade de Eleitores por 10 maiores Cidades')

plt.yticks(df_ord_eleitores['nome_municipio'])
 

plt.show()
total_uf = df.groupby(['uf']).sum()['total_eleitores']
uf = df.groupby(['uf']).sum().index

plt.figure(figsize=(10,10))

plt.barh(uf,total_uf, color=['green', 'yellow', 'blue', 'white'],edgecolor =['black'])
plt.title('Quantidade de Eleitores por Estado')

plt.yticks(df['uf'])
 

plt.show()

tot_homem = df['gen_masculino'].sum()
tot_mulher = df['gen_feminino'].sum()
#tot_sex_indef = df['gen_nao_informado'].sum()


plt.pie(np.array([tot_homem, tot_mulher]),explode = [0,0.1], labels = ['Homens', 'Mulheres'], colors = ['blue', 'red',], shadow = True)
plt.title('Votos por gênero')
plt.show()

df = pd.read_csv('../input/anv.csv', delimiter=',')
df.head(1)