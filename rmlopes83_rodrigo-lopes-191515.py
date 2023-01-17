import pandas as pd

df_br_eleitorado = pd.read_csv('../input/dataviz-facens-20182-aula-1-exerccio-2/BR_eleitorado_2016_municipio.csv', delimiter=',')
df_br_eleitorado.head(3)

resposta = [["uf", "Qualitativa Nominal"],
            ["nome_municipio", "Qualitativa Nominal"],
            ["total_eleitores", "Quantitativa Discreta"],
            ["f_16", "Quantitativa Discreta"],
            ["f_17", "Quantitativa Discreta"],
            ["f_18_20", "Quantitativa Discreta"],
            ["f_21_24", "Quantitativa Discreta"],
            ["f_25_34", "Quantitativa Discreta"],
            ["f_35_44", "Quantitativa Discreta"],
            ["f_45_59", "Quantitativa Discreta"],
            ["f_45_59", "Quantitativa Discreta"],
            ["f_60_69", "Quantitativa Discreta"],
            ["f_60_69", "Quantitativa Discreta"],
            ["f_70_79", "Quantitativa Discreta"],
            ["f_sup_79", "Quantitativa Discreta"],
            ["gen_masculino","Quantitativa Discreta"],
            ["gen_nao_informado","Quantitativa Discreta"],
            ["gen_feminino","Quantitativa Discreta"]] 

resposta = pd.DataFrame(resposta, columns=["Variavel", "Classificação"])
resposta
print('UF')
df_br_eleitorado.uf.value_counts().head()
print('Nome do municipio') 
df_br_eleitorado.nome_municipio.value_counts().head()
df_br_eleitorado.head(3)
table_eleitorado = df_br_eleitorado[resposta["Variavel"]].groupby('uf').sum().reset_index()
table_eleitorado
import numpy as np
import matplotlib.pyplot as plt

dados = table_eleitorado[['uf','total_eleitores']].sort_values(by=['total_eleitores'], ascending=False)

plt.figure(figsize=(15,5))
# Create bars
plt.bar(dados['uf'],dados['total_eleitores']/100000)

plt.title('Eleitores por UF')

# Create names on the x-axis
#plt.xticks(y_pos, bars)
 
# Show graphic
plt.show()
import numpy as np
import matplotlib.pyplot as plt

dados = table_eleitorado[['uf','gen_masculino']].sort_values(by=['gen_masculino'], ascending=False)

plt.figure(figsize=(15,5))
# Create bars
plt.bar(dados['uf'],dados['gen_masculino']/100000)

plt.title('Eleitores por UF')

# Create names on the x-axis
#plt.xticks(y_pos, bars)
 
# Show graphic
plt.show()
#dados gráfico donut
gen_mas = table_eleitorado.gen_masculino.sum()
gen_fem = table_eleitorado.gen_feminino.sum()
gen_NI = table_eleitorado.gen_nao_informado.sum()

#print(gen_mas)
#print(gen_fem)
#print(gen_NI)

# library
import matplotlib.pyplot as plt
 
# create data
names='Masculino', 'Feminino', 'Não Informado',
size=[gen_mas,gen_fem,gen_NI]
plt.title('Eleitores por genero') 
# Create a circle for the center of the plot
my_circle=plt.Circle( (0,0), 0.7, color='white')

plt.pie(size, labels=names, colors=['orange','green','black'])
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.show()

dados_ac = table_eleitorado[table_eleitorado['uf']=='AC']
dados_ac = dados_ac.drop(columns=['uf', 'total_eleitores','gen_masculino', 'gen_feminino','gen_nao_informado'])
dados_ac = dados_ac.transpose().reset_index()
dados_ac.columns = ["Faixa de Idade", "Total Eleitores"]


print(dados_ac)
#Gráfico

import matplotlib.pyplot as plt

plt.figure(figsize=(15,5))
# Create bars
plt.bar(dados_ac["Faixa de Idade"],dados_ac["Total Eleitores"],color=['green'])
plt.show()

df = pd.read_csv('../input/BR_eleitorado_2016_municipio.csv', delimiter=',')
df.head(1)