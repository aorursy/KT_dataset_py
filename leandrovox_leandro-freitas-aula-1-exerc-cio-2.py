import pandas as pd

resposta = [["idade", "Quantitativa Discreta"],["sexo","Qualitativa Nominal"]] #variáveis relacionadas a tempo são contínuas, mas podem ser discretas pois não há perdas -- (discretização)

resposta = pd.DataFrame(resposta, columns=["Variavel", "Classificação"])

resposta
df = pd.read_csv('../input/anv.csv', delimiter=',')

df.head(1)
colunas = [["aeronave_tipo_veiculo", "Qualitativa Nominal"], ["aeronave_operador_categoria", "Qualitativa Nominal"],

           ["aeronave_assentos", "Quantitativa Discreta"],["aeronave_ano_fabricacao", "Quantitativa Discreta"],

          ["aeronave_fase_operacao", "Qualitativa Nominal"],["aeronave_tipo_operacao", "Qualitativa Nominal"],

           ["aeronave_nivel_dano", "Qualitativa Nominal"],["total_fatalidades", "Quantitativa Discreta"],

           ["aeronave_motor_tipo", "Qualitativa Nominal"]]

df_novo = pd.DataFrame(colunas, columns=["Variavel", "Classificação"])

df_novo
dic_tabelas = {}

variaveis_qualitativas = [x for x in df_novo[df_novo['Classificação'] == 'Qualitativa Nominal']['Variavel']]

for v in variaveis_qualitativas:

    dic_tabelas[v] = df[v].value_counts()

    print(df[v].value_counts())

    print('\n')
import matplotlib.pyplot as plt

import numpy as np
for i, v in dic_tabelas['aeronave_motor_tipo'].items():

    plt.bar(i, v, color = 'b')

    plt.text(i, v, v, va='bottom', ha='center')

    

plt.title('Quantidade x Tipo de Motor')

plt.style.use('seaborn')

plt.gca().axes.get_yaxis().set_visible(False)

plt.show()
for i, v in dic_tabelas['aeronave_tipo_veiculo'].items():

    plt.bar(i, v, label = i + ' Q: ' + str(v))   

    

plt.title('Quantidade x Tipo de Veículo')



plt.gca().axes.get_xaxis().set_visible(False)

plt.legend()

plt.show()
for i, v in dic_tabelas['aeronave_operador_categoria'].items():

    plt.bar(i, v, label = i)

    plt.text(i, v, v, va='bottom', ha='center')    

    

plt.title('Quantidade x Tipo de Veículo')



plt.axis('off')

plt.legend()

plt.show()
plt.style.available
df = pd.read_csv('../input/anv.csv', delimiter=',')

df.head(1)