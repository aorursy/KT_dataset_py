import pandas as pd

resposta = [["idade", "Quantitativa Discreta"],["sexo","Qualitativa Nominal"]] #variáveis relacionadas a tempo são contínuas, mas podem ser discretas pois não há perdas -- (discretização)

resposta = pd.DataFrame(resposta, columns=["Variavel", "Classificação"])

resposta
#imports

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style("whitegrid")

sns.set()
df = pd.read_csv('../input/dataviz-facens-20182-aula-1-exerccio-2/anv.csv')

df.head()
df.drop(columns = ["aeronave_matricula", "codigo_ocorrencia", "aeronave_operador_categoria", "aeronave_fabricante",

"aeronave_modelo", "aeronave_tipo_icao", "aeronave_motor_tipo", "aeronave_pmd_categoria", 

"aeronave_assentos", "aeronave_pais_fabricante", "aeronave_registro_categoria", 

"aeronave_registro_segmento", "aeronave_voo_origem", "aeronave_voo_destino", 

"aeronave_fase_operacao", "aeronave_fase_operacao_icao", "aeronave_tipo_operacao",

"aeronave_dia_extracao"], inplace = True)
df.head()
df.describe()
resposta = [["aeronave_tipo_veiculo", "Qualitativa Nominal"], ["aeronave_motor_quantidade", "Qualitativa Ordinal"], ["aeronave_pmd", "Quantitativa Discreta"], ["aeronave_ano_fabricacao", "Quantitativa Discreta"], ["aeronave_pais_registro", "Qualitativa Nominal"], ["aeronave_nivel_dano", "Qualitativa Ordinal"], ["total_fatalidades", "Quantitativa Discreta"]]

resposta = pd.DataFrame(resposta, columns=["Variavel", "Classificação"])

resposta
df['aeronave_tipo_veiculo'].value_counts()

df["aeronave_motor_quantidade"].value_counts()
df["aeronave_pais_registro"].value_counts()
df["aeronave_nivel_dano"].value_counts()
g = sns.countplot(x = "aeronave_tipo_veiculo", data = df)

g.set_title(label = "Contagem de Aeronave por Tipo de Veículo")

g.set_xticklabels(g.get_xticklabels(), rotation=45)

plt.show()
g = sns.countplot(x = "aeronave_motor_quantidade", data = df)

g.set_title(label = "Contagem de Aeronaves por Quantidade de Motores")

g.set_xticklabels(g.get_xticklabels(), rotation=45)

plt.show()
g = sns.distplot(df["aeronave_pmd"], kde = False, bins = 100)

plt.xlim(0, 100000) #estabelecimento de limite devido aos dados estarem muito distribuídos

g.set_title(label = "Histograma de 'aeronave_pmd'")

plt.show()
plt.figure(figsize = (10, 5))

g = sns.boxplot(x = "aeronave_pmd", data = df)

g.set_title(label = "Boxplot de 'aeronave_pmd'")

plt.show()
g = sns.boxplot(x = "aeronave_pmd", data = df, showfliers = False)

g.set_title(label = "Boxplot de 'aeronave_pmd', sem outliers")

plt.show()
#limpeza dos dados

ano_fab = df["aeronave_ano_fabricacao"]

ano_fab.dropna(inplace = True)

ano_fab = ano_fab[ano_fab > 1900]
g = sns.distplot(ano_fab, kde = False)

g.set_title(label = "Histograma de 'aeronave_ano_fabricacao'")

plt.show()
g = sns.boxplot(ano_fab)

g.set_title(label = "Boxplot de 'aeronave_ano_fabricacao'")

plt.show()
sns.set_style("whitegrid")

g = sns.countplot(x = "aeronave_pais_registro", data = df)

g.set_title(label = "Contagem de Aeronave por País de Registro")

g.set_xticklabels(g.get_xticklabels(), rotation=90)

plt.show()
g = sns.countplot(x = "aeronave_nivel_dano", data = df)

g.set_title(label = "Contagem de Aeronave por Nível de Dano")

g.set_xticklabels(g.get_xticklabels(), rotation=45)

plt.show()
plt.figure(figsize = (12, 4))

g = sns.distplot(df["total_fatalidades"], kde = False)

g.set_title(label = "Histograma de total_fatalidades")

plt.show()
g = sns.boxplot(df["total_fatalidades"])

g.set_title(label = "Boxplot de 'total_fatalidades'")

plt.show()
g = sns.boxplot(df["total_fatalidades"], showfliers = False) #remoção de outliers

g.set_title(label = "Boxplot de 'total_fatalidades'")

plt.show()

#A grande maioria dos registros está concentrada em 0, por isso o boxplot não tem largura
df = pd.read_csv('../input/anv.csv', delimiter=',')

df.head(1)