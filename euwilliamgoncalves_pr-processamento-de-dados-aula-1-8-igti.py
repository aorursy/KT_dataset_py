import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import google

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# abrindo o arquivo csv que contém os dados a serem utilizados nesse exercício
customers = pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
# visualizando as 5 primeiras linhas do banco de dados
customers.head()
# contando os dados
customers.info()
# verificando existência de dados nulos
customers.isnull().sum()
# criando cópia dos dados
customers_null=customers

# adicionando valores nulos
for col in customers_null.columns:
    customers_null.loc[customers_null.sample(frac=0.1).index, col] = np.nan
# verificando dados e o acréscimo de valores nulos
customers_null.info()
# analisando primeiras 10 linhas do dataset
customers_null.head(10)
# verificando os campos nulos

customers_null.isnull().sum()
# deletando as linhas com valores nulos

customers_null.dropna()
# preenchendo os valores nan com o valor 0
customers_null.fillna(0)
# preenche valores nulos com a média da coluna

customers_null.fillna(customers_null.mean()) 
customers_null.describe() # estatísticas do dataset
# constrói o boxplot para as colunas desejadas (verificando outliers)

boxplot = customers.boxplot(column=['Age', 'Annual Income (k$)', 'Spending Score (1-100)'])
# Z-score (verificando outliers)

from scipy import stats
z = np.abs(stats.zscore(customers['Annual Income (k$)'].values))
threshold = 2
result = np.where(z > threshold)

df_salario_outlier = customers.iloc[result[0]]
df_salario_outlier
# analisando a distribuição dos clientes por gênero
sns.countplot(x='Gender', data=customers); # cria o gráfico que conta a quantidade de consumidores existentes em cada um dos gêneros
plt.title('Distribuição dos clientes quanto ao gênero'); #adiciona o título ao gráfico
# analisando a distribuição dos clientes quanto a idade através do histograma
customers.hist('Age', bins=35)  # seleciona a coluna idade para realizar o histograma
                                # os bins representam a quantidade de grupos em que se deseja dividir os dados
plt.title('Distribuição dos clientes pela idade')
plt.xlabel('Idade')
# CONVERTENDO COLUNAS DO TIPO OBJECT
cat_df_customers = customers.select_dtypes(include=['object']) # copiando as colunas que são do tipo object
cat_df_customers.head() #exibindo resultado
# APLICANDO O MAPEAMENTO
replace_map = {'Gender': {'Male': 1, 'Female': 2}} #define o dicionário a ser utilizado 
labels = cat_df_customers['Gender'].astype('category').cat.categories.tolist() #encontra a lista das variáveis categóricas
replace_map_comp = {'Gender' : {k: v for k, v in zip(labels,list(range(1,len(labels)+1)))}} #define o mapeamento

print(replace_map_comp)
cat_df_customers_replace = pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv') #importando dataset

cat_df_customers_replace.replace(replace_map_comp, inplace=True) #aplica mapeamento ao dataset
cat_df_customers_replace.head()
#APLICANDO LABEL ENCODERING
