import numpy as np #biblioteca utilizada para trabalhar com vetores

import pandas as pd #biblioteca para trabalhar com dataframes (planilhas excel)

import seaborn as sns #biblioteca utilizada para cirar gráficos mais "bonitos"

import matplotlib.pyplot as plt #biblioteca para criar gráficos "comuns" ao estilo Matplot

import google
#abrir o arquivo csv que contém os dados a serem utilizados durante a prática

customers = pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
#visualizando as 5 primeiras linhas do banco de dados

customers.head()
#verificando a existência de campos nulos

customers.info()
#verificando a existência de campos nulos

customers.isnull().sum()
#adicionando valores nulos

customers_null=customers

for col in customers_null.columns:

    customers_null.loc[customers_null.sample(frac=0.1).index, col] = np.nan
customers_null.info() #verificando as colunas nulas
#analisando o dataset

customers_null.head(10)
#verificando a existência de campos nulos

customers_null.isnull().sum()
#deletando a existência de campos nulos

customers_null.dropna()
#preenchendo os valores nan com o valor 0

customers_null.fillna(0)
customers_null.describe() #encontra as estatísticas do dataset
#preenchendo os valores médios da coluna

customers_null.fillna(customers_null.mean())
#analisando o banco de dados

customers.describe() #função que retorna uma naálise superficial dos dados
boxplot = customers.boxplot(column=['Age', 'Annual Income (k$)', 'Spending Score (1-100)']) 

#constroi o boxplot para as colunas desejadas
#z-score

from scipy import stats

z = np.abs(stats.zscore(customers['Annual Income (k$)'].values))

thereshold = 2

result=np.where(z > thereshold)



df_salario_outlier=customers.iloc[result[0]]

#print(z)
df_salario_outlier #todos os usuários com salário anual com possível outlier
#analisando a distribuição dos clientes por gênero

sns.countplot(x='Gender', data=customers); #cria o gráfico que conta a quantidade de consumidores existente em cada um dos generos

plt.title('Distribuição dos clientes quanto ao gênero'); #adiciona título no gráfico
#analisando a distribuição dos clientes quanto a idade através do histograma

customers.hist('Age', bins=35); #seleciona a coluna idade para realizar o histograma

                                #os bins indicam a quantidade de grupos que se deseja dividir os dados

plt.title('Distribuição dos clientes pela idade'); #adiciona o título ao gráfico(histograma)

plt.xlabel('Idade')
cat_df_customers = customers.select_dtypes(include=['object']) #copiando as colunas que são do tipo categóricas
cat_df_customers.head()
replace_map = {'Gender': {'Male': 1, 'Female': 2}} #define o dicionário a ser utilizado (map)

labels = cat_df_customers['Gender'].astype('category').cat.categories.tolist() #encontra a lista das variáveis categóricas

replace_map_comp = {'Gender' : {k: v for k, v in zip(labels,list(range(1,len(labels)+1)))}} #define o mapeamento



print(replace_map_comp)
cat_df_customers_replace =pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')

#realiza a cópia do dataset
cat_df_customers_replace.replace(replace_map_comp, inplace=True) #aplica o mapeamento para o dataset

cat_df_customers_replace.head()
#cat_df_customers_lc = customers

customers = pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
cat_df_customers_lc=customers
cat_df_customers_lc['Gender']=pd.Categorical(cat_df_customers_lc['Gender'])

cat_df_customers_lc.dtypes
cat_df_customers_lc['Gender'] = cat_df_customers_lc['Gender'].cat.codes

cat_df_customers_lc.head()
#importando o label encoding

from sklearn.preprocessing import LabelEncoder



le = LabelEncoder() #instanciando o objeto
#aplicando a codificação para as colunas categóricas

customers_label=pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')

customers_label['Gender'] = le.fit_transform(customers_label['Gender'])

customers_label.head(10)
#Get dummies

customers_one_hot=pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')

#custumers_one_hot['Gender']=pd.Categorical(customers_one_hot['Gender'])

customers_one_hot= pd.get_dummies(customers_one_hot)

#customers_one_hot head

customers_one_hot.head()
customers=pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
#importe OneHotEncoder

customers_one_hot=customers

from sklearn.preprocessing import OneHotEncoder



ohe = OneHotEncoder() #instancia o objeto
#aplica o one hot encoding para a coluna

customers_ohe = ohe.fit_transform(customers_one_hot['Gender'].values.reshape(-1,1)).toarray() #It returns an numpy array

customers_ohe.shape
customers_ohe