# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np # biblioteca utilizada para trabalhar com vetores

import pandas as pd # biblioteca para trabalhar com DataFrames

import seaborn as sns # biblioteca utilizada para criar gráficos mais 'bonitos'

import matplotlib.pyplot as plt # biblioteca para criar gráficos comuns ao estilo Matlab

import google
# abrir o arquivo csv que contém os dados a serem utilizados

customers = pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
#visualizando as 5 primeiras linhas do dataset

customers.head()
# verificando a existência de campos nulos

customers.info()
# verificando a existência de campos nulos

customers.isnull().sum()
# verificando valores nulos

customers_null=customers

for col in customers_null.columns:

    customers_null.loc[customers_null.sample(frac=0.1).index, col] = np.nan
customers_null.info() #verificando as colunas nulas
# analisando o dataset

customers_null.head(10)
# verificando a existência de campos nulos

customers_null.isnull().sum()
# coletando as linhas que possuem algum valor nulo

customers_null.dropna()
# preenchendo os valores 'nan' com valor 0

customers_null.fillna(0)
#encontra as estatísticas do dataset

customers_null.describe()
#preenchendo os valores médiosda coluna

customers_null.fillna(customers_null.mean())
# analisando os dados

customers.describe() #função que retorna uma análise superficial dos dados
# constroi um boxplot para as colunas desejadas

boxplot = customers.boxplot(column=['Age', 'Annual Income (k$)', 'Spending Score (1-100)']) 
# Z-score

from scipy import stats



z = np.abs(stats.zscore(customers['Annual Income (k$)'].values))

threshold = 2

result = np.where(z > threshold)



df_salario_outlier = customers.iloc[result[0]]

#print(z)
#todos os usuários com salário anual com possível outlier

df_salario_outlier
# analisando a distribuição dos clientes por gênero

sns.countplot(x='Gender', data=customers) #cria o gráfico que conta a quantidade de consumidores existentes em cada um dos gêneros

plt.title('Distribuição dos Clientes quanto ao Gênero') #adiciona título no gráfico
#analisando a distribuição dos clientes quanto a idade através do histograma

customers.hist('Age', bins=35); #seleciona a coluna idade para realizar o histograma

                                # os 'bins' indicam a quantidade de grupos que se deseja dividir os dados

plt.title('Distribuição dos Clientes pela Idade') #adiciona título ao gráfico

plt.xlabel('Idade');
cat_df_customers = customers.select_dtypes(include=['object']) #copiando as colunas que são tipos categóricas
cat_df_customers.head()
replace_map = {'Gender': {'Male': 1, 'Female': 2}} #define o dicionário a ser utilizado (map)

labels = cat_df_customers['Gender'].astype('category').cat.categories.tolist() #encontra a lista das variáveis categóricas

replace_map_comp = {'Gender': {k: v for k, v in zip(labels,list(range(1,len(labels)+1)))}} #define o mapeamento



print(replace_map_comp)
#realiza a cópia do dataset

cat_df_customers_replace = pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
cat_df_customers_replace.replace(replace_map_comp, inplace=True) #aplica o mapeamento ao dataset

cat_df_customers_replace.head()
#cat_df_customers_lc = customers

customers = pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
cat_df_customers_lc = customers
cat_df_customers_lc['Gender']=pd.Categorical(cat_df_customers_lc['Gender'])

cat_df_customers_lc.dtypes
cat_df_customers_lc['Gender'] = cat_df_customers_lc['Gender'].cat.codes

cat_df_customers_lc.head()
#Importando o Label Encoding

from sklearn.preprocessing import LabelEncoder



le = LabelEncoder() #Instanciando o objeto
#aplicando a codificação para as colunas categóricas

customers_label = pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')

customers_label['Gender'] = le.fit_transform(customers_label['Gender'])

customers_label.head(10)
# Get Dummies

customers_one_hot = pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')

#customers_one_hot['Gender']=pd.Categorical(customers_one_hot['Gender'])

customers_one_hot = pd.get_dummies(customers_one_hot)

customers_one_hot.head()
customers = pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
# Importando o OneHotEncoder

customers_one_hot = customers



from sklearn.preprocessing import OneHotEncoder



ohe = OneHotEncoder() # Instânciando o obejeto
#aplica o One Hot Encoding para a coluna

customers_ohe = ohe.fit_transform(customers_one_hot['Gender'].values.reshape(-1,1)).toarray() #It returns a numpy array

customers_ohe.shape
customers_ohe