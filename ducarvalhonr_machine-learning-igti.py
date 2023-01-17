#Este programa é utilizado para o desenvolvimento do trabalho prático da disciplina FAM do bootcamp de MLE
#importando as bibliotecas
import numpy as np #biblioteca utilizada para o tratamento de valores numéricos (vetores e matrizes)
import pandas as pd #biblioteca utilizada para o tratamento de dados via dataframes 
import seaborn as sns #biblioteca utilizada para criar gráficos mais bonitos
import matplotlib.pyplot as plt #biblioteca utilizada para construir os gráficos
import google
# abrir o arquivo csv que contém os dados s serem utilizados a prática
customers = pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
#apresentando as 5 primeiras linhas do banco de dados
customers.head()
#verificando a existência de campos nulos
customers.info()
#verificando a existência de campos nulos
customers.isnull().sum()
#adicionando valores nulos
customers_null=customers
for col in customers_null.columns:
    customers_null.loc[customers_null.sample(frac=0.1).index, col] = np.nan
#verificando as colunas nulas
customers_null.info()
#analisando o dataset
customers_null.head(10)
#verificando a existência de campos nulos
customers_null.isnull().sum()
#deletando as linhas que possuem algum valor nulo
customers_null.dropna()
#preenchendo os valores nan com o valor 0
customers_null.fillna(0)
#encontra as estastisticas do dataset
customers_null.describe()
#preenchendo os valores médios da coluna
customers_null.fillna(customers_null.mean())
#analisando o banco de dados
#função que retorna uma análise superficial dos dados
customers.describe()
#constroi o boplot para as colunas desejadas
boxplot = customers.boxplot(column=['Age', 'Annual Income (k$)', 'Spending Score (1-100)'])
#Z-score
from scipy import stats
z = np.abs(stats.zscore(customers['Annual Income (k$)'].values))
threshold = 2
result=np.where(z > threshold)

df_salario_outlier=customers.iloc[result[0]]
#print(z)
#todos os usuários com sajário anual com possivel outlier
df_salario_outlier
#analisando a distribuição dos clientes por gênero
#cria o gráfico que conta a quantidade de consumidores existente em cada um dos gêneros
#adiciona o título no gráfico
sns.countplot(x='Gender', data=customers);
plt.title('Distribuição dos clientes quanto ao gênero');
#analisando a distribuiçãoi dos clientes quanto a idade atrvés do histograma
#seleciona a coluna idade para realizar o histograma
#os "bins" indicam a quantidade de grupos que se deseja dividir os dados
#adiciona o título ao gráfico (histograma)
customers.hist('Age', bins=35);

plt.title('Distribuição dos clientes pela idade');
plt.xlabel('Idade');
#copiando as colunas que são do tipo categoricas
cat_df_customers = customers.select_dtypes(include=['object'])
cat_df_customers.head()
replace_map = {'Gender': {'Male': 1, 'Female': 2}} #define o dicionário a ser utilizado (map)
labels = cat_df_customers['Gender'].astype('category').cat.categories.tolist() #encontra a lista das variáveis categóricas
replace_map_comp = {'Gender' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}#define o mapeamento

print(replace_map_comp)
#aplica o mapeamento para o dataset
cat_df_customers_replace =pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
#aplica o mapeamento para o dataset
cat_df_customers_replace.replace(replace_map_comp, inplace=True)
cat_df_customers_replace.head()
#cat_df_customers_lc = customers
customers = pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
cat_df_customers_lc=customers
cat_df_customers_lc['Gender']=pd.Categorical(cat_df_customers_lc["Gender"])
cat_df_customers_lc.dtypes
cat_df_customers_lc['Gender'] = cat_df_customers_lc['Gender'].cat.codes
cat_df_customers_lc.head()
#importando o label encoding
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder() #instanciado o objeto
#aplicando a codificação para as colunas categéricas
customers_label=pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
customers_label['Gender'] = le.fit_transform(customers_label['Gender'])
customers_label.head(10)
#Get dummies
#customers_one_hot['Gender']=pd.Categorical(customers_one_hot['Gender'])
#customers_one_hot head
customers_one_hot=pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
customers_one_hot= pd.get_dummies(customers_one_hot)
customers_one_hot.head()
customers=pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
#importe OneHotEncoder
customers_one_hot=customers
from sklearn.preprocessing import OneHotEncoder

#intancia o objeto
ohe = OneHotEncoder()
#aplica o one hot encoding para a coluna
customers_ohe = ohe.fit_transform(customers_one_hot['Gender'].values.reshape(-1,1)).toarray() #It returns an numpy array
customers_ohe.shape
customers_ohe