# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns #biblioteca para gráficos
import matplotlib.pyplot as plt # gráficos também
import google

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#abrir arquivo csv
customers = pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
customers.head()
customers.info()
customers.isnull().sum()
#Adiconando valores nulos, apenas para treinamento
customers_null = customers
for col in customers_null.columns:
    customers_null.loc[customers_null.sample(frac=0.1).index,col] = np.nan
customers_null.info()
customers_null.isnull().sum()
customers_null.head(10)
#retira linhas com NaN 
customers_null.dropna()
#adicionar um valor no lugar do NaN 
customers_null.fillna(0)
customers_null.describe()
customers_null.fillna(customers_null.mean())
customers.describe()
boxplot = customers.boxplot(column = ['Age','Annual Income (k$)','Spending Score (1-100)'])
#cada traço representa um quartil(25% das entradas), 
#a caixa representa 50% das entrads(50% das entradas estão dentro da caixa)
#possível outiers são mostradas pela bolinha fora do quartil
#o traço no meio da caixa é a mediana

#Z-score, para verficar se existem outliers. 
#Z-score encontra média e desvio padrão, encontra valores de x desvios padrões (threshold)
from scipy import stats
z = np.abs(stats.zscore(customers['Annual Income (k$)'].values))
threshold = 2
result = np.where(z > threshold)

df_salario_outlier = customers.iloc[result[0]]
#mostrar os possíveis outliers
df_salario_outlier
sns.countplot(x= 'Gender',data = customers) #grafico que mostra a quantidade de cada genero
plt.title('Distribuição de genero') #cria título
customers.hist('Age',bins = 35) #histograma da idade, dividido em 35 grupos
plt.title('Distribuicao de idades') #título
plt.xlabel('Idade') #label do eixo x
#colunas categóricas
cat_df_customers = customers.select_dtypes(include = ['object'])
#aplicar mapeamento
replace_map = {'Gender': {'Male':1,'Female':2}} #define o dicionário a ser utilizado (map)
labels = cat_df_customers['Gender'].astype('category').cat.categories.tolist()#encontra a lista das variáveis categóricas
replace_map_comp = {'Gender': {k: v for k,v in zip(labels,range(1,len(labels)+1))}}
print(replace_map_comp)
rep_customers = pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
rep_customers.replace(replace_map_comp,inplace = True)
rep_customers.head()
#transforma a coluna do tipo object para o tipo category
customers = pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
customers['Gender'] = pd.Categorical(customers['Gender'])
customers.dtypes
#label encoding do pandas
customers['Gender'] = customers['Gender'].cat.codes
customers.head()
#aplicar Label Encoding do skitlearn
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
customers = pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
customers['Gender'] = le.fit_transform(customers['Gender'])
customers.head()
#get dummies
customers = pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
customers = pd.get_dummies(customers)
customers.head()
#one hot encoding
customers = pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder() #instancia o objeto
customers_one = one.fit_transform(customers['Gender'].values.reshape(-1,1)).toarray() #returns a numpy array
customers_one.shape
customers_one
