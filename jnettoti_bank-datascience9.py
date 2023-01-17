import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
# Pandas ~ função pd.read_csv() para carregar

arq = '../input/bank.csv'

dataset = pd.read_csv(arq, encoding='utf-8', engine='python', sep=';')



# Encoding: UTF-8 para reconhecer qualquer tipo de caractere universal no padrão Unicode

# Sep: separador utilizado no arquivo, sendo vírgulas, ponto e vírgula, etc.



# Documentação: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
dataset.head()

# Documentação: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.head.html
dataset.describe()

# Documentação: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html
dataset['age'].std()
dataset['duration'].mean()
dataset.isna()

# Documentação: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.isna.html
dataset.duplicated().sum()
dataset.drop_duplicates().apply(len)
# Dataset array 

df = pd.DataFrame(dataset, columns = ['age', 'campaign'] )



# Criação do histograma 

df.hist()



# Visualização do gráfico

plt.show() 
dataset['campaign'].value_counts().plot.barh(edgecolor='black', title='Número de contatos realizados antes da campanha')
dataset['previous'].value_counts().plot.pie(title='Quantidade de campanhas', label='Campanhas')
plt.scatter(df['age'], df['campaign']) 

plt.show()
plt.boxplot(df['age']) 

plt.show() 
x = dataset ['age']

y = dataset ['campaign']

z = np.random.rand(50)



colors = np.random.rand(len(dataset)) 

plt.scatter(x, y, s=z*1500,c=colors)



plt.show()
fig, ax = plt.subplots()

fig.set_size_inches(14, 10)



ax=sns.heatmap(dataset.corr())
import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
arq = '../input/bank.csv'

dataset = pd.read_csv(arq, encoding='utf-8', engine='python', sep=';')