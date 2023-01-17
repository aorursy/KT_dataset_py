import numpy as np # importa a biblioteca para fazer operações em matrizes
import pandas as pd # importa a biblioteca para facilitar a manipulação e análise dos dados

# importa o arquivo e guarda em um dataframe do Pandas
df_dataset = pd.read_csv('../input/iriscsv/iris.csv', sep=',', index_col=None) 
display(df_dataset.shape)
display(df_dataset.head(n=5))
# apresenta as principais estatísticas sobre a base de dados
df_detalhes = df_dataset.describe()

display(df_detalhes)
df_classe = df_dataset["classe"]
display(df_classe.shape)
display(df_classe.describe())
display(df_classe.unique())
df_amostrasSetosa = df_dataset[ df_dataset["classe"] == "Iris-setosa"]
display(df_amostrasSetosa.shape)
df_detalhesClasses = df_dataset['classe'].describe()

display(df_detalhesClasses)

# encontra as classes do problema
classes = df_dataset['classe'].unique()
print('\nClasses do problema: ', classes)

# conta a quantidade de dados em cada classe
for classe in classes:
    df_classe = df_dataset[ df_dataset['classe']==classe]
    
    print('Qtd. de dados da classe %s: %d' %(classe, df_classe.count().values[0]))
# apresenta as principais estatísticas sobre a base de dados
display(df_dataset.boxplot(figsize=(15,7)))
import seaborn as sns
import matplotlib.pyplot as plt

# matriz de gráficos scatter 
sns.pairplot(df_dataset, hue='classe', size=3.5);

# mostra o gráfico usando a função show() da matplotlib
plt.show()
#scatter plot
sns.lmplot(x='comprimento_sepala', y='largura_petala', data=df_dataset, 
           fit_reg=False, # No regression line
           hue='classe')   # Color by evolution stage

# cria um título para o gráfico
plt.title('Comprimento vs largura da sépala.')

# mostra o gráfico
plt.show()
# define a dimensão do gráfico
plt.figure(figsize=(10,7))

# cria o boxplot
sns.boxplot(x="classe", y="comprimento_sepala", data=df_dataset, whis=1.5)

#mostra o gráfico
plt.show()
# cria um gráfico de barras com a frequência de cada classe
sns.countplot(x="classe", data=df_dataset)

#mostra o gráfico
plt.show()
mean = df_dataset.mean()

std = df_dataset.std()

# criando um gráfico de barras vertical
plt.figure(figsize=(10,5))
mean.plot(kind="bar", rot=0, color="red", fontsize=13, yerr=std);
plt.show()

# criando um gráfico de barras horizontal
plt.figure(figsize=(10,5))
mean.plot(kind="barh", rot=0, color="red", fontsize=13, xerr=std);
plt.show()

# cria o histograma
n, bins, patches = plt.hist(df_dataset['comprimento_petala'].values,bins=10, color='red', edgecolor='black', linewidth=0.9)

#mostra o gráfico
plt.show()

# imprime as cestas de valores
print(bins)
# criando o gráfico de densidade 
densityplot = df_dataset.plot(kind='density')

# mostra o gráfico
plt.show()

# criando o gráfico de densidade apenas do atributo comprimento_petala
densityplot = df_dataset['comprimento_petala'].plot(kind='density')

# mostra o gráfico
plt.show()
# criando uma matriz X com os valores do data frame
X = df_dataset.iloc[:,:-1].values # exceto a coluna da classe (a última = -1)

# matriz de covariancia
covariance = np.cov(X, rowvar=False)

# matriz de correlação
correlation = np.corrcoef(X, rowvar=False)

print('Matriz de covariância: ')
display(covariance)

print('\n\nMatriz de correlação: ')
display(correlation)
# matriz de covariancia
df_covariance = df_dataset.cov()

# matriz de correlação
df_correlation = df_dataset.corr()

print('Matriz de covariância: ')
display(df_covariance)

print('\n\nMatriz de correlação: ')
display(df_correlation)
# cria um mapa de cores dos valoes da covariancia
sns.heatmap(df_covariance, 
        xticklabels=df_correlation.columns,
        yticklabels=df_correlation.columns)

plt.title('Covariância')
plt.show()

# cria um mapa de cores dos valoes da correlação
sns.heatmap(df_correlation, 
        xticklabels=df_correlation.columns,
        yticklabels=df_correlation.columns)

plt.title('Correlação')
plt.show()
# importa o arquivo e guarda em um dataframe do Pandas
df_dataset2 = pd.read_csv( '../input/data2csv/data2.csv', sep=',', index_col=None) 
display(df_dataset2.head(n=5))
display(df_dataset2.describe())
import seaborn as sns
import matplotlib.pyplot as plt

n, bins, patches = plt.hist(df_dataset2['atributo_d'].values,bins=10 , color ='orange', edgecolor = 'black', linewidth = 0.9)

plt.show()

display(bins, n)
import seaborn as sns
import matplotlib.pyplot as plt

#sns.pairplot(df_dataset2,hue = 'classe',size = 3.5)

#plt.show()

sns.lmplot(x='atributo_a',y='atributo_d',data=df_dataset2,            
           fit_reg = False,
           hue = 'classe')

plt.title('Atributo_d vs Atributo_a')

plt.show()
sns.countplot(x= "classe", data = df_dataset2)

plt.show()
#plt.figure(figsize=(10,7))
#sns.boxplot(x="classe", y="atributo_d",data = df_dataset2,whis=1.5)
#plt.show()

display(df_dataset2.boxplot(figsize=(15,7)))
df_covarancia = df_dataset2.cov()
display("Covarancia", df_covarancia)

df_correlation = df_dataset2.corr()
display("Correlação:", df_correlation)
def covariancia(atributo1, atributo2):
    """
    Função usada para calcular a covariância entre dois vetores de atributos
    """    
    
    #inicializando a covariancia. Essa é a variável que deve ser retornada pela função
    cov = 0 
    
    # número de objetos
    n = len(atributo1)
    
    ################# COMPLETE O CÓDIGO AQUI  #################

    
    
    
    ##########################################################
    
    return cov

atributo1 = df_dataset2['atributo_a'].values
atributo2 = df_dataset2['atributo_b'].values

print('Valor esperado: 4.405083')

cov = covariancia(atributo1, atributo2)
print('Valor retornado pela função: %1.6f' %cov)
def correlacao(atributo1, atributo2):
    """
    Função usada para calcular a correção entre dois vetores de atributos
    """
    
    #inicializando a covariancia. Essa é a variável que deve ser retornada pela função
    corr = 0 
    
    # número de objetos
    n = len(atributo1)
    
    ################# COMPLETE O CÓDIGO AQUI  #################
    # Se você for usar a função do Numpy para calcular o desvio padrão,
    # não se esqueça de usar o parâmetro ddof=1 para fazer a correção de Bessel
    #
    # Use a função que você criou no exercício anterior para calcular 
    # o valor da covariância que será usado para calcular a correlação

    
    ##########################################################
    
    return corr

atributo1 = df_dataset2['atributo_a'].values
atributo2 = df_dataset2['atributo_b'].values

print('Valor esperado: 0.264026')

corr = correlacao(atributo1, atributo2)
print('Valor retornado pela função: %1.6f' %corr)