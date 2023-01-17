import numpy as np # álgebra Linear

import pandas as pd # processamento de dados, E / S de arquivo CSV (por exemplo, pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
data = pd.read_csv('../input/pokemon.csv')
data.info()
data.corr()
# Vamos visualizar a correlação dos dados

f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
data.head(10)
# obter as colunas do nosso data set

data.columns
# Gráfico de linha

# color = cor, label = rótulo, linewidth = largura da linha, alpha = opacidade, grid = grade, 

# linestyle = estilo da linha

data.Speed.plot(kind = 'line', color = 'g',label = 'Speed',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.Defense.plot(color = 'r',label = 'Defense',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()
# Gráfico de dispersão

# x = ataque, y = defesa

data.plot(kind='scatter', x='Attack', y='Defense',alpha = 0.5,color = 'red')

plt.xlabel('Ataque')              # Rótulos

plt.ylabel('Defesa')

plt.title('Dispersão de Ataque e Defesa') # Titulo do gráfico
# Histograma

# caixas = número de barras na figura

data.Speed.plot(kind = 'hist',bins = 50,figsize = (12,12))

plt.show()
# clf() = limpa novamente, você pode iniciar um novo

data.Speed.plot(kind = 'hist',bins = 50)

plt.clf()

# Não podemos ver o gráfico devido a clf()
#criar dicionário  e procurar suas chaves e valores

dictionary = {'espanha' : 'madrid','usa' : 'vegas'}

print(dictionary.keys())

print(dictionary.values())
# As chaves devem ser objetos imutáveis, como string, booleano, float, número inteiro ou tubos

# A lista não é imutável

# Chaves são únicas

dictionary['espanha'] = "barcelona"    # atualizar entrada existente

print(dictionary)

dictionary['frança'] = "paris"       # Adicionar nova entrada

print(dictionary)

del dictionary['espanha']              # remover entrada com a chave 'espanha'

print (dictionary)

print('frança' in dictionary)        # verifique incluir ou não

dictionary.clear()                   # remova todas as entradas no dict

print(dictionary)
# Para executar todo o código, você precisa comentar esta linha

# del dictionary             

print(dictionary)       # dá erro porque o dicionário é deletado
# carregar dataset com read_csv()

data = pd.read_csv('../input/pokemon.csv')
series = data['Defense']        # data['Defense'] = series

print(type(series))

data_frame = data[['Defense']]  # data[['Defense']] = data frame

print(type(data_frame))
# Operador de comparação

print(3 > 2)

print(3!=2)



# Boolean operators

print(True and False)

print(True or False)
# 1 - Filtrando o quadro de dados do Pandas

x = data['Defense']>200

# Existem apenas 3 pokemons com maior valor de defesa do que 200

data[x]
# 2 - Filtrando pandas com *logical_and*

# Existem apenas 2 pokemons com maior valor de defesa que 2oo e maior valor de ataque que 100

data[np.logical_and(data['Defense']>200, data['Attack']>100 )]
# Isso também é o mesmo da linha de código anterior. Portanto, também podemos usar '&' para filtrar.

data[(data['Defense']>200) & (data['Attack']>100)]
# Permaneça em loop se a condição (i não for igual a 5) for verdadeira

i = 0

while i != 5 :

    print('i é: ',i)

    i +=1 

print(i,' é igual a 5')
# Permaneça em loop se a condição (i não for igual a 5) for verdadeira

lis = [1,2,3,4,5]

for i in lis:

    print('i é: ',i)

print('')



# Para dicionários

# Podemos usar o loop for para obter a chave e o valor do dicionário. 

dictionary = {'espanha':'madrid','frança':'paris'}

for key,value in dictionary.items():

    print(key," : ",value)

print('')



# no pandas, podemos alcançar índice e valor

for index,value in data[['Attack']][0:1].iterrows():

    print(index," : ",value)
# exemplo do que aprendemos acima

def tuble_ex():

    """ retorno definido t tuble"""

    t = (1,2,3)

    return t

a,b,c = tuble_ex()

print(a,b,c)
# veja o que sai na impressão



x = 2

def f():

    x = 3

    return x

print(x)      # x = 2 ecopo global

print(f())    # x = 3 escopo local
# E se não houver escopo local

x = 5

def f():

    y = 2*x        # não há escopo local x

    return y

print(f())         # usa escopo global x

# Primeiro escopo local pesquisado, depois escopo global pesquisado, 

# se dois deles não puderem ser encontrados por último, é contruído no escopo pesquisado.
# Como podemos aprender o que é construído no escopo

import builtins

dir(builtins)
# função aninhada

def square():

    """ retornar quadrado de valor """

    def add():

        """ adicione duas variáveis locais """

        x = 2

        y = 3

        z = x + y

        return z

    return add()**2

print(square())  
# argumentos padrão

def f(a, b = 1, c = 2):

    y = a + b + c

    return y

print(f(5))



# e se quisermos mudar argumentos padrão

print(f(5,4,3))
# argumentos flexíveis * args

def f(*args):

    for i in args:

        print(i)

f(1)

print("")

f(1,2,3,4)



# argumentos flexíveis ** kwargs que é dicionário

def f(**kwargs):

    """ imprime a chave e valor do dicionário"""

    for key, value in kwargs.items(): 

        print(key, " ", value)

f(country = 'espanha', capital = 'madrid', population = 123456)
# função lambda

square = lambda x: x**2     # onde x é o nome do argumento

print(square(4))



tot = lambda x,y,z: x+y+z   # onde x, y, z são nomes de argumentos

print(tot(1,2,3))
number_list = [1,2,3]

y = map(lambda x:x**2,number_list)

print(list(y))
# Exemplo de Iteração

name = "nelson"

it = iter(name)

print(next(it))    # imprime a próxima iteração

print(*it)
list1 = [1,2,3,4]

list2 = [5,6,7,8]

z = zip(list1,list2)

print(z)

z_list = list(z)

print(z_list)
un_zip = zip(*z_list)

un_list1,un_list2 = list(un_zip) # unzip returns tuble

print(un_list1)

print(un_list2)

print(type(un_list2))
# Exemplo de compreensão de lista

num1 = [1,2,3]

num2 = [i + 1 for i in num1 ]

print(num2)
# Condicionais em iterável

num1 = [5,10,15]

num2 = [i**2 if i == 10 else i-5 if i < 7 else i+5 for i in num1]

print(num2)
# vamos retornar pokemon csv e fazer mais um exemplo de compreensão de lista

# vamos classificar os pokemons com velocidade alta ou baixa. Nosso limite é a velocidade média.

threshold = sum(data.Speed)/len(data.Speed)

data["speed_level"] = ["high" if i > threshold else "low" for i in data.Speed]

data.loc[:10,["speed_level","Speed"]] # we will learn loc more detailed later
data = pd.read_csv('../input/pokemon.csv')

data.head()  #  mostra as primeiras 5 linhas
# tail mostra as últimas 5 linhas

data.tail()
# columns fornece nomes de colunas dos recursos

data.columns
# shape fornece o número de linhas e colunas em tupla

data.shape
# info: fornece tipos de dados como quadro de dados, número de amostra ou linha, número de recurso ou coluna, 

# tipos de recurso e uso de memória

data.info()
# Por exemplo, vamos ver a frequência dos tipos de pokemom

print(data['Type 1'].value_counts(dropna =False)) 

# se houver valores nan que também serão contados

# Como pode ser visto abaixo, existem 112 pokemon de água ou 70 pokemon de grama
# Por exemplo, o max HP é 255 ou a min defense é 5

data.describe() #ignora entradas nulas
# Por exemplo: compare o ataque de pokemons lendários ou não

# A linha preta no topo é máxima

# A linha azul na parte superior é de 75%

# Linha vermelha é mediana (50%)

# A linha azul na parte inferior é de 25%

# A linha preta na parte inferior é mínima

# Não há outliers

data.boxplot(column='Attack',by = 'Legendary')
# Em primeiro lugar, crio novos dados a partir de dados de pokemons para explicar facilmente o melt().

data_new = data.head()    # Coloco apenas 5 linhas em novos dados

data_new
# usando o melt()

# id_vars = o que não queremos no melt()

# value_vars = o que queremos no melt()

melted = pd.melt(frame=data_new,id_vars = 'Name', value_vars= ['Attack','Defense'])

melted
# Index é nome

# Eu quero fazer com que as colunas sejam variáveis

# Finalmente, os valores nas colunas são valor

melted.pivot(index = 'Name', columns = 'variable',values='value')
# Em primeiro lugar, vamos criar 2 dataframes

data1 = data.head()

data2= data.tail()

conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True)

# axis = 0: adiciona dataframes na linha

conc_data_row
data1 = data['Attack'].head()

data2= data['Defense'].head()

conc_data_col = pd.concat([data1,data2],axis =1) # axis = 0: adiciona dataframes na linha

conc_data_col
data.dtypes
# vamos converter o objeto (str) em categórico e int em flutuante.

data['Type 1'] = data['Type 1'].astype('category')

data['Speed'] = data['Speed'].astype('float')
# Como você pode ver, o Tipo 1 é convertido de objeto em categórico

# E Speed, é convertido de int para float

data.dtypes
# Vamos analisar se os dados de pokemon têm valor nan

# Como você pode ver, existem 800 entradas. No entanto, o Tipo 2 

# possui 414 objetos não nulos e, portanto, 386 objetos nulos

data.info()
# Vamos verificar o Tipo 2

data["Type 2"].value_counts(dropna =False)

# Como você pode ver, há 386 valores NAN
# Vamos apagar os valores NaN

data1=data

# também usaremos dados para preencher o valor que falta, então eu os atribuo à variável data1

data1["Type 2"].dropna(inplace = True)

# inplace = True significa que não o atribuímos a uma nova variável.

# Alterações atribuídas automaticamente aos dados

# Então funciona?
# Permite verificar com declaração assert

# Declaração de afirmação (Assert statement):

assert 1==1 # não retorna nada porque é verdade
# Para executar todo o código, precisamos comentar esta linha

# assert 1 == 2 # erro de retorno porque é falso

assert  data['Type 2'].notnull().all()

# não retorna nada porque apagamos os valores nan
data["Type 2"].fillna('empty',inplace = True)
assert  data['Type 2'].notnull().all()

# não retorna nada porque não temos valores nan
# dataframe a partir de um dicionário

country = ["Espanha","França"]

population = ["11","12"]

list_label = ["país","população"]

list_col = [country,population]

zipped = list(zip(list_label,list_col))

data_dict = dict(zipped)

df = pd.DataFrame(data_dict)

df
# adicionar novas colunas

df["capital"] = ["madrid","paris"]

df
# Broadcasting

df["renda"] = 0 #Broadcasting na coluna inteira

df
# Plotando todos os dados - fica bastante confuso

data1 = data.loc[:,["Attack","Defense","Speed"]]

data1.plot()
# Aplicando subplots para ficar mais claro e fácil de analisar

data1.plot(subplots = True)

plt.show()
# scatter plot  

data1.plot(kind = "scatter",x="Attack",y = "Defense")

plt.show()
# hist plot  

data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),normed = True)
# subplot do histograma com não cumulativo e cumulativo

fig, axes = plt.subplots(nrows=2,ncols=1)

data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),normed = True,ax = axes[0])

data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),normed = True,ax = axes[1],cumulative = True)

plt.savefig('grafico.png')

plt
data.describe()
time_list = ["1992-03-08","1992-04-12"]

print(type(time_list[1])) # Como você pode ver, a data é string

# no entanto, queremos que seja um objeto datetime 

datetime_object = pd.to_datetime(time_list)

print(type(datetime_object))
# fechar aviso (warnings)

import warnings

warnings.filterwarnings("ignore")



# Para praticar, vamos pegar os dados do pokemon e adicionar uma lista de horários

data2 = data.head()

date_list = ["10/01/1992","10/02/1992","10/03/1992","15/03/1993","16/03/1993"]

datetime_object = pd.to_datetime(date_list)

data2["date"] = datetime_object

# vamos fazer a data como índice

data2= data2.set_index("date")

data2 
# Agora podemos selecionar de acordo com nosso índice de datas

print(data2.loc["16/03/1993"])

print(data2.loc["10/03/1992":"16/03/1993"])
# Usaremos o data2 que criamos na parte anterior

data2.resample("A").mean()
# Vamos usar Resampling com o mês

data2.resample("M").mean()

# Como você pode ver, existem muitas nan porque o data2 não inclui todos os meses
# Na vida real (os dados são reais. Não criados por nós como o data2), 

# podemos resolver esse problema com interpolação (interpolate)

# Podemos interpolar a partir do primeiro valor

data2.resample("M").first().interpolate("linear")
# leitura de dados

data = pd.read_csv('../input/pokemon.csv')

data= data.set_index("#")

data.head()
# indexação usando colchetes

data["HP"][1]
# usando atributo de coluna e rótulo de linha

data.HP[1]
# usando loc accessor

data.loc[1,["HP"]]
# Selecionando apenas algumas colunas

data[["HP","Attack"]]
# Diferença entre fatiar (slicing) colunas: séries e dataframes

print(type(data["HP"]))     # series

print(type(data[["HP"]]))   # data frames
# Fatiando (slicing) e indexando séries

data.loc[1:10,"HP":"Defense"]   # 10 e "Defense" são inclusivas
# Reverse slicing 

data.loc[10:1:-1,"HP":"Defense"] 
# De um lugar até ao fim

data.loc[1:10,"Speed":] 
# Criando séries boleanas

boolean = data.HP > 200

data[boolean]
# combinando filtros

first_filter = data.HP > 150

second_filter = data.Speed > 35

data[first_filter & second_filter]
# Filtrando colunas com base em outras

data.HP[data.Speed<15]
# Funções simples de python

def div(n):

    return n/2

data.HP.apply(div)
# Ou podemos usar a função lambda

data.HP.apply(lambda n : n/2)
# Definindo Coluna Usando Outras Colunas

data["total_power"] = data.Attack + data.Defense

data.head()
# nosso nome de índice é este:

print(data.index.name)

# vamos alterar o nome

data.index.name = "index_name"

data.head()
# Substituir índice

# se queremos modificar o índice, precisamos mudar todos eles.

data.head()

# primeira copie nossos dados para data3 e depois altere o índice 

data3 = data.copy()

# vamos fazer com que o índice comece a partir de 100. Não é uma mudança notável, mas é apenas um exemplo

data3.index = range(100,900,1)

data3.head()
# permite ler o dataframe mais uma vez para começar do início

data = pd.read_csv('../input/pokemon.csv')

data.head()

# Como você pode ver, existe um índice. No entanto, queremos definir uma ou mais colunas para serem indexadas
# Definir índice: tipo 1 é externo Tipo 2 é índice interno

data1 = data.set_index(["Type 1","Type 2"]) 

data1.head(100)

# data1.loc["Fire","Flying"] # como usar índices
dic = {"tratamento":["A","A","B","B"],"gênero":["F","M","F","M"],"resposta":[10,45,5,9],"idade":[15,4,72,65]}

df = pd.DataFrame(dic)

df
# pivoting

df.pivot(index="tratamento",columns = "gênero",values="resposta")
df1 = df.set_index(["tratamento","gênero"])

df1

# vamos desempilhar
# nível determina índices

df1.unstack(level=0)
df1.unstack(level=1)
# alterar a posição do índice de nível interno e externo

df2 = df1.swaplevel(0,1)

df2
df
# df.pivot(index="tratamento",columns = "gênero",values="resposta")

pd.melt(df,id_vars="tratamento",value_vars=["idade","resposta"])
# vamos usar o dataframe df

df
# de acordo com o tratamento, use outras características

df.groupby("tratamento").mean()   # média é o método de agregação / redução

# existem outros métodos como sum, std, max ou min
# só podemos escolher um dos recursos

df.groupby("tratamento").idade.max() 
# Ou podemos escolher vários recursos

df.groupby("tratamento")[["idade","resposta"]].min() 
df.info()

# como você pode ver, o gênero é objeto

# No entanto, se usarmos groupby, podemos converter dados categóricos.

# Como os dados categóricos usam menos memória, acelere operações como agrupar

#df ["gênero"] = df ["gênero"]. astype ("categoria")

#df ["tratamento"] = df ["tratamento"]. astype ("categoria")

# df.info ()