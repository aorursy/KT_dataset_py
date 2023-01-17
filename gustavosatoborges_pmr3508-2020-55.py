'''

Nesse exercício programa o objetivo era a implementação do conceito em um classificador knn no python,
como foi proposto no início do curso.

A modelagem foi baseada nos exemplos encontrados no kaggle, assim como os dados utilizados abaixo.

Para tornar mais fácil o entendimento e avaliação, vou deixar comentado cada célula do Jupyter mostrando 
meu raciocínio e intenção, ao final do exercício programa vou fazer uma conclusão, comparando os resultados
obtidos com os resultados esperados pelos exemplos encontrados.

''' 
# Aqui se encontram as biblotecas utilizadas no exercício programa.

# Em especial, um destaque ao pandas que fará o tratamento de dados.
# E o sklearn que é responsável pela implementação do classificador knn.

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Importação dos rótulos 

sample = pd.read_csv("../input/adult-pmr3508/sample_submission.csv", sep=',', na_values = '?')
sample
# Importação dos dados de treino

train_data = pd.read_csv("../input/adult-pmr3508/train_data.csv", sep=',', na_values = '?')
train_data
# Verificação das dimensões dos dados

train_data.shape
# Importação dos dados para teste do classificador

# A seguir, é feito um merge apenas para linkar os rótulos aos dados teste. 
# Com a remoção dos dados NaN, os rótulos que não foram filtrados serão armazenados para o classificador


test_data = pd.read_csv("../input/adult-pmr3508/test_data.csv", sep=',')
test_data
#Com o merge passar a ter o mesmo número de colunas que os dados de teste

test_data.shape
# Nas células abaixo há a contagem dos valores de cada coluna, para dar uma ideia de como os dados está dispostos.
# Será feito uma contagem de cada coluna, e por fim alguns gráficos para ter uma ideia visual


train_data['native.country'].value_counts()

train_data['age'].value_counts()


train_data['workclass'].value_counts()

train_data['fnlwgt'].value_counts()

train_data['education'].value_counts()

train_data['education.num'].value_counts()

train_data['marital.status'].value_counts()
 
train_data['occupation'].value_counts()
 
train_data['relationship'].value_counts()
 
train_data['race'].value_counts()
 
train_data['sex'].value_counts()
 
train_data['capital.gain'].value_counts()
 
train_data['capital.loss'].value_counts()
 
train_data['hours.per.week'].value_counts()

train_data['income'].value_counts()

# Nas células abaixo há a distribuição por gráficos de barras apenas para efeito de visualização quantitativa dos dados.



plt.rcdefaults()
plt.style.use('seaborn-dark-palette')
fig, ax = plt.subplots(figsize = (12,18))

age = train_data['age'].value_counts().index
n_age = np.arange(len(age))
valor = train_data['age'].value_counts().values

ax.barh(n_age, valor, align = 'center')
ax.set_yticks(n_age)
ax.set_yticklabels(age)
ax.invert_yaxis()
ax.set_title('Contagem por Idade')
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tick_params(axis='x',which = 'both', bottom = False, labelbottom = False)

for i in ax.patches:
    ax.text(i.get_width()+.4, i.get_y()+.6, \
           str(round((i.get_width()), 0)), color = 'black')
    
plt.show()

plt.rcdefaults()
plt.style.use('seaborn-dark-palette')
fig, ax = plt.subplots(figsize = (8,6))

country = train_data['sex'].value_counts().index
n_country = np.arange(len(country))
valor = train_data['sex'].value_counts().values

ax.barh(n_country, valor, align = 'center')
ax.set_yticks(n_country)
ax.set_yticklabels(country)
ax.invert_yaxis()
ax.set_title('Contagem por Sexo')
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tick_params(axis='x',which = 'both', bottom = False, labelbottom = False)

for i in ax.patches:
    ax.text(i.get_width()+.4, i.get_y()+.6, \
           str(round((i.get_width()), 0)), color = 'black')
    
plt.show()

plt.rcdefaults()
plt.style.use('seaborn-dark-palette')
fig, ax = plt.subplots(figsize = (8,6))

education = train_data['education'].value_counts().index
n_education = np.arange(len(education))
valor = train_data['education'].value_counts().values

ax.barh(n_education, valor, align = 'center')
ax.set_yticks(n_education)
ax.set_yticklabels(education)
ax.invert_yaxis()
ax.set_title('Contagem por Escolaridade')
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tick_params(axis='x',which = 'both', bottom = False, labelbottom = False)

for i in ax.patches:
    ax.text(i.get_width()+.4, i.get_y()+.6, \
           str(round((i.get_width()), 0)), color = 'black')
    
plt.show()

plt.rcdefaults()
plt.style.use('seaborn-dark-palette')
fig, ax = plt.subplots(figsize = (8,6))

occupation = train_data['occupation'].value_counts().index
n_occupation = np.arange(len(occupation))
valor = train_data['occupation'].value_counts().values

ax.barh(n_occupation, valor, align = 'center')
ax.set_yticks(n_occupation)
ax.set_yticklabels(occupation)
ax.invert_yaxis()
ax.set_title('Contagem pela Ocupação')
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tick_params(axis='x',which = 'both', bottom = False, labelbottom = False)

for i in ax.patches:
    ax.text(i.get_width()+.4, i.get_y()+.6, \
           str(round((i.get_width()), 0)), color = 'black')
    
plt.show()

# No data prep será removido os dados faltantes (NaN) com a função dropna() e será feito um replace nos rótulos (income)

train_data
#Como o kaggle só admite um csv com o mesmo número de linhas fixo, o dropna() não será utlizado,
# e o data prep fica em torno de substituir os NaN pelo parâmetro que mais aparece.

train_data = train_data.dropna()
train_data.drop_duplicates(keep='first', inplace=True)



train_data['income'] = train_data['income'].replace(['<=50K', '>50K'],[0,1])
sample['income'] = sample['income'].replace(['<=50K', '>50K'],[0,1])





train_data['native.country'].value_counts()

train_data['workclass'].value_counts()
train_data['occupation'].value_counts()
train_data
test_data
# Nesse primeiro teste foi modelado um knn para k = 3 como teste inicial

train_data.isnull().sum(axis = 0)

# Talvez a parte mais importante é o entendimento dessas células seguintes, em que algumas colunas são selecionadas.
# As colunas nesse primeiro caso são numéricas apenas, e como parâmetro y, é colocado os rótulos tanto de teste quanto de treino.    
    
x_train = train_data[["age","education.num","capital.gain", "capital.loss", "hours.per.week"]]
y_train = train_data.income

x_test = test_data[["age","education.num","capital.gain", "capital.loss", "hours.per.week"]]
y_test = sample.income
# Nesse primeiro teste o k é representado por 3

knn = KNeighborsClassifier(n_neighbors=3)
# faz a validação cruzada dando como sáida seus valores
# cv escolhido foi 5, porque para 10 que foi o padrão encontrado acabava não mudando o resultado significativamente
# acabando apenas demorando mais tempo para processar.

scores = cross_val_score(knn, x_train, y_train, cv=5)
scores
# Média da validação cruzada

scores.mean()
# Faz o ajuste dos dados de treino para retornar a estimação dos rótulos dos dados teste.

knn.fit(x_train,y_train)
y1_test_pred = knn.predict(x_test)
y1_test_pred
# Faz a comparação entre a estimação obtida e os rótulos, para verificar a acurácia do programa. 

accuracy_score(y_test, y1_test_pred)

# A intenção é melhorar o classificador obtendo um parâmetro K ideal, para isso será feito um teste para diversos k
# Cria um loop para obter a acurácia para diversos valores de K

neighbors = list(range(1,35, 2))
cv_scores = []

for K in neighbors:
    knn = KNeighborsClassifier(n_neighbors = K)
    scores = cross_val_score(knn,x_train,y_train,cv = 5,scoring =
    "accuracy")
    cv_scores.append(scores.mean())

# Estimativa do melhor valor k

mse = [1-x for x in cv_scores]

optimal_k = neighbors[mse.index(min(mse))]
print("O melhor valor para o parâmetro K é : {}".format(optimal_k))
# Plot do respectivos K

def plot_accuracy(knn_list_scores):
    pd.DataFrame({"K":[i for i in range(1,35, 2)], "Acurácia":knn_list_scores}).set_index("K").plot.bar(figsize= (9,6),ylim=(0.7,0.9),rot=0)
    plt.show()
plot_accuracy(cv_scores)
# como teste do classificador é feito uma transformação dos dados não numéricos 

n_train = train_data.apply(preprocessing.LabelEncoder().fit_transform)
n_train
# É feito uma matriz de correlação para ver quais as colunas que possuem maior correlação com income,
# para saber quais variáveis podem otimizar o modelamento do classificador e quais não.

mask = np.triu(np.ones_like(n_train.corr(), dtype=np.bool))
f, ax = plt.subplots(figsize=(14, 14))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(n_train.corr(),annot = True, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
# O mesmo é feito com a data base de teste

n_test = test_data.apply(preprocessing.LabelEncoder().fit_transform)
n_test
x_train = n_train.iloc[:,0:14]
x_train
y_train = n_train.income
y_train
x_test = n_test.iloc[:,0:14]
x_test
y_test = sample.income
y_test
knn = KNeighborsClassifier(n_neighbors=27)
knn.fit(x_train,y_train)
y2_test_pred = knn.predict(x_test)
accuracy_score(y_test,y2_test_pred)
# Teste feito utlizando toda a base de dados 

x_train = n_train[["age", "workclass","education","fnlwgt","education.num", "marital.status",
        "occupation", "relationship", "race", "sex", "capital.gain", "capital.loss",
        "hours.per.week", "native.country"]]
# Teste feito utlizando toda a base de dados 

x_test = n_test[["age", "workclass","education","fnlwgt", "education.num", "marital.status",
        "occupation", "relationship", "race", "sex", "capital.gain", "capital.loss",
        "hours.per.week", "native.country"]]
knn = KNeighborsClassifier(n_neighbors=27)
knn.fit(x_train,y_train)
y3_test_pred = knn.predict(x_test)
accuracy_score(y_test,y3_test_pred)
# Teste feito com apenas as colunas mais relevantes

x_train = n_train[["age", "workclass", "education.num",
        "occupation", "race", "sex", "capital.gain", "capital.loss",
        "hours.per.week"]]



# Teste feito com apenas as colunas mais relevantes

x_test = n_test[["age", "workclass", "education.num",
        "occupation", "race", "sex", "capital.gain", "capital.loss",
        "hours.per.week"]]
knn = KNeighborsClassifier(n_neighbors=27)
knn.fit(x_train,y_train)
y4_test_pred = knn.predict(x_test)
accuracy_score(y_test,y4_test_pred)
# O resultado obtido poderia ser exportado para fim de armazenamento dos rótulos.

submission = pd.DataFrame()
submission[0] = test_data.index
submission[1] = y2_test_pred
submission.columns = ['Id','income']
submission['income'] = submission['income'].replace([0, 1],['<=50K','>50K'])
submission.to_csv('submission.csv',index = False)
submission

'''

Ao término do programa alguns apontamentos podem ser feitos, o primeiro deles é em relação
a implementação do código que não tem grande segredo, com o auxílio dos notebooks encontrados no kaggle
foi dados uma boa base para implementação do classificador knn. O segundo apontamento é em relação aos
resultados obtidos que não foram satisfatórios.

O resultados obtidos em especial os 'accuracy_score' foram longe de um aceitável quando comparado aos
números obtidos nos notebooks do kaggle. Os resultados obtidos giraram em torno de 0.5, o que indica
que o classificador não está funcionando. Para resolver esse problema eu tentei identificar sem sucesso,
posteriormente tentei mudar diversas partes no código como não dropar dados, tentei usar outros métodos como
o uso do StandardScaler() também sem sucesso.

Por fim, deixei esse problema encontrado de lado e dei continuidade ao exercício programa, e baseado nos
outros exemplos de notebooks, primeiro foi encontrado o melhor parâmetro k possível, no caso k = 25.
A partir daí, os dados não numéricos foram convertidos para numéricos para serem analisados, na primeira análise
foi utilizado as 16 colunas de dados dando o melhor resultado possível, no segundo caso foi utlizado 14 colunas,
e no último caso apenas 9 colunas dando o resultado intermediário. Novamente, esse resultados não condizeram
com os outros exemplo encontrados no kaggle, porém serviram pra mostrar que de acordo com as colunas adotadas
para o 'fit', pode alterar o modelamento e resultar em um classificador melhor ou pior.

E conforme foi visto em aulas, o resultado obtido com esse classificador foi insatisfatório, visto que
ele tem acurácia de 50%, o que mostra que o modelo está chutando valores.


'''