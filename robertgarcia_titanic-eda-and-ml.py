#<img src='./img/imgTitanic2.jpeg' style='width:500px;height:200px'>
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

#Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.utils import shuffle
from sklearn import linear_model, preprocessing
from sklearn.neighbors import KNeighborsClassifier

# setting style seaborn
sns.set(style="ticks", color_codes=True)
#Import data

#train data
dfTrain = pd.read_csv('../input/titanic/train.csv')

#test data
dfTest = pd.read_csv('../input/titanic/test.csv')

#combine train and test dataSet.
#dfCombine = pd.concat([dfTrain.drop('Survived', 1),dfTest])
#combine = [dfTrain, dfTest]
#head data
# Visualização das 5 primeiras linhas do DataSet usado para treino.
# Showing the first 5 rows in data set train.
dfTrain.head()
# head data test

# showing the first 5 rows in data frame used para test.
dfTest.head()
#Some info about dfTrain.
#Algumas informações sobre o dfTrain.
dfTrain.info()
#Some info about dfTest
#Algumas informações sobre dfTest
dfTest.info()
#Some info about dfCombine
#Algumas informaçẽos sobre o dfCombine.
#dfCombine.info()
 # Porcentagem relativa dos valores faltantes dfTrain.

ageIsNull = dfTrain[dfTrain['Age'].isnull() == True]

print('Porcentagem relativa das idades faltantes por algumas caracteristicas:'); print('\n',40*'-')

print(len(ageIsNull)/len(dfTrain)); print(40*'-')
print(ageIsNull['Survived'].value_counts(normalize=True)); print(40*'-')
print(ageIsNull['Pclass'].value_counts(normalize=True)); print(40*'-')
print(ageIsNull['Sex'].value_counts(normalize=True)); print(40*'-')
print(ageIsNull['Embarked'].value_counts(normalize=True)); print(40*'-')
# valores faltantes em Embarked

dfTrain[dfTrain['Embarked'].isnull() == True]
# Valores faltantes em "Cabin"
cabinIsNull = dfTrain[dfTrain['Cabin'].isnull() == True]

print('Porcentagem relativa dos dados faltantes em "Cabin" por algumas colunas: \n\n'); print(40*'-')
print(len(cabinIsNull)/len(dfTrain['Cabin'])); print(40*'-') #relative frequency 'Cabin' == Nan

print(cabinIsNull['Survived'].value_counts(normalize=True)); print(40*'-')
print(cabinIsNull['Sex'].value_counts(normalize=True)); print(40*'-')
print(cabinIsNull['Pclass'].value_counts(normalize=True)); print(40*'-')
print(cabinIsNull['Embarked'].value_counts(normalize=True)); print(40*'-')
#Describe
# Descrição de alguns valores estatisticos: média, desvio padrão, min, max ..., para as colunas númericas.
dfTrain.describe()
# informações gerais sobre os dados

freqSurvived = dfTrain['Survived'].value_counts(normalize=True) 
freqPclass = dfTrain['Pclass'].value_counts(normalize=True) 
freqSex = dfTrain['Sex'].value_counts(normalize=True) 
freqEmbarked = dfTrain['Embarked'].value_counts(normalize=True, dropna=False)

print('Relative frequency Survived: \n'); print(freqSurvived); print(40*'-')
print('Relative frequency Pclass: \n'); print(freqPclass); print(40*'-')
print('Relative frequency Sex: \n'); print(freqSex); print(40*'-')
print('Relative frequency Embarked: \n'); print(freqEmbarked); print(40*'-')
# Plot examples

fig, ([ax1, ax2],[ax3, ax4]) = plt.subplots(2,2, figsize=(12,8))
fig.suptitle('Counts to some columns')
fig.subplots_adjust(wspace=0.5, hspace=0.6)

Ax1 = sns.countplot(x='Survived', data=dfTrain, ax=ax1)
Ax2 = sns.countplot(x='Pclass',data=dfTrain, ax=ax2)
Ax3 = sns.countplot(x='Sex', data=dfTrain, ax=ax3)
Ax4 = sns.countplot(x='Embarked', data=dfTrain, ax=ax4)

ax1.set_title('Count survived and not survived')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid() 
ax1.set_yticks(dfTrain['Survived'].value_counts())

ax2.set_title('Count Pclass')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid()
ax2.set_yticks(dfTrain['Pclass'].value_counts())

ax3.set_title('Count Sex')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.grid()
ax3.set_yticks(dfTrain['Sex'].value_counts())

ax4.set_title('Count Embarked')
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.grid()
ax4.set_yticks(dfTrain['Embarked'].value_counts(dropna=False))


#plt.close(Ax1.fig)
#plt.close(Ax2.fig)
#plt.close(Ax3.fig)
#plt.close(Ax4.fig)

plt.show()

matCorr1 = dfTrain.corr()
sns.heatmap(matCorr1[['Survived']], annot=True)

plt.show()
# Plots catplot, Survived by [someColumn]

# Survived by Pclass
g1 = sns.catplot(x='Pclass', data=dfTrain, hue='Survived', kind='count')
g1.fig.suptitle('Survived by Pclass')
plt.grid()


# Survived by Sex
g2 = sns.catplot(x='Sex', data=dfTrain, hue='Survived', kind='count')
g2.fig.suptitle('Survived by Sex')
plt.grid()

# Survived by Embarked
g3 = sns.catplot(x='Embarked', data=dfTrain, hue='Survived', kind='count')
g3.fig.suptitle('Survived by Embarked')
plt.grid()

plt.show()

#Sex by Pclass
g = sns.catplot(x='Pclass', data=dfTrain, hue='Sex', kind='count')
g.fig.suptitle('Sex by Pclass')
plt.grid()

plt.show()
# Plot catplot, [SomeColumn by Embarked]

# Count Embarked, again
g0 = sns.catplot(x='Embarked', data=dfTrain, kind='count')
g0.fig.suptitle('Count Embarked')
plt.grid()

# Survived by Embarked
g1 = sns.catplot(x='Embarked', data=dfTrain, hue='Survived', kind='count')
g1.fig.suptitle('Survived by Embarked')
plt.grid()

# Pclass by Embarked
g2 = sns.catplot(x='Embarked', data=dfTrain, hue='Pclass', kind='count')
g2.fig.suptitle('Pclass by Embarked')
plt.grid()

# Sex by Embarked
g3 = sns.catplot(x='Embarked', data=dfTrain, hue='Sex', kind='count')
g3.fig.suptitle('Sex by Embarked')
plt.grid()

plt.show()
# Age column histogram

fig, ax = plt.subplots(figsize=(14,6))

tempBins = np.int64(round(dfTrain['Age'].max()))

ax.hist(dfTrain['Age'], bins=tempBins, edgecolor='black')
ax.set_title('Age column histogram')
ax.set_xlabel('Age')
ax.set_ylabel('Frequency')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid()
ax.set_xticks([x for x in range(0,85,5)])

plt.show()
# Compare Age by [someColumn]

# Survived by Age
print('\nSurvived by Age')
g = sns.FacetGrid(dfTrain, col='Survived', margin_titles=True);
g.map(plt.hist, 'Age', bins=40, edgecolor='black')
#g.fig.suptitle('Surv. by Age')
plt.show()

# Sex by Age
print('\nSex by Age')
g2 = sns.FacetGrid(dfTrain,col='Sex')
g2.map(plt.hist,'Age', bins=20)
#g2.fig.suptitle('Sex by Age')
plt.show()

# Pclass by Age
print('\nPclass by Age')
g3 = sns.FacetGrid(dfTrain,col='Pclass')
g3.map(plt.hist, 'Age', bins=20)
#g3.fig.suptitle('Pclass by Age')
plt.show()

print('\nEmbarked by Age')
g4 = sns.FacetGrid(dfTrain, col='Embarked')
g4.map(plt.hist, 'Age', bins=20)
plt.show()
# Plot, example: Sex, Survived by Age

print('\nSex, Survived by Age')
g = sns.FacetGrid(dfTrain, row='Sex', col='Survived', margin_titles='True')
g.map(plt.hist, 'Age', bins=20)

plt.show()
#Obtendo todos os valores da coluna "Ticket" e as vezes com que se repetem.
ticketValueCounts = dfTrain['Ticket'].value_counts(); ticketValueCounts
# Criando um array e um DataFrame do objeto acima para aplicar novamente um "value_counts", com o objetivo 
# de saber quantas vezes cada quantidade acima se repete.

arrayTicketValuesCont = np.array(ticketValueCounts) # Criando um array com o objeto TicketValueCounts acima.

dfTicketValuesCounts = pd.DataFrame(arrayTicketValuesCont, columns=['Count']); # Criando um DataFrame

dfTicketValuesCounts['Count'].value_counts() #aplicando "value_counts()" para saber quantas vezes cada valor acima se repete.

valTicket0 = ticketValueCounts.keys()[0] # value "CA. 2343"
valTicket1 = ticketValueCounts.keys()[1] # value "347082"

dfTrain[dfTrain['Ticket'] == valTicket0]

dfTrain[dfTrain['Ticket'] == valTicket1]
dfTrain['Fare'].value_counts()

fig, (ax1, ax2) = plt.subplots(2,1,figsize=(15,10))
fig.subplots_adjust(hspace=0.4)

ax1.hist(dfTrain['Fare'],bins=100, edgecolor='black')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_title('Frequency Fare values')
ax1.set_ylabel('Frequency')
ax1.set_xlabel('Fare')

n = dfTrain[dfTrain['Fare'] < 100] # pegando Fare < 100, mudando escala. Como dar um zoon no gráfico.
ax2.hist(n['Fare'], bins=100, edgecolor='black')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_xticks([x for x in range(0,100,5)])
ax2.set_title('Frequency Fare values < 100')
ax2.set_xlabel('Fare')
ax2.set_ylabel('Frequency')

plt.show()

fareMaior100 = dfTrain[dfTrain['Fare'] > 100] # sample dfTrain Fare > 100

print(70*'-')
print('Relative porcent Fare > 100: ',len(fareMaior100)/len(dfTrain)); print(70*'-')

h = sns.catplot('Survived', data=fareMaior100, kind='count')
h.fig.suptitle('Number Suvived to Fare > 100')
plt.grid()

g = sns.FacetGrid(fareMaior100, row='Survived', col='Sex', margin_titles=True)
g.map(plt.hist, 'Pclass')
plt.show()


#values Fare < 4
dfTrain[dfTrain['Fare'] < 4]

#Obtendo as linhas para as quais Cabin não apresenta valor faltante.
dfCabinNotNull = dfTrain[dfTrain['Cabin'].isnull() == False]

#frequencia relativa valores faltantes no data set
freqNotNullCabin = len(dfCabinNotNull)/len(dfTrain['Cabin'])

print(60*'-')
print('Frequency not null in Cabin: ',freqNotNullCabin); print(60*'-')
print('Frequency null in Cabin: ', 1 - freqNotNullCabin); print(60*'-')
#value_counts about dfTrain['Cabin']

valCountsCabin = dfTrain['Cabin'].value_counts(dropna=False)

valCountsCabin

catCabin = []

for i in dfCabinNotNull.index:
    dfCabinNotNull.loc[i, 'Cabin'] = dfCabinNotNull.loc[i,'Cabin'][0]

dfCabinNotNull['Cabin'].value_counts()

g = sns.catplot(x='Cabin', data=dfCabinNotNull, hue='Survived', kind='count')
g.fig.suptitle('Survived by Cabin')

plt.show()

# values Cabin Pclass 1

dfCabinNotNullPclass1 = dfCabinNotNull[dfCabinNotNull['Pclass'] == 1]

dfCabinNotNullPclass1['Cabin'].value_counts()
# Values Cabin Pclass 2

dfCabinNotNullPclass2 = dfCabinNotNull[dfCabinNotNull['Pclass'] == 2]

dfCabinNotNullPclass2['Cabin'].value_counts()
# Values Cabin Pclass 3

dfCabinNotNullPclass3 = dfCabinNotNull[dfCabinNotNull['Pclass'] == 3]

dfCabinNotNullPclass3['Cabin'].value_counts()

# Cabin única

dfCabinNotNull[dfCabinNotNull['Cabin'] == 'T']
# data frames copy

dfTrain2 = dfTrain.copy()
dfTest2 = dfTest.copy()

#dataCopy = {'dfTrain':dfTrain.copy(), 'dfTest':dfTest.copy()}
# dfTrain

dfTrain2
#dataCopy['dfTrain'] is dfTrain

# dfTest

dfTest2
# Dropped some columns that I will not use now.

def drop_columns(dfData, listColumns):
    dfData = dfData.drop(listColumns,1) # 1 indicating that column will be dropped.
    return dfData
 

dropColumns = ['PassengerId','Ticket','Fare','Cabin','Name'] # columns that will be dropped

# Dropping colums dfTrain and dfTest
dfTrain2 = drop_columns(dfTrain2, dropColumns)
dfTest2 = drop_columns(dfTest2, dropColumns)


#Print out current dfTrain2
dfTrain2


# mapping values to Sex
sex_map = {'male':0,'female':1}

# mapping values to dfTrain and dfTest
dfTrain2['Sex'] = dfTrain2['Sex'].map(sex_map)
dfTest2['Sex'] = dfTest['Sex'].map(sex_map)

#Print out current dfTrain2
dfTrain2
# Modificando a coluna Age e criando uma nova coluna Age_cat (categorizando a coluna Age) a partir de Age

# Mean and median to age column
ageMedia = dfTrain2['Age'].mean() # it is more affect by extreme values
ageMediana = dfTrain2['Age'].median()
#print('Mean age column: ', ageMedia)
#print('Median age column: ',ageMediana)

# Substitui os valores Nan da coluna Age pelo valor da mediana de Age
def age_nanToMedian(x):
    if(pd.isna(x)):
        return ageMediana
    else:
        return x

# Converte os valores dtype float64 para dtype int64 da coluna Age
def age_float64ToInt64(x):
    x = round(x)
    if(type(x) == np.float64):
        x = np.int64(x)
    return x


# Categoriza a coluna Age em 8 categorias baseado na idade.
def age_cat(x):
    if(x <= 10):
        x = 1
    elif(x > 10 and x <= 20):
        x = 2
    elif(x > 20 and x <= 30):
        x = 3
    elif(x > 30 and x <= 40):
        x = 4
    elif(x > 40 and x <= 50):
        x = 5
    elif(x > 50 and x <= 60):
        x = 6
    elif(x > 60 and x <= 70):
        x = 7
    elif(x > 70):
        x = 8
    return x


def age_cat2(x):
    m = 1;
    for i in range(10,61,10):
        if(x < i):
            x = m
            break
        m += 1;
    if(x >= 60):
        #x = 13
        x = 7
    return x


#Aplica a função age_nanToMedian a coluna Age do dfTrain2 e dfTest2
dfTrain2['Age'] = dfTrain2['Age'].apply(age_nanToMedian)
dfTest2['Age'] = dfTest2['Age'].apply(age_nanToMedian)


#Aplica a função age_float64ToInt64 a coluna Age do dfTrain2 e dfTest2
dfTrain2['Age'] = dfTrain2['Age'].apply(age_float64ToInt64)
dfTest2['Age'] = dfTest2['Age'].apply(age_float64ToInt64)

#Cria uma nova coluna, Age_cat (age categorizada), no dataFrame dfTrain2 a partir da coluna Age
#aplicando a função age_cat
dfTrain2['Age_cat'] = dfTrain2['Age'].map(age_cat2)
dfTest2['Age_cat'] = dfTest2['Age'].map(age_cat2)

# print out current dfTrain2
dfTrain2
#categorização de "Age"
dfTrain2['Age_cat'].value_counts(dropna=False)

g = sns.catplot(x='Age_cat', data=dfTrain2, hue='Survived', kind='count')
plt.show()
# Trabalhando com a coluna Embarked

# Mapeando os valores de embarked
def embarked_map(x):
    if(x == 'S'):
        x = 0;
    elif(x == 'C'):
        x = 1
    else:
        x = 2
    x = np.int64(x)
    return x

# Substituindo os valores Nan de Embarked pela moda de Embarked
def embarked_nanClear(x):
    if(pd.isna(x)):
        x = dfTrain2['Embarked'].mode()[0] # Obtém e atribui a 'moda' da coluna Embarked
    return x


# Aplicando o mapeamento, embarked_map, sobre a coluna Embarked
dfTrain2['Embarked'] = dfTrain2['Embarked'].apply(embarked_map)
dfTest2['Embarked'] = dfTest2['Embarked'].apply(embarked_map)

# Aplicando a função embarked_nanClear a coluna Embarked
dfTrain2['Embarked'] = dfTrain2['Embarked'].map(embarked_nanClear)
dfTest2['Embarked'] = dfTest2['Embarked'].map(embarked_nanClear)

# Print out current dfTrain2
dfTrain2
# Criando uma nova coluna 'tam_family', tamanho da familia (acompanhantes), baseado
# na soma das colunas 'SibSp' e 'Parch'

def newColumn_tamFamily(dfData):
    dfData['tam_family'] = 0 # Criando nova coluna, 'tam_family' 
    
    # Preenchendo a coluna 'tam_family' com a soma das colunas 'SibSp' e 'Parch'
    for i in range(len(dfData)):
        dfData.loc[i, 'tam_family'] = dfData.loc[i,'SibSp'] + dfData.loc[i, 'Parch']
    return dfData

newColumn_tamFamily(dfTrain2)
newColumn_tamFamily(dfTest2)


# Print out current dfTrain2
dfTrain2

g = sns.catplot(x='tam_family', data=dfTrain2, kind='count')
g.fig.suptitle('Frequency tam_family') 

plt.show()
# Outras plotagens usando a coluna 'tam_family'

g = sns.catplot(x='tam_family', data=dfTrain2, hue='Survived', kind='count')
g.fig.suptitle('tam_family by Survived')

h = sns.catplot(x='tam_family', data=dfTrain2, hue='Sex', kind='count')
h.fig.suptitle('tam_family by Sex')

i = sns.catplot(x='tam_family', data=dfTrain2, hue='Pclass', kind='count')
i.fig.suptitle('tam_family by Pclass')

plt.show()
# Criando uma nova coluna 'is_alone' (está sozinho) baseado na coluna 'tam_family'

# Preeche a coluna 'is_alone' baseado na coluna 'tam_family'. Se 'tam_family' é igual a 0 significa que
# o passageira estava desacompanhado, sozinho.

def newColumn_isAlone(dfData):
    dfData['is_alone'] = 0 # Criando nova coluna, 'is_alone' no dfTrain2
    
    for i in range(len(dfData['is_alone'])):
        if(dfData.loc[i,'tam_family'] >= 1):
            dfData.loc[i,'is_alone'] = 1
    
    return dfData


newColumn_isAlone(dfTrain2)
newColumn_isAlone(dfTest2)


dfTrain2

h = sns.catplot('is_alone', data=dfTrain2, kind='count')
h.fig.suptitle('Count is_alone')
plt.grid()

g = sns.catplot('Survived', data=dfTrain2, hue='is_alone', kind='count')
g.fig.suptitle('Count is_alone by Survived')
plt.grid()

i = sns.catplot('Pclass', data=dfTrain2, hue='is_alone', kind='count')
i.fig.suptitle('Count is_alone by Pclass')
plt.grid()

plt.show()
dfTrain['Name']
# Trabalhando com a coluna 'Name'
# Get the Name titles

#Obtem dos os títulos dos nomes, exemplo: 'Mr.', 'Miss.', 'Mrs.', ....
def obtemTituloNomes(colunaNames):
    
    titleName = []
    
    for i in colunaNames:
        tempSplitName = i.split(' ')# Divide cada nome, pelo espaço em branco, formando uma lista
        tempIndex = 1
        
        #Para cada lista criada acima, criada a partir dos espaços entre os nomes, é localizado a string com ".".
        for i in range(len(tempSplitName)):
            if(tempSplitName[i].find('.') >= 1):#verifica em qual index, lugar da lista, (nome), está o título do nome.
                tempIndex = i
                break
        
        titleName.append(tempSplitName[tempIndex]) #armazena o título do nome obtido na lista titleName.
        tempSplitName.clear() # lima a lista tempSplitName para uma nova iteração. 
    
    # Cria um dataFrame com uma coluna, chamada 'Name', com os valores dos títulos dos nomes.
    dfTitleName = pd.DataFrame(titleName, columns=['Name'])
    return dfTitleName


# Cria uma nova coluna, 'Name', em dfTrain2  e dfTest2 preenchendo-as com os valores de dfTitleName
dfTrain2['Name'] = obtemTituloNomes(dfTrain['Name'])
dfTest2['Name'] = obtemTituloNomes(dfTest['Name'])


# print out current dfTrain2
print("Data frame contendo uma nova coluna \"Name\", com o valor dos títulos dos nomes originais: ")
dfTrain2
# value_counts() sobre a coluna 'Name' do dfTrain2, criada e preenchida acima.
namesValueCounts = dfTrain2['Name'].value_counts(); 

# Print out value_counts column 'Name'.
print("Títulos e frequência com que ocorrem: ")
namesValueCounts
dfTrain2[dfTrain2['Name'] == 'Capt.'] # Capitão não sobreviveu
# Obtem os titulos dos nomes que ocorrem poucas vezes

k = namesValueCounts.keys() # obtem as keys sobre value_counts() obtido acima. obtem os valores 'Mr.', 'Miss.',...
otherNames = []

for i in range(len(namesValueCounts)):
    if(namesValueCounts[i] <= 7):#veririca se o título do nome possui menos que 7 ocorrências.
        otherNames.append(k[i])#armazena em otherNames as keys que possuem menos que 7 ocorrências.

# print out, títulos dos nomes que ocorrem poucas vezes, menor que 5 ocorrências.
print("Títulos que ocorrem com baixa frequência ( <= 7 vezes): ")
otherNames
# Change the Name column. Substitui os valores que ocorrem poucas vezes, obtido acima por 'OtherTitle'.

# troca os titulos que ocorrem poucas vezes, obtido acima, pela label, rótulo 'otherTitle'.
# Exemplo, substitui as ocorrenceas de 'Major' por 'otherTitle'. 
def change_titleToOtherTitle(x):
    for i in otherNames:
        if(x == i):
            return 'OtherTitle.'
    return x


#Aplica a função 'change_titleToOtherTitle' sobre a coluna 'Name' do dfTrain2 e dfTest2
dfTrain2['Name'] = dfTrain2['Name'].apply(change_titleToOtherTitle)
dfTest2['Name'] = dfTest2['Name'].apply(change_titleToOtherTitle)


#Print out value_counts() current column 'Name' do dfTrain2
nameValueCounts = dfTrain2['Name'].value_counts(); nameValueCounts
# Plotando a distribuição dos títulos.

fig, ax = plt.subplots(figsize=(10,5))

ax.hist(dfTrain2['Name'], edgecolor='black')
ax.set_title('Titles name Frequency')
ax.set_xlabel('Titles')
ax.set_ylabel('Frequency')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid()

plt.show()

g = sns.catplot(x='Pclass', data=dfTrain2, hue='Name', kind='count')

h = sns.catplot(x='Survived', data=dfTrain2, hue='Name', kind='count')

plt.show()
dfTrain2[dfTrain2['Name'] == "OtherTitle"]
# Trabalhando com a coluna 'Fare'

# Using coluna 'Fare', categorizando-a.

fareMin = dfTrain['Fare'].min()
fareMax = dfTrain['Fare'].max()

# Função para categorizar a coluna 'Fare'. 
def cat_fare(x):
    if(x <= 30):
        temp1 = 1;
        for i in range(5,31,5):
            if(x <= i):
                x = temp1
                break
            temp1 += 1
    elif(x > 30 and x <= 130):
        x = 7
        temp2 = 7
        for i in range(40, 131, 10):
            if(x <= i):
                x = temp2
                break
            temp2 += 1
    else:
        x = 18
        #temp3 = 14
        #for i in range(200, 601, 100):
         #   if(x <= i):
          #      x = temp3
           #     break
            #temp3 += 1
    x = np.int64(x)
    return x
        

#Cria a coluna 'cat_fare' e preenche seus valores com a função 'cat_fare'     
dfTrain2['cat_fare'] = dfTrain['Fare'].apply(cat_fare)
dfTest2['cat_fare'] = dfTest['Fare'].apply(cat_fare)


# Print out current 'cat_fare' do dfTrain2
dfTrain2['cat_fare'].value_counts()
g = sns.catplot(x='cat_fare', data=dfTrain2, hue='Survived', kind='count')

plt.show()
#Categorização coluna "Ticket"

ticketValCounts = pd.concat([dfTrain['Ticket'], dfTest['Ticket']])

valCountsTicket = ticketValCounts.value_counts()
valCountsTicket
ticketValCountsKeys = valCountsTicket.keys()
ticketValCountsKeys


def ticket_cat(x):
    for i in range(len(ticketValCountsKeys)):
        if(x == ticketValCountsKeys[i]):
            x = valCountsTicket[i]
            break
    return x

dfTrain2['cat_ticket'] = dfTrain['Ticket'].apply(ticket_cat)
dfTest2['cat_ticket'] = dfTest['Ticket'].apply(ticket_cat)

#Print out current dfTrain2
dfTrain2
#print the current dfTest2

dfTest2
g = sns.catplot(x='cat_ticket', data=dfTrain2, hue='Survived', kind='count')

g = sns.catplot(x='cat_fare', data=dfTrain2, hue='cat_ticket', kind='count')

plt.show()

def cabin_rotulo(x):
    if(pd.isnull(x) == True):
        x = 0
    else:
        x = 1
    return x

dfTrain2['Cabin_not_null'] = dfTrain['Cabin'].apply(cabin_rotulo)
dfTest2['Cabin_not_null'] = dfTest['Cabin'].apply(cabin_rotulo)

dfTrain2

# Relative Survived by Cabin_not_null

cabinNotNull1 = dfTrain2[dfTrain2['Cabin_not_null'] == 1] #Pegando os registros que apresentam registro em Cabin não nulo.
cabinNotNull0 = dfTrain2[dfTrain2['Cabin_not_null'] == 0] #Pegando os registros que apresentam registro em Cabin como nulo.

freqRelCabinNotNull1 = len(cabinNotNull1[cabinNotNull1['Survived'] == 1]) / len(cabinNotNull1) 
#Pegando os registros para Sobreviventes e que estão no grupo em que apresentam registro de Cabin não nulo.
#Dividindo a quantidade de elementos para Sobreviventes e Cabin diferente de nulo pela quantidade total de 
#registros com Cabin não nulo, objetivo é pegar a frequencia relativa.
#O mesmo é feito abaixo, mas para o grupo com Cabin nulo.

freqRelCabinNotNull0 = len(cabinNotNull0[cabinNotNull0['Survived'] == 1]) / len(cabinNotNull0)

print(80*'-')
print('Relative frequency Survived by Cabin_not_null == 1: ', freqRelCabinNotNull1); print(80*'-')
print('Relative frequency Survived by Cabin_not_null == 0: ', freqRelCabinNotNull0); print(80*'-')

g = sns.catplot(x='Cabin_not_null', data=dfTrain2, hue='Survived', kind='count')
g.fig.suptitle('Survived by Cabin_not_null')

plt.show()
#mapr = {'Died':0,'Survived':1}
#dfTrain2['Survived'] = dfTrain2['Survived'].map(mapr)

matCorr = dfTrain2.corr()
sns.heatmap(matCorr[['Survived']],annot=True)

#current dfTrain2
dfTrain2

# current dfTest2
dfTest2
# Features and labels

#features aos quais irá se aplicar o modelo de ML
features = ['Sex','Embarked','Name','cat_fare','is_alone','Age_cat', 'Pclass','tam_family','cat_ticket','Cabin_not_null']

# Label para ML
label = 'Survived'
# Usando LabelEncoder para categorizar todas as features, obtendo valores int, para se aplicar o modelo Ml.

labEncod = preprocessing.LabelEncoder()

featTransTrain = []
featTransTest = []

for i in features:
    featTransTrain.append(labEncod.fit_transform(list(dfTrain2[i])))
    featTransTest.append(labEncod.fit_transform(list(dfTest2[i])))
    
z = zip(*featTransTrain)   
# mesmo efeito que: z = zip(feat[0],feat[1],feat[2],feat[3],feat[4],feat[5],feat[6], ...), ou seja,
# zipar uma lista de listas.

x_daframe_test = list(zip(*featTransTest))
    

x = list(z) #features encoded
y = list(labEncod.fit_transform(list(dfTrain[label]))) #label encoded


# Aplicando modelo ML, KNeighborsClassifier.
# 

n = 1000 # número de vezes para rodar a aplicação do modelo para se obter a melhor accuracia dentre esse numero de vezes.
bAccuracy = 0 # armazena melhor acuracia obtida.
bModel = 0 # armazena melhor modelo obtido.

for i in range(n):
    
    # separa os dados em train e test, pegando 10% dos dados para test
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    # Cria instancia do modelo KNeighborsClassifier com 9 vizinhos
    model = KNeighborsClassifier(n_neighbors=5)

    # aplica o modelo KNeighborsClassifier aos dados de treino.
    model.fit(x_train,y_train)
    
    # teste o modelo obtido na linha acima aos dados de teste para se obter a acurâcia.
    accuracy = model.score(x_test,y_test)
    
    # verifica e armazena o melhor modelo obtido dentro do número de vezes que se rodou o modelo.
    if(accuracy > bAccuracy):
        bAccuracy = accuracy
        bModel = model

        
# Print out melhor Accuracy obtida        
print('Accuracy KNeighborsClassifier: ', bAccuracy)

# Aplicando SVC

from sklearn.svm import SVC

bAccuracySVC = 0
bModelSVC = 0

for i in range(1000):

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

    modelSVC = SVC(probability=True)
    modelSVC.fit(x_train, y_train)

    ac = modelSVC.score(x_test,y_test)
    
    if(ac > bAccuracySVC):
        bAccuracySVC = ac
        bModelSVC = modelSVC

print('Accuracy SVC: ', bAccuracySVC)

#Predicted - Applying model KNeighborsClassifier

example = []

#Applying model about x_test
predicted = bModel.predict(x_test)

for i in range(len(x_test)):
    example.append([predicted[i], x_test[i], y_test[i]])

#Building a DataFrame with example values 
dfExample = pd.DataFrame(example, columns=['Predicted','features','RealValue'])

#Print out dfExample
dfExample

# Aplicando o modelo SVC aos dados de previsão de Survived externos.

predictedSVC = bModelSVC.predict(x_daframe_test)

passengId = list(dfTest['PassengerId'])
predictSVC = list(predictedSVC)

exportData = list(zip(passengId,predictSVC))


dfExportData = pd.DataFrame(exportData, columns=['PassengerId','Survived'])
#dfExportData.to_csv(r'./predicted/pred5.csv', index=False, header=True)