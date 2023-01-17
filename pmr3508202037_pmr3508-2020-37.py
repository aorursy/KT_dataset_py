#importando bibliotecas

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import numpy as np



#armazenando e imprimindo resultados no notebook

%matplotlib inline



#configurando estilo de plotagem

plt.style.use('fivethirtyeight')
#importando arquivo de treino

df_train = pd.read_csv("../input/adult-pmr3508/train_data.csv", na_values = '?')



#setando Id como índice

df_train.set_index('Id',inplace=True)



#dimensão do DataFrame

print('Dimensão do DataFrame:', df_train.shape)



#imprimindo as 5 primeiras linhas do DataFrame

df_train.head()
#calculando total de missing data

total = df_train.isnull().sum().sort_values(ascending = False)



#calculando razão entre missing data e total de dados

percent = ((df_train.isnull().sum()/df_train.isnull().count())*100).sort_values(ascending = False)



#construindo DataFrame com resultados

missing_data = pd.concat([total, percent], axis = 1, keys = ['Total', '%'])



missing_data
print('Occupation:\n')

print(df_train['occupation'].describe()) #imprimindo uma descrição de occupation



print('\n\nWorkclass:\n')

print(df_train['workclass'].describe()) #imprimindo uma descrição de workclass



print('\n\nNative.country:\n')

print(df_train['native.country'].describe()) #imprimindo uma descrição de native.country
value = df_train['workclass'].describe().top #identificando moda em workclass

df_train['workclass'] = df_train['workclass'].fillna(value) #preenche missing data com a moda



value = df_train['native.country'].describe().top #identificando moda em native.country

df_train['native.country'] = df_train['native.country'].fillna(value) #preenche missing data com a moda





value = df_train['occupation'].describe().top #identificando moda em occupation

df_train['occupation'] = df_train['occupation'].fillna(value) #preenche missing data com a moda
#calculando total de missing data

total = df_train.isnull().sum().sort_values(ascending = False)



#calculando razaão entre missing data e total de dados

percent = ((df_train.isnull().sum()/df_train.isnull().count())*100).sort_values(ascending = False)



#construindo DataFrame com resultados

missing_data = pd.concat([total, percent], axis = 1, keys = ['Total', '%'])



missing_data
#imprimindo descrição dos dados de treino

df_train.describe()
#plotando histograma de capital.gain

plt.figure(figsize=(14, 7))

df_train['capital.gain'].hist()

plt.xlabel('Capital gain',size=18)

plt.ylabel('Quantity',size=18)

plt.title('Capital gain histogram',size=20)
#plotando histograma de capital.loss

plt.figure(figsize=(14, 7))

df_train['capital.loss'].hist()

plt.xlabel('Capital loss',size=18)

plt.ylabel('Quantity',size=18)

plt.title('Capital loss histogram',size=20)
#plotando histograma de age

plt.figure(figsize=(14, 7))

df_train['age'].hist()

plt.xlabel('Age',size=18)

plt.ylabel('Quantity',size=18)

plt.title('Age histogram',size=20)
#plotando distribuição da idade

sns.set()

plt.figure(figsize=(14,7))

sns.distplot(df_train['age'], bins = 70)

plt.xlabel('Age',size=18)

plt.ylabel('Quantity',size=18)

plt.title('Distribution of age',size=20)
#plotando histograma hours.per.week

plt.figure(figsize=(14, 7))

df_train['hours.per.week'].hist()

plt.xlabel('Hours per week',size=18)

plt.ylabel('Quantity',size=18)

plt.title('Hours per week histogram',size=20)
#plotando hitograma de trabalhadores >40 hours.per.week

over_work = df_train[df_train['hours.per.week'] > 40]

plt.figure(figsize=(14, 7))

over_work['hours.per.week'].hist(bins = 5)

plt.xlabel('Hours per week',size=18)

plt.ylabel('Quantity',size=18)

plt.title('Hours per week histogram',size=20)
mean = over_work['hours.per.week'].describe()['mean']

print('{0} horas por semana ({1} horas por dia com finais de semana livre). O que é algo que já começa a ser bastante desgastante para o trabalhador.'.format(int(mean), int(mean/5)))
def compare_histogram(df, obj_var, test_var, obj_labels = None, alpha = 0.7):

    '''

    Recebe:

        1x DataFrame

        2x atributos em formato de string

    Retorna:

        1x histograma de comparação

    '''

    if obj_labels is None:

        obj_labels = df[obj_var].unique()

    

    temp = []

    n_labels = len(obj_labels)

    for i in range(n_labels):

        temp.append(df[df[obj_var] == obj_labels[i]])

        temp[i] = np.array(temp[i][test_var]).reshape(-1,1)



    fig = plt.figure(figsize= (14,7))

    

    for i in range(n_labels):

        plt.hist(temp[i], alpha = alpha)

    plt.xlabel(test_var,size=18)

    plt.ylabel('Quantity',size=18)

    plt.title('Histogram over \'' + test_var + '\' filtered by \'' + obj_var + '\'',size=20)

    plt.legend(obj_labels)
#comparando renda com relação a idade

compare_histogram(df_train, 'income', 'age')
#plotando gráfico de sex

plt.figure(figsize=(14, 7))

df_train['sex'].value_counts().plot(kind = 'pie')
#calculando quantidade de homens

male = df_train[df_train['sex'] == 'Male'].count()[0]



#calculando quantidade de mulheres

female = df_train.shape[0] - male



print("Há {0} homens e {1} mulheres, ou seja, apenas {2:3.2f}% são mulheres.".format(male, female, female*100/(female+male)))
#comparando renda com relação ao sexo

compare_histogram(df_train, 'income', 'sex')
# male_income = [qtd > 50K, qtd <= 50K]

male_income = []

temp = df_train[df_train['sex'] == 'Male'] #filtrando quais são homens

male_income.append(temp[temp['income'] == '>50K'].count()[0]) #filtrando renda >50k

male_income.append(male-male_income[0])



# female_income = [qtd > 50K, qtd <= 50K]

female_income = []

temp = df_train[df_train['sex'] == 'Female'] #filtrando quais são mulheres

female_income.append(temp[temp['income'] == '>50K'].count()[0]) #filtrando renda >50k

female_income.append(female-female_income[0])



#porcentagem de homens que ganham mais de 50k

male_over = male_income[0]/male



#porcentagem de mulheres que ganham mais de 50k

female_over = female_income[0]/female



print('Dentre os homens, {0:1.2f}% possuem renda anual superior a 50.000, já dentre as mulheres, temos {1:2.2f}% apenas que possuem renda anual superior a 50.000.'.format(male_over*100, female_over*100))
#comparando horas semanais trabalhadas com relação ao sexo

compare_histogram(df_train, 'sex', 'hours.per.week')
male = df_train[df_train['sex'] == 'Male'] #filtrando quais são homens

female = df_train[df_train['sex'] == 'Female'] #filtrando quais são mulheres



#gerando histograma do ocupação dos homens

plt.figure(figsize=(13, 7))

male['occupation'].value_counts().plot(kind = 'bar')

plt.ylabel('Quantity',size=18)

plt.title('Histogram of male over occupations',size=20)



#gerando histograma do ocupação das mulheres

plt.figure(figsize=(13, 7))

female['occupation'].value_counts().plot(kind = 'bar',color='pink')

plt.ylabel('Quantity',size=18)

plt.title('Histogram of female over occupations',size=20)
#plotando gráfico de raça

plt.figure(figsize=(13, 7))

df_train['race'].value_counts().plot(kind = 'pie')
#comparando renda com relação à raça

compare_histogram(df_train, 'income', 'race')
#calculando número de indivíduos

white = df_train[df_train['race'] == 'White'].count()[0] #filtrando por raça branca

black = df_train[df_train['race'] == 'Black'].count()[0] #filtrando por raça egra

amer = df_train[df_train['race'] == 'Amer-Indian-Eskimo'].count()[0] #filtrando por raça indígena americana

other = df_train[df_train['race'] == 'Other'].count()[0] #filtrando por outras raças

asian = df_train[df_train['race'] == 'Asian-Pac-Islander'].count()[0] #filtrando por raça asiática insular



#aplicando filtro de renda à raça

white_income = []

temp = df_train[df_train['race'] == 'White']

white_income.append(temp[temp['income'] == '>50K'].count()[0]) #filtrando por renda > 50k

white_income.append(white-white_income[0])



black_income = []

temp = df_train[df_train['race'] == 'Black']

black_income.append(temp[temp['income'] == '>50K'].count()[0]) #filtrando por renda > 50k

black_income.append(black-black_income[0])



amer_income = []

temp = df_train[df_train['race'] == 'Amer-Indian-Eskimo']

amer_income.append(temp[temp['income'] == '>50K'].count()[0]) #filtrando por renda > 50k

amer_income.append(amer-amer_income[0])



asian_income = []

temp = df_train[df_train['race'] == 'Asian-Pac-Islander'] 

asian_income.append(temp[temp['income'] == '>50K'].count()[0]) #filtrando por renda > 50k

asian_income.append(asian-asian_income[0])



other_income = []

temp = df_train[df_train['race'] == 'Other']

other_income.append(temp[temp['income'] == '>50K'].count()[0]) #filtrando por renda > 50k

other_income.append(other-other_income[0])



print('Brancos:\n   {0:1.2f}% recebem mais de 50.000\n'.format(white_income[0]*100/white))

print('Negros:\n   {0:1.2f}% recebem mais de 50.000\n'.format(black_income[0]*100/black))

print('Amer-Indian-Eskimo:\n   {0:1.2f}% recebem mais de 50.000\n'.format(amer_income[0]*100/amer))

print('Asian-Pac-Islander:\n   {0:1.2f}% recebem mais de 50.000\n'.format(asian_income[0]*100/asian))

print('Outros:\n   {0:1.2f}% recebem mais de 50.000'.format(other_income[0]*100/other))
black = df_train[df_train['race'] == 'Black'] #filtrando indivíuduos negros

white = df_train[df_train['race'] == 'White'] #filtrando indivíuduos brancos



plt.figure(figsize=(13, 7))

white['occupation'].value_counts().plot(kind = 'bar')

plt.ylabel('Quantity',size=18)

plt.title('Histogram of white people over occupations',size=20)



plt.figure(figsize=(13, 7))

black['occupation'].value_counts().plot(kind = 'bar')

plt.ylabel('Quantity',size=18)

plt.title('Histogram of black people over occupations',size=20)
var1 = 'race'

var2 = 'education.num'



data = pd.concat([df_train[var2], df_train[var1]], axis=1)



f, ax = plt.subplots(figsize=(15, 7))



sns.boxplot(x=var1, y=var2, data=data, notch = True)

plt.title('Boxplot of education num over race',size=20)
over_50k = df_train[df_train['income'] == '>50K'] #filtrando rendas >50k

under_50k = df_train[df_train['income'] == '<=50K'] #filtrando rendas <=50k



plt.figure(figsize=(13, 7))

over_50k['occupation'].value_counts().plot(kind = 'bar')

plt.ylabel('Quantity',size=18)

plt.title('Histogram of income over 50K over occupations',size=20)



plt.figure(figsize=(13, 7))

under_50k['occupation'].value_counts().plot(kind = 'bar')

plt.ylabel('Quantity',size=18)

plt.title('Histogram of income under 50K over occupations',size=20)
#calculando número por renda

df_train['income'].value_counts()
#definindo base de treino

base = df_train
quantitative_columns = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']

qualitative_columns = ['education', 'marital.status', 'occupation', 'relationship', 'race',

                       'sex', 'native.country', 'income']
def isprivate(value):

    '''Verifica se um valor é Private. Retorna 1 para True e 0 caso contrário.'''

    if value == 'Private':

        return 1

    return 0



def catg(value, categories, ordenation = None):

    if ordenation is not None:

        ordenation = np.arange(0, len(categories))

    for pos in ordenation:

        if value == categories[pos]:

            return pos

    return -1



def equals(value, x):

    '''

    Recebe:

        1x valor

        1x lista

    Retorna:

        1, se a lista apresentar algum elemento igual ao valor

        0, se a lista não apresentar elementos igual ao valor

    '''

    for v in x:

        if v == value:

            return 1

    return 0
base['workclass'].unique() #valores do atributo workclass
# privado: 1 se trabalha para o privado, 0 caso contrario

private = pd.DataFrame({'private': base['workclass'].apply(isprivate)})
df_train['native.country'].value_counts() #calculando número de indivíduos de cada país
# usa: 1 se é sul_global, 0 caso contrário

countries = ['Mexico', 'Philippines', 'Puerto-Rico', 'El-Salvador', 'India', 'Cuba', 'Jamaica',

             'South', 'China', 'Dominican-Republic', 'Vietnam', 'Guatemala', 'Columbia', 'Taiwan',

             'Haiti', 'Iran', 'Nicaragua', 'Peru', 'Ecuador', 'Trinadad&Tobago', 'Cambodia',

             'Laos', 'Thailand', 'Yugoslavia', 'Outlying-US(Guam-USVI-etc)', 'Honduras']

sul_global = pd.DataFrame({'sul.global': base['native.country'].apply(equals, args = [countries])})
base['education'].unique() #valores do atributo education
edu_order = [15, 11, 5, 12, 10, 1, 14, 7, 2, 8, 4, 13, 0, 3, 6, 9]

args = [base['education'].unique(), edu_order]

education_classes = pd.DataFrame({'education.classes': base['education'].apply(catg, args = args)})
aux = pd.cut(base['hours.per.week'], bins = [-1, 25, 40, 60, 200], labels = [0, 1, 2, 3])

hours_per_week_clusters = pd.DataFrame({'hours.per.week.clusters': aux})

hours_per_week_clusters = hours_per_week_clusters.astype(np.int)
median = np.median(base[base['capital.gain'] > 0]['capital.gain'])

aux = pd.cut(base['capital.gain'],

             bins = [-1, 0, median, base['capital.gain'].max()+1],

             labels = [0, 1, 2])

capital_gain_clusters = pd.DataFrame({'capital.gain.clusters': aux})

capital_gain_clusters = capital_gain_clusters.astype(np.int)



median = np.median(base[base['capital.loss'] > 0]['capital.loss'])

aux = pd.cut(base['capital.loss'],

             bins = [-1, 0, median, base['capital.loss'].max()+1],

             labels = [0, 1, 2])

capital_loss_clusters = pd.DataFrame({'capital.loss.clusters': aux})

capital_loss_clusters = capital_loss_clusters.astype(np.int)
new_data = pd.concat([sul_global, private, education_classes, 

                      hours_per_week_clusters, capital_gain_clusters, 

                      capital_loss_clusters], axis = 1)
new_data.head()
#adicionando coluna de renda com os valores do rótulo

aux = base['income'].apply(equals, args = [['>50K']]) 

aux = pd.concat([new_data, pd.DataFrame({'income': aux})], axis = 1)

new = aux.astype(np.int)

aux.head()
#gerando matriz de correlação para análise da renda

corr_mat = aux.corr()

sns.set()

plt.figure(figsize=(10,8))

sns.heatmap(corr_mat, annot=True)
#excluindo atributos não utilizados na análise

base = base.drop(['fnlwgt', 'education', 'sex', 'native.country', 'workclass', 'marital.status'], axis = 1)

base.columns
#concatenando tabelas com os valores dos rótulos e a base selecionada

base = pd.concat([new_data, base], axis = 1)

base.head()
from sklearn import preprocessing as prep



names = ['occupation', 'relationship', 'race']

enc_x = []

for i in range(len(names)):

    enc_x.append(prep.LabelEncoder()) #faz o encode dos rótulos com valores entre 0 e n_classes-1

enc_y = prep.LabelEncoder()
i = 0

for name in names:

    base[name] = enc_x[i].fit_transform(base[name])

    i += 1

base['income'] = enc_y.fit_transform(base['income'])

base.head()
aux = base.astype(np.int)



corr_mat = aux.corr() #matriz de correlação

f, ax = plt.subplots(figsize=(20, 13))

sns.heatmap(corr_mat, vmax=.7, square=True, cmap="coolwarm", annot = True)
unselected_columns = [] #vetor que armazena colunas pouco relevantes

unselected_columns.append('capital.loss')

unselected_columns.append('capital.gain')

unselected_columns.append('sul.global')

unselected_columns.append('private')

unselected_columns.append('education.classes')

unselected_columns.append('hours.per.week.clusters')



base = base.drop(unselected_columns, axis = 1) #excluindo colunas pouco relevantes segundo a matriz de correlação

base.head()
aux = base.astype(np.int)
corr_mat = aux.corr() #nova matriz re correlação

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corr_mat, vmax=.7, square=True, cmap="coolwarm")
base.head()
from sklearn.preprocessing import StandardScaler
X = base.drop(['income'], axis = 1)

y = base['income']



scaler_x = StandardScaler() #normalizando os dados



X = scaler_x.fit_transform(X)
#importando bibliotecas para algoritmo kNN e cross-validation

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score
scores_mean = [] #vetor que armazena a média

scores_std = [] #vetor que armazena o desvio-padrão



k_lim_inf = 1

k_lim_sup = 30



folds = 5 



k_max = None

max_std = 0

max_acc = 0



i = 0

print('Finding best k...')

for k in range(k_lim_inf, k_lim_sup):

    

    KNNclf = KNeighborsClassifier(n_neighbors=k, p = 1) #definindo classificador kNN

    

    score = cross_val_score(KNNclf, X, y, cv = folds) #calculando a pontuação

    

    scores_mean.append(score.mean()) #armazenando a média das pontuações no vetor média

    scores_std.append(score.std()) #armazenando os desvios das pontuações no vetor desvio-padrão

    

    #identificando os que possuem melhor desempenho

    if scores_mean[i] > max_acc:

        k_max = k

        max_acc = scores_mean[i]

        max_std = scores_std[i]

    i += 1

    if not (k%3):

        print('   K = {0} | Best CV acc = {1:2.2f}% +/-{3:4.2f}% (best k = {2})'.format(k, max_acc*100, k_max, max_std*100))



#imprimindo o k com melhor desempenho

print('\nBest k: {}'.format(k_max))
plt.figure(figsize=(14, 7))

plt.errorbar(np.arange(k_lim_inf, k_lim_sup), scores_mean, scores_std,

             marker = 'o', markerfacecolor = 'purple' , linewidth = 3,

             markersize = 10, color = 'coral', ecolor = 'purple', elinewidth = 1.5)





yg = []

x = np.arange(0, k_lim_sup+1)

for i in range(len(x)):

    yg.append(max_acc)

plt.plot(x, yg, '--', color = 'purple', linewidth = 1)

plt.xlabel('k',size=18)

plt.ylabel('Accuracy',size=18)

plt.title('KNN performed on several values of k',size=20)

plt.axis([0, k_lim_sup, min(scores_mean) - max(scores_std), max(scores_mean) + 1.5*max(scores_std)])
k = k_max #definindo o melhor k



KNNclf = KNeighborsClassifier(n_neighbors=k, p = 1) #definindo classificador kNN

KNNclf.fit(X, y) #ajustando o classificador aos dados
#importando dados de teste

df_test = pd.read_csv("../input/adult-pmr3508/test_data.csv", na_values='?')



#setando Id como índice do DataFrame

df_test.set_index('Id', inplace = True)



#imprimindo 5 os primeiros elementos dos dados de teste

df_test.head()
#capital.gain.cluster

median = np.median(df_test[df_test['capital.gain'] > 0]['capital.gain'])

aux = pd.cut(df_test['capital.gain'],

             bins = [-1, 0, median, df_test['capital.gain'].max()+1],

             labels = [0, 1, 2])

capital_gain_clusters = pd.DataFrame({'capital.gain.clusters': aux})

capital_gain_clusters = capital_gain_clusters.astype(np.int)



#capital.loss.cluster

median = np.median(df_test[df_test['capital.loss'] > 0]['capital.loss'])

aux = pd.cut(df_test['capital.loss'],

             bins = [-1, 0, median, df_test['capital.loss'].max()+1],

             labels = [0, 1, 2])

capital_loss_clusters = pd.DataFrame({'capital.loss.clusters': aux})

capital_loss_clusters = capital_loss_clusters.astype(np.int)



new_data = pd.concat([capital_gain_clusters, capital_loss_clusters], axis = 1)
features = ['age', 'education.num', 'occupation', 'relationship', 'race', 'hours.per.week']



base_test = pd.concat([new_data, df_test[features]], axis = 1)



base_test.head()
#calculando número total de missing data na base de teste

total = base_test.isnull().sum().sort_values(ascending = False) 



#calculando porcentagem de missing data na base de teste

percent = ((base_test.isnull().sum()/base_test.isnull().count())*100).sort_values(ascending = False)



#concatenando resultados em um DataFrame

missing_data = pd.concat([total, percent], axis = 1, keys = ['Total', '%'])

missing_data
value = base_test['occupation'].describe().top #obtendo a moda

base_test['occupation'] = base_test['occupation'].fillna(value) #substituindo a moda nos dados faltantes

base_test
#verificando se os dados faltantes foram tratados

total = base_test.isnull().sum().sort_values(ascending = False)

percent = ((base_test.isnull().sum()/base_test.isnull().count())*100).sort_values(ascending = False)

missing_data = pd.concat([total, percent], axis = 1, keys = ['Total', '%'])

missing_data.head()
names = ['occupation', 'relationship', 'race']



i = 0

for name in names:

    base_test[name] = enc_x[i].transform(base_test[name])

    i += 1
base_test.head()
X_prev = scaler_x.transform(base_test.values) #vetor de predição
temp = KNNclf.predict(X_prev) #aplicando a vetor de predição no classificador



temp = enc_y.inverse_transform(temp)

temp = {'Income': temp}

predictions = pd.DataFrame(temp) #transformando a predição em um DataFrame

predictions.head()
#convertendo resultados para um arquivo submission.csv

predictions.to_csv("submission.csv", index = True, index_label = 'Id') 
#importando bibliotecas

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import numpy as np



#armazenando e exibindo resultados no notebook

%matplotlib inline
#importando dados de teste

df_train = pd.read_csv('../input/rs-costa-rican-household-poverty-level-prediction/train.csv')



#setando Id com índice do DataFrame

df_train.set_index('ID_num', inplace = True)

df_train.head()
#selecionando atributos

features = ['idhogar', 'parentesco1', 'escolari', 'SQBmeaned', 'hogar_nin', 'hogar_total', 'area1',

            'lugar1', 'cielorazo', 'pisonotiene', 'v14a', 'abastaguano', 'v2a1',

            'hacdor', 'meaneduc', 'SQBovercrowding', 'abastaguadentro',

            'tipovivi1', 'Target']

base = df_train[features] 
base = base[base['parentesco1'] == 1]

base.shape



base = base.drop(['idhogar', 'parentesco1'], axis = 1)
base = base.astype(np.float)

print('shape:', base.shape)

base.head()
corrmat = base.corr() #matriz de correlação

sns.set()

plt.figure(figsize=(13,9))

sns.heatmap(corrmat)
#calculando número total de missing data

total = base.isnull().sum().sort_values(ascending = False)



#calculando porcentagem de missing data

percent = ((base.isnull().sum()/base.isnull().count())*100).sort_values(ascending = False)



#concatenando resultados em um DataFrame

missing_data = pd.concat([total, percent], axis = 1, keys = ['Total', '%'])

missing_data.head()
base = base.drop(['v2a1'], axis = 1) #eliminando atributo v2a1
base['SQBmeaned'].plot(kind = 'box')

col = 'SQBmeaned'

base[col] = base[col].fillna(base[col].describe().mean()) #substituindo missing data pela média
base['meaneduc'].plot(kind = 'box')

col = 'meaneduc'

base[col] = base[col].fillna(base[col].describe().mean()) #substituindo missing data pela média
#calculando número total de missing data

total = base.isnull().sum().sort_values(ascending = False)



#calculadno porcentagem de missing data

percent = ((base.isnull().sum()/base.isnull().count())*100).sort_values(ascending = False)



#concatenando resultados em um Data Frame

missing_data = pd.concat([total, percent], axis = 1, keys = ['Total', '%'])

missing_data.head()
def labels(x):

    '''

    Função que recebe um valor e retorna o respectivo rótulo

    '''

    if x == 1.0:

        return 'extreme poverty'

    if x == 2.0:

        return 'moderate poverty'

    if x == 3.0:

        return 'vulnerable households'

    return 'non vulnerable households'
base['Target'] = base['Target'].apply(labels)
cols = ['escolari', 'SQBmeaned', 'area1', 'lugar1', 'hogar_nin', 'hogar_total', 'SQBovercrowding']



sns.set()

sns.pairplot(base, hue = 'Target', vars = cols)
var2 = 'escolari'

var1 = 'Target'



f, ax = plt.subplots(figsize=(14, 8))



sns.boxplot(x=var1, y=var2, data=base, order = ['extreme poverty', 'moderate poverty', 'vulnerable households', 'non vulnerable households'])

plt.title('Boxplot of escolari over Target',size=20)
var2 = 'hogar_nin'

var1 = 'Target'



f, ax = plt.subplots(figsize=(14, 8))



sns.boxplot(x=var1, y=var2, data=base, order = ['extreme poverty', 'moderate poverty', 'vulnerable households', 'non vulnerable households'])

plt.title('Boxplot of hogar_nin over Target',size = 20)
var2 = 'SQBmeaned'

var1 = 'Target'



f, ax = plt.subplots(figsize=(14, 8))



sns.boxplot(x=var1, y=var2, data=base, order = ['extreme poverty', 'moderate poverty', 'vulnerable households', 'non vulnerable households'])

plt.title('Boxplot of escolari over Target',size = 20)
base.hist(column='cielorazo', by ='Target', figsize=(10,10))
base.hist(column='pisonotiene', by ='Target', figsize=(10,10))
base.hist(column='v14a', by ='Target', figsize=(10,10))
corrmat = base.corr() #matriz de correlação

sns.set()

plt.figure(figsize=(13,10))

sns.heatmap(corrmat)
base = base.drop(['cielorazo', 'v14a', 'abastaguano', 'tipovivi1', 'hacdor'], axis = 1)

base.head()
from sklearn.preprocessing import StandardScaler
X = base.drop('Target', axis = 1)

y = base['Target']
scaler_x = StandardScaler() #definindo escalonador



X = scaler_x.fit_transform(X) #ajustando os dados ao escalonador
from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsClassifier
scores_mean = [] #vetor de médias

scores_std = [] #vetor desvio-padrão



k_lim_inf = 1

k_lim_sup = 36



folds = 5



k_max = None

max_acc = 0



i = 0

print('Finding best k...')

for k in range(k_lim_inf, k_lim_sup):

    

    KNNclf = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='manhattan', p=2)

    

    score = cross_val_score(KNNclf, X, y, cv = folds) #calculando valor

    

    scores_mean.append(score.mean()) #adicionando média ao vetor correspondente

    scores_std.append(score.std()) #adicionando desvio-padrão ao vetor correspondente

    

    if scores_mean[i] > max_acc: #definindo o melhor k

        k_max = k

        max_acc = scores_mean[i]

    i += 1

    if not (k%3):

        print('   K = {0} | Best CV acc = {1:2.2f}% (best k = {2})'.format(k, max_acc*100, k_max))

print('\nBest k: {}'.format(k_max))
plt.figure(figsize=(14, 7))

plt.errorbar(np.arange(k_lim_inf, k_lim_sup), scores_mean, scores_std,

             marker = 'o', markerfacecolor = 'purple' , linewidth = 3,

             markersize = 10, color = 'coral', ecolor = 'purple', elinewidth = 1.5)





yg = []

x = np.arange(0, k_lim_sup+1)

for i in range(len(x)):

    yg.append(max_acc)

plt.plot(x, yg, '--', color = 'purple', linewidth = 1)

plt.xlabel('k',size=18)

plt.ylabel('accuracy',size=18)

plt.title('KNN performed on several values of k',size=20)

plt.axis([0, k_lim_sup, min(scores_mean) - max(scores_std), max(scores_mean) + 1.5*max(scores_std)])