# Todas as bibliotecas usadas na tarefa estão aqui.



import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from sklearn import preprocessing as prep

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder as le

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from subprocess import check_output

%matplotlib inline

plt.style.use('seaborn')
# Aqui coloco algumas funções úteis para categorização e filtragem.



def isprivate(value):

    if value == 'Private':

        return 1

    return 0



def isUSA(value):

    if value == 'United-States':

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

    for v in x:

        if v == value:

            return 1

    return 0



def filtro_binario(valor, alvo):

    if valor == alvo:

        return 1

    return 0



def equals(value, x):

    for v in x:

        if v == value:

            return 1

    return 0
# Leitura do arquivo base e identificação de nulos



dados_treino = pd.read_csv("../input/adult-pmr3508/train_data.csv", na_values = '?')

dados_treino.set_index('Id',inplace=True)

dados_treino.info()

print('\nDataFrame (linhas,colunas):', dados_treino.shape)
# Tratamento dos dados faltantes



valor_workclass = dados_treino['workclass'].describe().top

dados_treino['workclass'] = dados_treino['workclass'].fillna(valor_workclass)



valor_native_country = dados_treino['native.country'].describe().top

dados_treino['native.country'] = dados_treino['native.country'].fillna(valor_native_country)



valor_occupation = dados_treino['occupation'].describe().top

dados_treino['occupation'] = dados_treino['occupation'].fillna(valor_occupation)
# Eliminação de duplicatas



dados_treino.drop_duplicates(keep='first', inplace=True)
# Pós-tratamento para nulos e duplicatas



dados_treino.info()

print('\nDataFrame (linhas,colunas):', dados_treino.shape)
# Divido as variáveis em qualitativas e quantitativas e crio um auxiliar "base"

# Para que, logo abaixo, seja feito a categorização do



base = dados_treino

quantitative_columns = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']

qualitative_columns = ['education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country', 'income']

base.columns
# Native country

# Como percebemos, temos uma grande discrepância entre o número de pessoas consideradas nos EUA.

# Usaremos então uma divisão entre estadounidenses e os demais.

dados_treino['native.country'].value_counts()

USA = pd.DataFrame({'USA': base['native.country'].apply(isUSA)})



# Workclass

# Privado: 1 se trabalha para o privado, 0 caso contrario

private = pd.DataFrame({'private': base['workclass'].apply(isprivate)})



# Hours per week

# Se torna mais relevante clusterizar as quantidades de horas trabalhadas.

# O agrupamento permite melhor correlação futuramente.

aux = pd.cut(base['hours.per.week'], bins = [-1, 25, 40, 60, 200], labels = [0, 1, 2, 3])

hours_per_week_clusters = pd.DataFrame({'hours.per.week.clusters': aux})

hours_per_week_clusters = hours_per_week_clusters.astype(np.int)



# Para Capital gain e Capital loss, temos variáveis contínuas.

# Sendo assim, para haver a correlação futura, clusterizo essas variáveis.



# Capital gain

median = np.median(base[base['capital.gain'] > 0]['capital.gain'])

aux = pd.cut(base['capital.gain'],

             bins = [-1, 0, median, base['capital.gain'].max()+1],

             labels = [0, 1, 2])

capital_gain_clusters = pd.DataFrame({'capital.gain.clusters': aux})

capital_gain_clusters = capital_gain_clusters.astype(np.int)



# Capital loss

median = np.median(base[base['capital.loss'] > 0]['capital.loss'])

aux = pd.cut(base['capital.loss'],

             bins = [-1, 0, median, base['capital.loss'].max()+1],

             labels = [0, 1, 2])

capital_loss_clusters = pd.DataFrame({'capital.loss.clusters': aux})

capital_loss_clusters = capital_loss_clusters.astype(np.int)



base['education'].unique()
# Education

# Ordeno os diversos níveis de escolaridade com uma variável ordinal

# (conforme array acima)



edu_order = [15, 11, 5, 12, 10, 1, 14, 7, 2, 8, 4, 13, 0, 3, 6, 9]

args = [base['education'].unique(), edu_order]

education_classes = pd.DataFrame({'education.classes': base['education'].apply(catg, args = args)})
# Defino 'new_data' para usar logo depois

new_data = pd.concat([USA, private, education_classes, 

                      hours_per_week_clusters, capital_gain_clusters, 

                      capital_loss_clusters], axis = 1)
aux = base['income'].apply(equals, args = [['>50K']])

aux = pd.concat([new_data, pd.DataFrame({'income': aux})], axis = 1)

new = aux.astype(np.int)

aux.head()
# Matriz de correlação

# Uma análise rápida já nos mostra a relação entre renda e capital recebida (clusterizada).

# O mesmo acontece com horas trabalhadas (clusterizadas)

corr_mat = aux.corr(method ='pearson')

corr_mat

sns.set()

plt.figure(figsize=(10,8))

sns.heatmap(corr_mat, annot=True)
base
# 

base = base.drop(['fnlwgt', 'education', 'sex', 'native.country', 'workclass', 'marital.status'], axis = 1)

base.columns
base = pd.concat([new_data, base], axis = 1)

base.head()
names = ['occupation', 'relationship', 'race']

enc_x = []

for i in range(len(names)):

    enc_x.append(prep.LabelEncoder())

enc_y = prep.LabelEncoder()
i = 0

for name in names:

    base[name] = enc_x[i].fit_transform(base[name])

    i += 1



base['income'] = enc_y.fit_transform(base['income'])
aux = base.astype(np.int)



corr_mat = aux.corr()

f, ax = plt.subplots(figsize=(20, 13))

sns.heatmap(corr_mat, vmax=.7, square=True, cmap="coolwarm", annot = True)
unselected_columns = []

unselected_columns.append('capital.loss')

unselected_columns.append('capital.gain')

unselected_columns.append('USA')

unselected_columns.append('private')

unselected_columns.append('education.classes')

unselected_columns.append('hours.per.week.clusters')



base = base.drop(unselected_columns, axis = 1)

# KNN



base.shape
X = base.drop(['income'], axis = 1)

y = base['income']
scaler_x = StandardScaler()



X = scaler_x.fit_transform(X)
scores_mean = []

scores_std = []

# De qual k até qual será testado.

k_lim_inf = 1

k_lim_sup = 30

# Divisão dos folds para não haver overfitting.

folds = 5

# Pertinentes para encontrar a maior acurácia, usado no loop abaixo.

k_max = None

max_std = 0

max_acc = 0



# Otimização do k

i = 0

print('Encontrando o k com melhor taxa de acerto (score):')

for k in range(k_lim_inf, k_lim_sup):

    # Aplicação do KNN

    KNNclf = KNeighborsClassifier(n_neighbors=k, p = 1)

    # Cross-validation com divisão em 5 folds

    score = cross_val_score(KNNclf, X, y, cv = folds)

    # Guarda a taxa de acerto com sua margem de erro

    scores_mean.append(score.mean())

    scores_std.append(score.std())

    # Se a taxa de acerto (score) for melhor, substitui como o melhor até o momento.

    if scores_mean[i] > max_acc:

        k_max = k

        max_acc = scores_mean[i]

        max_std = scores_std[i]

    i += 1

    if not (k%3):

        print('   K = {0} | Acurácia = {1:2.2f}% +/-{3:4.2f}% (até agora, melhor k = {2})'.format(k, max_acc*100, k_max, max_std*100))

print('\nO k com melhor score é: {}'.format(k_max))
#Faço o fit usando o k otimizado.

k = k_max



KNNclf = KNeighborsClassifier(n_neighbors=k, p = 1)

KNNclf.fit(X, y)
# Leitura do arquivo de teste do modelo e identificação de nulos



df_test = pd.read_csv("../input/adult-pmr3508/test_data.csv", na_values='?')

df_test.set_index('Id', inplace = True)

df_test.head()
# capital.gain.cluster (semelhante ao feito com train_data)

median = np.median(df_test[df_test['capital.gain'] > 0]['capital.gain'])

aux = pd.cut(df_test['capital.gain'],

             bins = [-1, 0, median, df_test['capital.gain'].max()+1],

             labels = [0, 1, 2])

capital_gain_clusters = pd.DataFrame({'capital.gain.clusters': aux})

capital_gain_clusters = capital_gain_clusters.astype(np.int)



# capital.loss.cluster (semelhante ao feito com train_data)

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
total = base_test.isnull().sum().sort_values(ascending = False)

percent = ((base_test.isnull().sum()/base_test.isnull().count())*100).sort_values(ascending = False)

missing_data = pd.concat([total, percent], axis = 1, keys = ['Total', '%'])

missing_data.head()
value = base_test['occupation'].describe().top

base_test['occupation'] = base_test['occupation'].fillna(value)
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
X_prev = scaler_x.transform(base_test.values)
temp = KNNclf.predict(X_prev)



temp = enc_y.inverse_transform(temp)

temp = {'Income': temp}

predictions = pd.DataFrame(temp)
predictions.head()
predictions.to_csv("submission.csv", index = True, index_label = 'Id')
dados_analise = pd.read_csv("../input/adult-pmr3508/train_data.csv", na_values = '?')
dados_analise["sex"].value_counts().plot(kind="bar")
dados_analise["race"].value_counts().plot(kind="bar")