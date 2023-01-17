import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns



import sklearn

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score

from sklearn import preprocessing
# Lendo o arquivo de treino



#train_data_raw = pd.read_csv("train_data.csv", sep=r'\s*,\s*', engine='python', na_values="?")

train_data_raw = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv", sep=r'\s*,\s*', engine='python', na_values="?")



train_data_raw.head(2)
# Preparando os dados de Treino



train_data = train_data_raw



# Tirando a coluna 'Id' e Renomeando colunas. Note que 'income' é o 'Target'. 



train_data = train_data.drop('Id', axis = 1)

train_cols_names = ["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

                    "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

                    "Hours per week", "Country", "Target"]

train_data.columns = train_cols_names



print('Dimensões : ',train_data.shape)
# É necessário lidar com dados faltantes do arquivo de Treino



train_data.isna().sum()
# Como são poucos valores faltantes, iremos apenas excluir essas linhas



train_data = train_data.dropna()

print('Dimensões s/ valores na : ', train_data.shape)
# Agora, fazendo a mesma preparação para os dados de Teste



#test_data_raw = pd.read_csv("test_data.csv", sep=r'\s*,\s*', engine='python', na_values="?")

test_data_raw = pd.read_csv("/kaggle/input/adult-pmr3508/test_data.csv", sep=r'\s*,\s*', engine='python', na_values="?")



test_data = test_data_raw



# Tirando a coluna 'Id' e Renomeando colunas Note que não há label 'Target' para o teste



test_data = test_data.drop('Id', axis = 1)

test_cols_names = ["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

                   "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

                   "Hours per week", "Country"]

#test_cols_names = train_cols_names[:-1]

test_data.columns = test_cols_names



test_data.head(2)
test_data.shape
test_data.isna().sum()
test_data['Workclass'].describe()
test_data['Occupation'].describe()
test_data['Country'].describe()
# Alocando a  moda para 'Country' e 'Workclass'



top = test_data['Workclass'].describe().top

test_data['Workclass'] = test_data['Workclass'].fillna(top)



top = test_data['Country'].describe().top

test_data['Country'] = test_data['Occupation'].fillna(top)
test_data['Occupation'].value_counts().plot(kind = 'pie')

plt.title('Distribuição de "Occupation" da base de teste')
# De forma randômica



#Occup_list = test_data['Occupation'].dropna().unique()

#test_data['Occupation'] = test_data['Occupation'].fillna(pd.Series(np.random.choice(Occup_list, 

#                                                                                    size=len(test_data.index))))



# Alocação pela moda para 'Occupation'



top = test_data['Occupation'].describe().top

test_data['Occupation'] = test_data['Occupation'].fillna(top)
test_data['Occupation'].value_counts().plot(kind = 'pie')

plt.title('Distribuição de "Occupation" da base de teste com valores faltantes ajustados')
# Fazendo uma análise para dados numéricos

numeric_cols = ["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]

train_data[numeric_cols].describe(include = 'all')
non_numeric_cols = ["Workclass", "Education", "Martial Status",

                    "Occupation", "Relationship", "Race", "Sex", "Country", "Target"]

train_data[non_numeric_cols].describe()
aux_data = train_data.apply(preprocessing.LabelEncoder().fit_transform)

corr = aux_data.corr()

corr.style.background_gradient(cmap = 'Reds')
train_data.groupby(['Age','Target']).size().unstack(fill_value=0).plot.bar(width = 1, figsize=(15,7))
train_data.groupby(['Workclass','Target']).size().unstack(fill_value=0).plot.bar(width = 1, figsize=(10,5), stacked = True,

                                                                           edgecolor = 'white')
pct_wkcl = (train_data.groupby(['Workclass','Target'])['Target'].count()/train_data.groupby(['Workclass'])['Target'].count())*100

pct_wkcl.unstack(fill_value=0).plot.bar(width = 1, figsize=(10,5), stacked = True,

                                               edgecolor = 'white')

#pct_wkcl
pct_edu = (train_data.groupby(['Education','Target'])['Target'].count()/train_data.groupby(['Education'])['Target'].count())*100

pct_edu.unstack(fill_value=0).plot.bar(width = 1, figsize=(10,5), stacked = True,

                                               edgecolor = 'white')
pct_edunum = (train_data.groupby(['Education-Num','Target'])['Target'].count()/train_data.groupby(['Education-Num'])['Target'].count())*100

pct_edunum.unstack(fill_value=0).plot.bar(width = 1, figsize=(10,5), stacked = True,

                                               edgecolor = 'white')
train_data.groupby(['Martial Status','Target']).size().unstack(fill_value=0).plot.bar(width = 1, figsize=(10,5), stacked = True,

                                                                           edgecolor = 'white')
train_data.groupby(['Occupation','Target']).size().unstack(fill_value=0).plot.bar(width = 1, figsize=(10,5), stacked = True,

                                                                           edgecolor = 'white')
pct_occ = (train_data.groupby(['Occupation','Target'])['Target'].count()/train_data.groupby(['Occupation'])['Target'].count())*100

pct_occ.unstack(fill_value=0).plot.bar(width = 1, figsize=(10,5), stacked = True,

                                               edgecolor = 'white')
train_data.groupby(['Relationship','Target']).size().unstack(fill_value=0).plot.bar(width = 1, figsize=(10,5), stacked = True,

                                                                           edgecolor = 'white')
train_data.groupby(['Race','Target']).size().unstack(fill_value=0).plot.bar(width = 1, figsize=(10,5), stacked = True,

                                                                            edgecolor = 'white')
train_data.groupby(['Sex','Target']).size().unstack(fill_value=0).plot.bar(width = 1, figsize=(10,5), stacked = True,

                                                                        edgecolor = 'white')

train_data.groupby(['Country','Target']).size().unstack(fill_value=0).plot.bar(width = 1, figsize=(10,5), stacked = True,

                                                                        edgecolor = 'white')

# Determinando os atributos e labels utilizados para determinar o melhor k

train_cols_names = ["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

                    "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

                    "Hours per week", "Country", "Target"]



use_cols = ["Age", "Education-Num", "Martial Status",

             "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

             "Hours per week"]



Xtrain = train_data[use_cols].apply(preprocessing.LabelEncoder().fit_transform)

Ytrain = train_data['Target']
# Determinando o melhor k do classificado utilizando 5 folders 



def find_k_best(Xtrain, Ytrain) : 

    

    cv_folds = 5

    k_best = None

    accuracy_best = 0



    k_list = np.array(range(1,36,1))

    accuracy_list = np.array([])



    for k in k_list:

    

    # Utilizando validação cruzada e determinando aquele com melhor acurácia

    

        knn = KNeighborsClassifier(n_neighbors = k)

        scores = cross_val_score(knn, Xtrain, Ytrain, cv = cv_folds)

        accuracy = scores.mean()

        accuracy_list = np.append(accuracy_list, accuracy)

        

        if accuracy  > accuracy_best:

            k_best = k

            accuracy_best = accuracy

        

    return k_best, k_list, accuracy_best, accuracy_list



#k_best, k_list, accuracy_best, accuracy_list = find_k_best(Xtrain, Ytrain)

        
# Guardando a resposta obtida 

accuracy_list = np.array([0.79924424, 0.82062929, 0.81940255, 0.82752558, 0.82639831,

                          0.83011179, 0.8275255 , 0.83163695, 0.82861972, 0.83170316,

                          0.83014485, 0.83140477, 0.83080803, 0.83273103, 0.83110643,

                          0.83302946, 0.83196844, 0.8322337 , 0.83243262, 0.83316206,

                          0.83273102, 0.83326152, 0.83289679, 0.83306254, 0.83249888,

                          0.83220042, 0.83220047, 0.83176948, 0.83190211, 0.83233312,

                          0.83147112, 0.83200164, 0.83203479, 0.83239949, 0.83147119])



accuracy_best = 0.8332615244800812



k_list = np.array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,

                   18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,

                   35])



k_best = 22
plt.figure(20, figsize=(15,7))



plt.plot(accuracy_list, '-o', label = 'Acurácia')

plt.plot(accuracy_best*np.ones(len(k_list)), '-.', label = 'Melhor Acurácia')



plt.xticks(list(range(0,35)), k_list ,alpha = 1)

plt.xlim(-0.5, 34.5)



plt.title('Influência do k na acurácia')

plt.xlabel('k')

plt.ylabel('Acurácia')

plt.grid(color='gray', linestyle = '-', linewidth = 1, alpha = 0.5)

plt.legend()

plt.show()
print('O melhor k foi estimado em k = {:2.0f} com acurácia de {:2.2f}%'.format(k_best, accuracy_best*100))
# Preprocessamento de dados não numéricos e escala



train_cols_names = ["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

                    "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

                    "Hours per week", "Country", "Target"]



use_cols = ["Age", "Education-Num", "Martial Status",

             "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

             "Hours per week"]



Xnum_train = train_data[use_cols].apply(preprocessing.LabelEncoder().fit_transform)

Ynum_train = train_data['Target']



Xnum_test = test_data[use_cols].apply(preprocessing.LabelEncoder().fit_transform)



#Xnum_train = train_data[train_cols_names[:-1]].apply(preprocessing.LabelEncoder().fit_transform)

#Ynum_train = train_data['Target']



#Xnum_test = test_data[test_cols_names].apply(preprocessing.LabelEncoder().fit_transform)
# Treinando o Classificador



k = 22

knn = KNeighborsClassifier(n_neighbors = k)

knn.fit(Xnum_train, Ynum_train)



# Predição



Ypred = knn.predict(Xnum_test)

Ypred
id_index = pd.DataFrame({'Id' : list(range(len(Ypred)))})

df_Ypred = pd.DataFrame({'income' : Ypred})

result = id_index.join(df_Ypred)
result
result['income'].value_counts()
result.to_csv("submission.csv", index = False)