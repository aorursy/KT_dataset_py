# Bibliotecas: 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

import scikitplot as skplt

from sklearn.metrics import accuracy_score

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Carregando os dados



df = pd.read_csv('/kaggle/input/costa-rican-household-poverty-prediction/train.csv')

test = pd.read_csv('/kaggle/input/costa-rican-household-poverty-prediction/test.csv')



df.shape, test.shape
# Juntando os dataframes



df_all = df.append(test)



df_all.shape
# Verificando as cinco primeiras linhas do dataset: 



df_all.head()
# Quais colunas do dataframe são do tipo object



df_all.select_dtypes('object').head()
# Olhando a coluna dependency



df_all['dependency'].value_counts()
# Analisando os dados da coluna edjefa



df_all['edjefa'].value_counts()
# Analisando os dados da coluna edjefe



df_all['edjefe'].value_counts()
# Vamos transformar 'yes' em 1 e 'no' em 0 nas colunas edjefa e edjefe



mapeamento = {'yes': 1, 'no': 0}



df_all['edjefa'] = df_all['edjefa'].replace(mapeamento).astype(int)

df_all['edjefe'] = df_all['edjefe'].replace(mapeamento).astype(int)
# Olhando a coluna dependency



df_all['dependency'].value_counts()
# Vamos transformar 'yes' em 1 e 'no' em 0 na coluna dependency



df_all['dependency'] = df_all['dependency'].replace(mapeamento).astype(float)
# Verificando se ainda há colunas do dataframe do tipo object para transformar:



df_all.select_dtypes('object').head()
# Verificando algumas informações do dataset:



df_all.info()
# Verificando a porcentagem total de valores nulos no dataset:



total = np.product(df_all.shape)

miss_values = df_all.isnull().sum().sum()

porcentagem_total = (miss_values/total) * 100

print(f'A porcentagem total de valores nulos do dataset é de: {(porcentagem_total)}')
# Verificando a porcentagem de valores nulos por variável:



total_num = df_all.isnull().sum().sort_values(ascending = False)



perc = df_all.isnull().sum()/df_all.isnull().count()*100



perc1 = (round(perc,2).sort_values(ascending = False))



#Criando o dataframe: 



df_miss = pd.concat([total_num, perc1], axis = 1, keys = ['Total de Valores Nulos', 'Porcentagem %']).sort_values(by = 'Porcentagem %', ascending = False)



top_miss = df_miss[df_miss['Porcentagem %'] > 0]

top_miss.reset_index(inplace = True)



top_miss
# Variável rez_esc:



df_all.rez_esc.value_counts()
df_all.loc[df_all['rez_esc'].notnull()]['age'].describe()
df_all.loc[df_all['rez_esc'].isnull()]['age'].describe()
# Tratamento da coluna rez_esc:



df_all.loc[((df_all['age'] > 19) | (df_all['age'] < 7)) & (df_all['rez_esc'].isnull()), 'rez_esc'] = 0
df_all['rez_esc-missing'] = df_all['rez_esc'].isnull()
# Tratando os outliers da coluna:



df_all.loc[df_all['rez_esc'] > 5, 'rez_esc'] = 5
df_all.loc[df_all['rez_esc'].isnull()]['age'].describe()
df_all['rez_esc'].fillna(0, inplace = True)
# Variável v18q1:



df_all.v18q1.value_counts()
# Vamos verificar outra variável relacionada

# v18q: owns a tablet (0 indica que a pessoa não possui tablet/ 1 indica que a pessoa possui tablet)



df_all.v18q.value_counts()
# Imputando o valor 0 na variável v18q1:



df_all['v18q1'].fillna(0, inplace = True)
sns.distplot(a = df_all['v18q1'], kde = False)
# Variável v2a1: 



df_all.v2a1.value_counts()
# Verificando quantas pessoas alugam as casas (tipovivi3, = 1 rented):



df_all.tipovivi3.value_counts().sort_values()
# Verificando quantas pessoas possuem casa própria (tipovivi1, =1 own and fully paid house):



df_all.tipovivi1.value_counts().sort_values()
# Quantidade de casas que ainda estão sendo pagas (tipovivi2, "=1 own,  paying in installments"):



df_all.tipovivi2.value_counts().sort_values()
# tipovivi4, =1 precarious



df_all.tipovivi4.value_counts()
# tipovivi5, "=1 other(assigned,  borrowed)"



df_all.tipovivi5.value_counts()
# Imputando o valor 0 na variável v2a1:



df_all['v2a1'].fillna(0, inplace=True)
# Tratamento SQmeaned e meaneduc:



df_all['SQBmeaned'].fillna(-1, inplace=True)

df_all['meaneduc'].fillna(-1, inplace=True)
# Verificando a porcentagem total de valores nulos no dataset:



total = np.product(df_all.shape)

miss_values = df_all.isnull().sum().sum()

porcentagem_total = (miss_values/total) * 100

print(f'A porcentagem total de valores nulos do dataset é de: {(porcentagem_total)}')
# Verificando a quantidade de casas do dataset:



df_all.idhogar.nunique()
# Verificando a quantidade de chefes de família: 



df_all.parentesco1.value_counts()
# Verificando quantas pessoas têm em quantas casas:

# r4t3: Total persons in the household



(

    df_all.groupby('r4t3').idhogar.nunique()

    .reset_index()

    .rename(columns = {'r4t3':'Quantidade de Pessoas na Casa', 'idhogar':'Quantidade de Casas'})

    .sort_values(by = ['Quantidade de Casas'], ascending = True)

    .reset_index()

    .drop(['index'], axis = 1)

    

)
# Verificando a quantidade de pessoas que vivem na casa:

# tamviv, number of persons living in the household



(

    df_all.groupby('tamviv').idhogar.nunique()

    .reset_index()

    .rename(columns={'tamviv':'Quantidade de Pessoas que vivem na Casa', 'idhogar':'Quantidade de Casas'})

    .sort_values(by=['Quantidade de Casas'], ascending=True)

    .reset_index()

    .drop(['index'], axis = 1)

    

)
# Verificando a quantidade de casas por nível de renda: 



(

df_all.groupby('Target').idhogar.count()

)
# Visualizando a distribuição da Target por meio de um gráfico:



niveis_renda = df_all['Target'].value_counts().sort_index()



from collections import OrderedDict



cores = OrderedDict({1: 'red', 2: 'orange', 3: 'blue', 4: 'green'})



niveis_renda.plot.bar(figsize = (8, 6), 

                      color = cores.values(),

                      edgecolor = 'k', linewidth = 2)



plt.xlabel('Níveis de Renda')

plt.ylabel('Quantidade')

plt.xticks(rotation = 70) 

plt.title('Distribuição dos Níveis de Renda')
# Verificando a quantidade de meninos mais jovens que 12 anos de idade por casa:

# r4h1, Males younger than 12 years of age



(

    df_all.groupby('r4h1').idhogar.nunique()

    .reset_index()

    .rename(columns={'r4h1':'Meninos mais jovens que 12 anos de idade','idhogar':'Quantidade de Casas'})

    

)
# Verificando a quantidade de meninos de 12 anos e mais velhos:

# r4h2, Males 12 years of age and older



(

    df_all.groupby('r4h2').idhogar.nunique()

    .reset_index()

    .rename(columns={'r4h2':'Homens com 12 anos de idade ou mais velhos','idhogar':'Quantidade de Casas'})

    

)
# Verificando o total de homens na casa:

# r4h3, Total males in the household



(

    df_all.groupby('r4h3').idhogar.nunique()

    .reset_index()

    .rename(columns={'r4h3':'Total de Homens na Casa','idhogar':'Quantidade de Casas'})

    

)
# Quantidade de meninas mais jovens que 12 anos de idade:

# r4m1, Females younger than 12 years of age



(

    df_all.groupby('r4m1').idhogar.nunique()

    .reset_index()

    .rename(columns={'r4m1':'Meninas mais jovens que 12 anos de idade','idhogar':'Quantidade de Casas'})

    

)
# Meninas de 12 anos de idade e mais velhas: 

# r4m2, Females 12 years of age and older



(

    df_all.groupby('r4m2').idhogar.nunique()

    .reset_index()

    .rename(columns={'r4m2':'Meninas de 12 anos de idade ou mais velhas','idhogar':'Quantidade de Casas'})

    

)
# Total de mulheres na casa:

# r4m3, Total females in the household



(

    df_all.groupby('r4m3').idhogar.nunique()

    .reset_index()

    .rename(columns={'r4m3':'Total de Mulheres na Casa','idhogar':'Quantidade de Casas'})

    

)
# Visualizando a quantidade de pessoas mais jovens que 12 anos na casa:



plt.figure(figsize = (8, 6))

sns.distplot(a = df_all['r4t1'])

plt.title('Número de pessoas mais jovens que 12 anos na casa')

plt.xlabel('Quantidade')
# Visualizando a quantidade de pessoas de 12 anos ou mais velhas na casa:



plt.figure(figsize = (8, 6))

sns.distplot(a = df_all['r4t2'])

plt.title('Número de pessoas de 12 anos ou mais velhas na casa')

plt.xlabel('Quantidade')
# Visualizando o total de pessoas na casa: 

# r4t3, Total persons in the household



plt.figure(figsize = (8, 6))

sns.distplot(a = df_all['r4t3'])

plt.title('Total de Pessoas na Casa')

plt.xlabel('Quantidade')
# Verificando a quantidade de anos de educação de chefas em relação ao nível de renda: 

# 1 = extreme poverty

# 2 = moderate poverty

# 3 = vulnerable households

# 4 = non vulnerable households



edjefas_renda = pd.pivot_table(

df_all, 

columns = 'Target',

index = 'edjefa',

values = 'Id',

aggfunc = {'Id':'count'}

)



edjefas_renda
# Verificando a quantidade de anos de educação de chefes em relação ao nível de renda: 

# 1 = extreme poverty

# 2 = moderate poverty

# 3 = vulnerable households

# 4 = non vulnerable households



edjefes_renda = pd.pivot_table(

df_all, 

columns = 'Target',

index = 'edjefe',

values = 'Id',

aggfunc = {'Id':'count'}

)



edjefes_renda
# Verificar a quantidade de quartos pela média de aluguel:



room_rent = (

    df_all.groupby('rooms').v2a1.mean()

).reset_index()



room_rent.rename(columns = {'rooms':'Quantidade de Quartos', 'v2a1':'Média do Aluguel Mensal'})
# Verificando a correlação entre variáveis: 

# Correlação tamanho da casa: 



corr_casa = df_all[['tamhog','tamviv','hhsize','r4t3']]



corr_casa.corr()
# Visualizando a correlação: 



sns.lmplot('tamhog', 'hhsize', data = df_all, fit_reg = True);

plt.title('Size of the Household vs Household Size')
# Visualizando a correlação: 



sns.lmplot('tamviv', 'r4t3', data = df_all, fit_reg = True);

plt.title('Number of Persons living in the Household vs Total Persons in the Household')
# Retirando as variáveis hhsize e r4t3: 



df_all = df_all.drop(['hhsize','r4t3'], axis = 1)
# Verificando a correlação entre a variável idade e idade ao quadrado:



corr_idade = df_all[['age','SQBage']].corr()

print(f'A correlação entre age e SQBage é de: \n{(corr_idade)}')
# Visualizando a correlação: 



sns.lmplot('age', 'SQBage', data = df_all, fit_reg = False);

plt.title('Squared Age versus Age')
# Retirando a variável age: 



df_all = df_all.drop(['age'], axis = 1)
# Separando as colunas para treinamento:



feats = [c for c in df_all.columns if c not in ['Id', 'idhogar','Target']]
# Separar os dataframes:



train, test = df_all[~df_all['Target'].isnull()], df_all[df_all['Target'].isnull()]



train.shape, test.shape
# Instanciando o random forest classifier



from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(max_depth=None, random_state=42, n_jobs=4, n_estimators=700,

                            min_impurity_decrease=1e-3, min_samples_leaf=2,

                            verbose=0, class_weight='balanced')
# Treinando o modelo



rf.fit(train[feats], train['Target'])
# Prever o Target de teste usando o modelo treinado



test['Target'] = rf.predict(test[feats]).astype(int)
# Vamos verificar as previsões



test['Target'].value_counts(normalize=True)
# Avaliando a importancia de cada coluna (cada variável de entrada)



fig = plt.figure(figsize=(15, 20))



pd.Series(rf.feature_importances_, index = feats).sort_values().plot.barh()



plt.title('Importância de cada Variável no Modelo Random Forest')
# Matriz de Confusão



skplt.metrics.plot_confusion_matrix(train['Target'], rf.predict(train[feats])) 
# Acurácia: 



(650 + 1202 + 929 + 4664) / 9557
# Criando o arquivo para submissão



#test[['Id','Target']].to_csv('submission.csv', index=False)
# Trabalhando com XGBoost



from xgboost import XGBClassifier
xgb = XGBClassifier(n_estimators = 200, learning_rate = 0.09, random_state = 42, class_weight = 'balanced')
xgb.fit(train[feats], train['Target'])
test['Target'] = xgb.predict(test[feats]).astype(int)
accuracy_score(train['Target'], xgb.predict(train[feats]))
# Avaliando a importancia de cada coluna (cada variável de entrada)



fig = plt.figure(figsize=(15, 20))



pd.Series(xgb.feature_importances_, index = feats).sort_values().plot.barh()



plt.title('Importância de cada Variável no Modelo XGBoost')
# Matriz de Confusão



skplt.metrics.plot_confusion_matrix(train['Target'], xgb.predict(train[feats])) 
# Criando o arquivo para submissão



#test[['Id','Target']].to_csv('submission.csv', index=False)
# Trabalhando com AdaBoost



from sklearn.ensemble import AdaBoostClassifier
abc = AdaBoostClassifier(n_estimators = 200, learning_rate = 1.0, random_state = 42)
abc.fit(train[feats], train['Target'])
accuracy_score(train['Target'], abc.predict(train[feats]))
# Avaliando a importancia de cada coluna (cada variável de entrada)



fig = plt.figure(figsize=(15, 20))



pd.Series(abc.feature_importances_, index = feats).sort_values().plot.barh()



plt.title('Importância de cada Variável no Modelo AdaBoost')
# Matriz de Confusão



skplt.metrics.plot_confusion_matrix(train['Target'], abc.predict(train[feats])) 
# Criando o arquivo para submissão



test[['Id','Target']].to_csv('submission.csv', index=False)