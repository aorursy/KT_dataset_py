# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Arquivo para treinamento do modelo

dt_train = pd.read_csv("/kaggle/input/titanic/train.csv")

dt_train.head()
# Arquivo para criação do output final

dt_test = pd.read_csv("/kaggle/input/titanic/test.csv")

dt_test.head()
# Sumarização - Data Quality Report

# dt_train.describe()

dt_train.describe(include='all')
#  Total de registros

print('Total de registros:', len(dt_train))



# verificando o tipo de dado

#print(type(train_data.Survived))



# Total de sobreviventes

totais_sob_dt_train = dt_train[dt_train.Survived == 1]  

print('Sobreviventes:', len(totais_sob_dt_train))
# Totais por sexo

# Homens

totais_mh_dt_train = dt_train[(dt_train.Sex == 'male') & (dt_train.Survived == 0)]  

print('Total de homens mortos:', len(totais_mh_dt_train))

# Só pra ter certeza que a query esta funcionando :3

totais_sh_dt_train = dt_train[(dt_train.Sex == 'male') & (dt_train.Survived == 1)]  

print('Total de homens sobreviventes:', len(totais_sh_dt_train))



# Mulheres

totais_sm_dt_train = dt_train[(dt_train.Sex == 'female') & (dt_train.Survived == 1)]  

print('Total de mulheres sobreviventes:', len(totais_sm_dt_train))

# Só pra ter certeza que a query esta funcionando :3

totais_mm_dt_train = dt_train[(dt_train.Sex == 'female') & (dt_train.Survived == 0)]  

print('Total de mulheres mortas:', len(totais_mm_dt_train))



# Sem dado de sobrevivencia

totais_na_data = dt_train[((dt_train.Survived != 1) & (dt_train.Survived != 0)) ]

print('Total sem valor valido:', len(totais_na_data))
# Totais por classe

totais_1st_train_data = dt_train[(dt_train.Pclass == 1) & (dt_train.Survived == 1)]  

print('Total de sobreviventes 1st:', len(totais_1st_train_data))

#

totais_2nd_train_data = dt_train[(dt_train.Pclass == 2) & (dt_train.Survived == 1)]  

print('Total de sobreviventes 2nd:', len(totais_2nd_train_data))

#

totais_3rd_train_data = dt_train[(dt_train.Pclass == 3) & (dt_train.Survived == 1)]  

print('Total de sobreviventes 3rd:', len(totais_3rd_train_data))
# Teste

nome = 'Cumings, Mrs. John Bradley '

print(nome.lower().find(',') , nome.lower().find('.'))

print(nome[7+2:12])
# cria lista com os pronomes

Pron = []



# popula listar com os pronomes encontrados

lst_trat = []

for index, row in dt_train.iterrows():

    idComma = (row['Name'].find(',') +2)

    idDot = row['Name'].find('.')

    pronome = row['Name'][idComma:idDot]

    Pron.append(pronome)

    if pronome not in lst_trat:

        lst_trat.append(pronome)

    

lst_trat
# Adiciona lista a uma nova coluna

dt_train['Pron'] = Pron



dt_train.head()
# cria lista com os pronomes

Pron = []



# popula listar com os pronomes encontrados

lst_trat = []

for index, row in dt_test.iterrows():

    idComma = (row['Name'].find(',') +2)

    idDot = row['Name'].find('.')

    pronome = row['Name'][idComma:idDot]

    Pron.append(pronome)

    if pronome not in lst_trat:

        lst_trat.append(pronome)

        

dt_test['Pron'] = Pron

dt_test.head()
# Itens com de idade preenchida

dt_itens_com_idade = dt_train[dt_train.Age.notnull()] 

dt_itens_com_idade.head()
# deleta colunas que não serão utilizadas, para a agrupar

group_itens = dt_itens_com_idade.drop(['PassengerId','Survived', 'Pclass', 'Name', 'Sex', 'SibSp', 'Parch','Ticket','Fare','Cabin','Embarked'], axis=1)



# Media de idade

group_itens.groupby('Pron')['Age'].mean()
# verifica se existe idade Zerada

dt_itens_idade_zerado = dt_train[dt_train.Age == 0] 

dt_itens_idade_zerado.head()
# Como não existe, zera colunas nulas/NaN

dt_train['Age'] = dt_train['Age'].fillna(value=0)

dt_test['Age'] = dt_test['Age'].fillna(value=0)
# Verifica o resultado

dt_itens_idade_zerado = dt_train[dt_train.Age == 0] 

dt_itens_idade_zerado.head()
# Verifica o resultado

dt_itens_idade_zerado_teste = dt_test[dt_test.Age == 0] 

dt_itens_idade_zerado_teste.head()
# teste

#from itertools import islice



#for idx, row in islice(dt_train.iterrows(), 20):

#    if row['Age']==0:

#        print(row['Age'],row['Name'])
# cria uma nova lista

faixa_etaria = []



#1 : criança <= 12

#2 : adolescente/jovem <= 25 (segundo o statuto é ate 18 u.u)

#3 : adulto <= 40

#4 : adulto <= 59

#5 : idoso > 59



# Popula nova lista com faixa etaria

for index, row in dt_train.iterrows():

    # para os itens sem idade, popula faixa etaria conforme a media encontrada no agrupamento dos pronomes

    if row['Age'] == 0:

        if row['Pron'] == 'Master':

            faixa_etaria.append(1)

        elif row['Pron'] == 'Miss' or row['Pron'] == 'Mme' or row['Pron'] == 'Mlle':

            faixa_etaria.append(2)

        elif row['Pron'] == 'Don' or row['Pron'] == 'Jonkheer' or row['Pron'] == 'Mr' or row['Pron'] == 'Mrs' :

            faixa_etaria.append(3)        

        elif row['Pron'] == 'Ms' or row['Pron'] == 'Rev' or row['Pron'] == 'the Countess':

            faixa_etaria.append(3)            

        elif row['Pron'] == 'Col' or row['Pron'] == 'Lady' or row['Pron'] == 'Major' or row['Pron'] == 'Dr' or row['Pron'] == 'Sir':

            faixa_etaria.append(4)

        elif row['Pron'] == 'Capt':

            faixa_etaria.append(5)

    elif row['Age'] > 0:

        if row['Age'] <= 12:

            faixa_etaria.append(1)

        elif row['Age'] <= 25:

            faixa_etaria.append(2)

        elif row['Age'] <= 40:

            faixa_etaria.append(3)

        elif row['Age'] <= 59:

            faixa_etaria.append(4)

        elif row['Age'] > 59:

            faixa_etaria.append(5)

            

#len(faixa_etaria)
# Adiciona lista a uma nova coluna com ids de faixa etaria

dt_train['Faixa'] = faixa_etaria



dt_train.head()
# Aplica para arquivo teste

# cria uma nova lista

faixa_etaria = []



#1 : criança <= 12

#2 : adolescente/jovem <= 25 (segundo o statuto é ate 18 u.u)

#3 : adulto <= 40

#4 : adulto <= 59

#5 : idoso > 59



# Popula nova lista com faixa etaria

for index, row in dt_test.iterrows():

    # para os itens sem idade, popula faixa etaria conforme a media encontrada no agrupamento dos pronomes

    if row['Age'] == 0:

        if row['Pron'] == 'Master':

            faixa_etaria.append(1)

        elif row['Pron'] == 'Miss' or row['Pron'] == 'Mme' or row['Pron'] == 'Mlle':

            faixa_etaria.append(2)

        elif row['Pron'] == 'Don' or row['Pron'] == 'Jonkheer' or row['Pron'] == 'Mr' or row['Pron'] == 'Mrs' :

            faixa_etaria.append(3)        

        elif row['Pron'] == 'Ms' or row['Pron'] == 'Rev' or row['Pron'] == 'the Countess':

            faixa_etaria.append(3)            

        elif row['Pron'] == 'Col' or row['Pron'] == 'Lady' or row['Pron'] == 'Major' or row['Pron'] == 'Dr' or row['Pron'] == 'Sir':

            faixa_etaria.append(4)

        elif row['Pron'] == 'Capt':

            faixa_etaria.append(5)

    elif row['Age'] > 0:

        if row['Age'] <= 12:

            faixa_etaria.append(1)

        elif row['Age'] <= 25:

            faixa_etaria.append(2)

        elif row['Age'] <= 40:

            faixa_etaria.append(3)

        elif row['Age'] <= 59:

            faixa_etaria.append(4)

        elif row['Age'] > 59:

            faixa_etaria.append(5)

            

#len(dt_test)

dt_test['Faixa'] = faixa_etaria

dt_test.head()
dt_train.describe(include='all')
dt_test.describe(include='all')
# Classificar sexo

dt_train['Sex'] = dt_train['Sex'].map({'male': 0, 'female': 1})

dt_test['Sex'] = dt_test['Sex'].map({'male': 0, 'female': 1})

dt_test.head()
# deleta colunas que não serão utilizadas

dt_train.drop(['Parch', 'Ticket', 'Cabin', 'Embarked','Name','Age', 'Pron', 'Fare'], axis=1, inplace=True)

dt_test.drop(['Parch', 'Ticket', 'Cabin', 'Embarked','Name','Age', 'Pron', 'Fare'], axis=1, inplace=True)
dt_train.head()
dt_test.head()
# Remove coluna Survived

dt_train_final = dt_train.drop("Survived", axis=1) 

# adiciona em nova variavel para ser utilizada como target

target_train = dt_train["Survived"] 
# Regressão logistica

from sklearn.linear_model import LogisticRegression



lr = LogisticRegression(random_state = 0)



regressao = lr.fit(dt_train_final, target_train)

predict_regressao = regressao.predict(dt_test)





# verifica a precisão

score_regressao = regressao.score(dt_train_final, target_train)*100

print("Precisão: ", score_regressao)



# Dataset retorno

df_predict_regressao = pd.DataFrame()

df_predict_regressao["PassengerId"] = dt_test['PassengerId']

df_predict_regressao["Survived"] = predict_regressao
# Arvore de decisão

from sklearn.tree import DecisionTreeClassifier



arvore = DecisionTreeClassifier(max_depth=5, random_state=0)

arvore.fit(dt_train_final, target_train)



# verifica a precisão

score_arvore = arvore.score(dt_train_final, target_train)*100

print("Precisão: ", score_arvore)



# dataset retorno

submissao_arvore = pd.DataFrame()

submissao_arvore["PassengerId"] = dt_test['PassengerId']

submissao_arvore["Survived"] = arvore.predict(dt_test)
# Naive Bayes

from sklearn.naive_bayes import GaussianNB



gnb = GaussianNB()

bayes = gnb.fit(dt_train_final, target_train)

predict_bayes = bayes.predict(dt_test)



# verifica a precisão

score_bayes = bayes.score(dt_train_final, target_train)*100

print("Precisão: ", score_bayes)



# dataset retorno

df_predict_bayes = pd.DataFrame()

df_predict_bayes["PassengerId"] = dt_test['PassengerId']

df_predict_bayes["Survived"] = predict_bayes
# SVM

from sklearn import svm



mvs = svm.SVC(kernel = 'linear')

vetor = mvs.fit(dt_train_final, target_train)

predict_vetor = vetor.predict(dt_test)



# verifica a precisão

score_vetor = vetor.score(dt_train_final, target_train)*100

print("SVM: ",score_vetor)



# dataset retorno

df_predict_svm = pd.DataFrame()

df_predict_svm['PassengerId'] = dt_test['PassengerId']

df_predict_svm['Survived'] = vetor.predict(dt_test)
# Gerar arquivo csv de saida usando o dataframe de melhor score

submissao_arvore.to_csv('gender_submission.csv', index=False)