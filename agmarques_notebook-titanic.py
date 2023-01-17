# Importando bibliotecas

import pandas as pd

import string

import numpy as np

import random
# Caminho dos arquivos

path_train = '/kaggle/input/titanic/train.csv'

path_test  = '/kaggle/input/titanic/test.csv'
# Criando os dataframes de treino e teste ao mesmo tempo

df_train = pd.read_csv(path_train, sep=',')

df_test  = pd.read_csv(path_test, sep=',')
# Amostra DF de treino

df_train.head(3)
# Amostra DF de teste

df_test.head(3)
# Correlação sem tratamento ou engenharia básica de recurso

df_train_corr = df_train.corr()

df_train_corr['Survived'].sort_values(ascending=False)
# Primeiro tratamento será definir dados categóricos numéricos para o atributo Age

df_train['Sex'] = df_train['Sex'].apply(lambda x: 1 if x == 'female' else 0)

df_test['Sex']  = df_test['Sex'].apply(lambda x: 1 if x == 'female' else 0)
# Visualização de dados nulos/ausentes

df_train.isnull().sum()
# Arredondar o valor da passagem para facilitar algum tipo de tratamento

df_train['Fare'] = round(df_train['Fare'],0)

df_test['Fare'] = round(df_test['Fare'],0)
# Numerar faixa etária de idade

def faixa_etaria(x):

    """Função para definir faixa etárias

       Criança = 0, Adolescente = 1, Adulto = 2 e Idoso = 3

       input: Age/Idade

       output: Categorização numérica

    """

    if x == 0.0:

        return 0    

    elif x < 12.9:

        return 1

    elif x < 19.9:

        return 2

    elif x < 59.9:

        return 3

    elif x >= 60:

        return 4

    else:

        return -1    
# Aplicar função para faixa etária de idade em uma nova coluna

df_train['Idade'] = df_train['Age'].apply(faixa_etaria)

df_test['Idade'] = df_test['Age'].apply(faixa_etaria)
def substrings_in_string(big_string, substrings):

    """Esta função caracteres de um texto com uma lista

       input: texto de uma determinada coluna + lista para comparação

       output: posição na lista encontrada

    """

    for substring in substrings:

        if str.find(str(big_string), substring) != -1:

            return substring

    print(big_string)

    return np.nan
# Lista de títulos pré definida

title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev','Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess','Don', 'Jonkheer']

# Execução da função para preenchimento da nova coluna Title

df_train['Title']=df_train['Name'].map(lambda x: substrings_in_string(x, title_list))

df_test['Title']=df_test['Name'].map(lambda x: substrings_in_string(x, title_list))
def replace_titles(x):

    """Função para substituir títulos

       input: DataFrame completo para leitura de duas colunas (Title e Sex)

       output: Ajuste do título

    """

    title=x['Title']

    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:

        return 'Mr'

    elif title in ['Countess', 'Mme']:

        return 'Mrs'

    elif title in ['Mlle', 'Ms']:

        return 'Miss'

    elif title =='Dr':

        # Nesta parte da função se o title for Dr o preenchimento será de acordo com o sexo para Mr ou Mrs

        if x['Sex']=='Male':

            return 'Mr'

        else:

            return 'Mrs'

    else:

        return title
# Substituindo coluna Title, repare que não foi definido uma coluna específica, foi passado todo DF.

df_train['Title']=df_train.apply(replace_titles, axis=1)

df_test['Title']=df_test.apply(replace_titles, axis=1)
# Aproveitando a função substrings_in_string na coluna Cabin

cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']

df_train['Deck']=df_train['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))

df_test['Deck']=df_test['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))
# Preencher Deck vazios

df_train['Deck'].fillna('X',inplace=True)

df_test['Deck'].fillna('X',inplace=True)
# Identificação mais aproximada da cabine verdadeira

# As cabines mais luxuosas estão entre A ao E

letras1 = 'ABCDE'

letras2 = 'FG'

def deck_pclass(x,p):

    if x == 'X':

        if p == 1:

            return random.choice(letras1)

        elif p == 2:

            return random.choice(letras2)            

        else:

            return 'T'  

    else:

        return x
# Substituir Letras por número de forma ordenada, o factorize do panda realiza esta alteração, mas sem ordem

# Categorização numérica

def factorize_order(x):

    if x == 'A':

        return 1

    elif x == 'B':

        return 2

    elif x == 'C':

        return 3

    elif x == 'D':

        return 4

    elif x == 'E':

        return 5

    elif x == 'F':

        return 6

    elif x == 'G':

        return 7

    elif x == 'T':

        return 8

    else:

        return 9  
# Transformar em Categorização numérica

df_train['Deck']=df_train['Deck'].map(lambda x: factorize_order(x))

df_test['Deck']=df_test['Deck'].map(lambda x: factorize_order(x))
# Criar coluna Family_Size

df_train['Family_Size']=df_train['SibSp']+df_train['Parch']

df_test['Family_Size']=df_test['SibSp']+df_train['Parch']
# Método do pandas factorize() defini um valor inteiro para cada texto/grupo

df_train['Titulo'], cat_temp = df_train['Title'].factorize()

df_test['Titulo'], cat_temp = df_test['Title'].factorize()
# Correlação após alguns tratamentos

df_train_corr2 = df_train.corr()

df_train_corr2['Survived'].sort_values(ascending=False)
# Rever a correlação antes das alterações

df_train_corr['Survived'].sort_values(ascending=False)
# Aplicando Logistic regression

from sklearn import linear_model, model_selection
# Determinando os valores de X de acordo com o resultado da correlação e valor de y

X = df_train[['Sex','Fare','Titulo','Idade', 'Deck','Family_Size']]

y = df_train['Survived']
# LogisticRegression

model = linear_model.LogisticRegression()

model.fit(X,y)

model.classes_, model.coef_, model.intercept_

model.score(X,y)
# LinearRegression

model2 = linear_model.LinearRegression()

model2.fit(X,y)

model2.coef_, model2.intercept_

model2.score(X,y)
# LinearRegression

model2 = linear_model.LinearRegression()

model2.fit(X,y)

model2.coef_, model2.intercept_

model2.score(X,y)
from sklearn.tree import DecisionTreeClassifier
# DecisionTreeClassifier

model3 = DecisionTreeClassifier()

model3.fit(X,y)

model3.score(X,y)
# Criando X de teste com as mesmas alterações realizadas no treino

X_test = df_test[['Sex','Fare','Titulo','Idade', 'Deck','Family_Size']]
# Visualizando se existe algum valor nulo para evitar erro

X_test.isnull().sum()
# Substituir pela média o único valor nulo

X_test['Fare'].fillna(value = np.mean(X_test['Fare']), inplace=True)
X_test.isnull().sum()
# Realizar o teste com o melhor modelo tree

y_pred = model3.predict(X_test)
# Gerar arquivo para envio

result = pd.Series(y_pred, index=df_test['PassengerId'], name='Survived')

#result.to_csv('/kaggle/input/titanic/titanic.csv',sep=',', header=True)

result.head(10)