# Importando as bibliotecas necessárias para a criação e organização do df

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
train = pd.read_csv("../input/ta192/train.csv")

train = train.set_index("ID") # ID não será considerado na análise, usado somente para identificação

train.head()
def variaveis(df, coluna):

    '''

    Função que cria dicionários com as variáveis existentes em cada coluna,

    facilitando a análise de quais variáveis devem ser alteradas

    '''

    dicionario = {}

    for i in df[coluna]:

        if i in dicionario:

            dicionario[i] += 1

        else:

            dicionario[i] = 1

    return dicionario
sex_variaveis = variaveis(train, 'SEX')

print(sex_variaveis)
# transformando todas os sinõnimos em um só

train['SEX'] = train['SEX'].str.replace('male','m')

train['SEX'] = train['SEX'].str.replace('female','f')

train['SEX'] = train['SEX'].str.replace('fem','f')

train['SEX'] = train['SEX'].fillna(0.5) #substituindo NaN por um valor intermediário entre 0 e 1

sex_variaveis = variaveis(train, 'SEX')

print(sex_variaveis)
train = train.replace(['m', 'f'], [0 ,1]) #masc = 0 e fem = 1, dados numéricos são necessários para o algoritmo

sex_variaveis = variaveis(train, 'SEX') 

print(sex_variaveis)
education_variaveis = variaveis(train, 'EDUCATION')

print(education_variaveis)
train["EDUCATION"].fillna('others', inplace = True)

train = train.replace(['high school', 'university', 'graduate school', 'others'], [0, 1, 2, 3])

education_variaveis = variaveis(train, 'EDUCATION')

print(education_variaveis)
marriage_variaveis = variaveis(train, 'MARRIAGE')

print(marriage_variaveis)
train['MARRIAGE'] = train['MARRIAGE'].str.replace('MARRIED', 'married')

train['MARRIAGE'] = train['MARRIAGE'].str.replace('SINGLE', 'single')

train["MARRIAGE"] = train['MARRIAGE'].str.replace('OTHERS', 'others')

train['MARRIAGE'].fillna('others', inplace = True)

train = train.replace(['married', 'single', 'others'], [0, 1, 2])

marriage_variaveis = variaveis(train, 'MARRIAGE')

print(marriage_variaveis)
train["AGE"].describe()
train['AGE'].loc[train["AGE"] > 100]
train['AGE'].loc[train["AGE"] < 0]
train["AGE"].fillna(train["AGE"].median(), inplace = True)

train["AGE"].replace(to_replace = -10000000, value = train["AGE"].median(), inplace = True)

train["AGE"].replace(to_replace = 180, value = train["AGE"].median(), inplace = True)
train["AGE"].describe() #Agora sim uma média normal
colunas = list(train.columns.values)

for col in colunas[5:23]:

    train[col].fillna(0, inplace  = True)

train["LIMIT_BAL"].fillna(0, inplace = True)
train.head(10)
train.isnull().any() #Verificando se algum NaN foi esquecido em alguma coluna
#Importação das bibliotecas necessárias do sklearn

from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import RandomizedSearchCV
# Separação do dataframe

X_train = train.drop("default.payment.next.month", axis=1)

y_train = train["default.payment.next.month"]
clf = RandomForestClassifier(n_estimators = 100, random_state=0)


# Number of trees

n_estimators = [int(x) for x in np.linspace(start = 400, stop = 2000, num = 10)]



max_features = ['auto', 'sqrt']



# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]



max_depth.append(None)



# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10]



# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree

bootstrap = [True, False]





random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}
grid = RandomizedSearchCV(clf, param_distributions = random_grid, n_iter = 20, cv = 3, verbose=2, random_state=42, n_jobs = -1 )
#grid.fit(X_train, y_train) 

#Não vou executar esta parte do código por demorar muito tempo para rodar, mas os parâmetros já estão no modelo
#grid.best_params_
#Modelo com os parâmetros encontrados

clf = RandomForestClassifier(n_estimators= 400,

 min_samples_split= 5,

 min_samples_leaf= 4,

 max_features= 'auto',

 max_depth= 10,

 bootstrap=True)
tree = clf.fit(X_train, y_train)
test = pd.read_csv("../input/ta192/test.csv")

test = test.set_index("ID")

test.head()
#LIMIT_BAL

test["LIMIT_BAL"].fillna(0, inplace = True)



#SEX

test['SEX'].fillna(0.5, inplace = True)

test = test.replace(['m', 'male', 'female', 'fem', 'f'], [0 ,0, 1, 1, 1])



#EDUCATION

test["EDUCATION"].fillna('others', inplace = True)

test = test.replace(['high school', 'university', 'graduate school', 'others'], [0, 1, 2, 3])



#MARRIAGE

test['MARRIAGE'].fillna('others', inplace = True)

test = test.replace(['married', 'single', 'others', 'MARRIED', 'SINGLE', 'OTHERS'], [0, 1, 2, 0, 1, 2])



#Outras colunas:

colunas = list(test.columns.values)

for col in colunas[4:23]:

     test[col].fillna(0, inplace  = True)
test.head()
train.isnull().any()
y_pred = tree.predict(test)
df_com_id = pd.read_csv("../input/ta192/test.csv") #dataframe cujo papel é fornecer o "ID" para o CSV



output = pd.DataFrame({'ID': df_com_id["ID"], 'default.payment.next.month': y_pred})



output.to_csv("tentativa4.csv", index=False)