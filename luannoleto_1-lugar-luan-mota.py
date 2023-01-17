import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from pylab import rcParams

import pandas as pd

from collections import Counter

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc

from sklearn.metrics import roc_auc_score

from sklearn import preprocessing

from sklearn.model_selection import StratifiedKFold

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression
# Exibir gráficos dentro do Jupyter Notebook

%matplotlib inline



# Definir tamanho padrão para os gráficos

rcParams['figure.figsize'] = 17, 4
train_original = pd.read_csv('../input/data-train-competicao-ml-1-titanic/train.csv')

test_original = pd.read_csv('../input/data-train-competicao-ml-1-titanic/test.csv')
# Detecção de outliers baseado no método de Tukey



def detect_outliers(df,n,features):

    """

    Recebe um dataframe 'df' de features e retorna uma lista de índices

    correspondentes que possuem mais de 'n' outliers de acordo com o método

    de Tukey.

    

    IMPORTANTE:descobri que deveria ter usado a função 'nanpercentile' para 

    desconsiderar os valores NAN, assim teria 3 outliers e não 2 como é mostrado.

    """

    outlier_indices = []

    

    # interagir em cada coluna (feature) passado

    for col in features:

        

        # 1º quartil (25%)

        Q1 = np.percentile(df[col], 25) #retorna o valor absoluto do número que é maior que 25% e menor que os 75% restante

        print('Q1:{}'.format(Q1))

        # 3º quartil (75%)

        Q3 = np.percentile(df[col],75)  #retorna o valor absoluto do número que é maior que 75% e menor que os 25% restante

        print('Q3:{}'.format(Q3))

        

        # calculando o interquartil (IQR) = grau de espalhamento dos dados

        IQR = Q3 - Q1

        print('IQR:{}'.format(IQR))

        # aplicando um step no IQR para representar o máximo de diferença que aceitaremos

        outlier_step = 1.5 * IQR

        

        # Detecta um outlier que está acima do limiar superior Q3 + (1.5 * IQR)

        outlier_list_col = df[(df[col] > Q3 + outlier_step )].index

        

        # guarda o índice do outlier no final da lista

        outlier_indices.extend(outlier_list_col)

        

    #Conta quantas vezes aquele índice foi um outlier

    outlier_indices = Counter(outlier_indices)

    

    #percorre o outlier_indices (índice=k,quantas vezes ele aparece=v) e se o número de vezes for maior que o n então considera um ourlier e guarda esse índice

    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )

    

    return multiple_outliers   



# passao dataset somente de treino, n=2 e as colunas (features)

Outliers_to_drop = detect_outliers(train_original,2,["Age","SibSp","Parch","Fare"])



#localiza esses índices no dataset para printar na tela

train_original.loc[Outliers_to_drop]
train_original = train_original.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)   #retiro os outrliers

train_len = len(train_original)                                                           #guardo o tamanho o treino para separar o dataset no final da mesma forma

dataset =  pd.concat(objs=[train_original, test_original], axis=0).reset_index(drop=True) #junto o treino com o teste para ter somente um dataset e facilitar na criação de novas features

dataset = dataset.fillna(np.nan)                                                          #preencho os valores de Na e NaN com np.nan = NaN
print(dataset.head())
print(dataset.dtypes)
# Somente atributos numéricos são considerados

print(dataset.describe())
# Quantidade absoluta

print(dataset.isnull().sum())
dataset["Fare"].hist()
#faço a normalização dos dados(0 e 1) já que existe valores muito diferentes e pode interferir no treinamento

dataset["Fare"] = (dataset["Fare"]-min(dataset["Fare"]))/(max(dataset["Fare"])-min(dataset["Fare"]))
dataset["Fare"].hist()
dataset["Embarked"].describe()
#preencho a featura que possui 2 valores NaN com o que mais aparece no dataset

dataset["Embarked"] = dataset["Embarked"].fillna("S")
#substituo os valores de homem = 0 e mulher = 1

dataset["Sex"] = dataset["Sex"].map({"male": 0, "female":1})
# Somente atributos numéricos são considerados

plt.suptitle("Gráfico de Calor das Correlações entre os Atributos Numéricos")

sns.heatmap(dataset.drop(labels =["PassengerId","Survived"], axis = 1).corr(), annot=True, cmap='Blues')
g = sns.catplot(y="Age",x="Sex",data=dataset,kind="box")

g = sns.catplot(y="Age",x="Sex",hue="Pclass", data=dataset,kind="box")

g = sns.catplot(y="Age",x="Parch", data=dataset,kind="box")

g = sns.catplot(y="Age",x="SibSp", data=dataset,kind="box")
index_NaN_age = list(dataset["Age"][dataset["Age"].isnull()].index) #pega todos os índices que idade é igual a NaN



for i in index_NaN_age :

    #faz a média das idades do dataset todo

    age_med = dataset["Age"].median()

    #faz a média da idade em que os features SibSp, Parch e Pclass tem os mesmos valores que o índice do valor NaN no feature AGE

    age_pred = dataset["Age"][((dataset['SibSp'] == dataset.iloc[i]["SibSp"]) & (dataset['Parch'] == dataset.iloc[i]["Parch"]) & (dataset['Pclass'] == dataset.iloc[i]["Pclass"]))].median()

    #verifica se age_pred é NaN, ou seja, caso tenha encontrado outros índices e feito a média, atribui esse valor no dataset. Caso contrário atribui a média do dataset original

    if not np.isnan(age_pred) :

        dataset['Age'].iloc[i] = age_pred

    else :

        dataset['Age'].iloc[i] = age_med
dataset_title = [i.split(",")[1].split(".")[0].strip() for i in dataset["Name"]] #retira somente os títulos dos nomes para criar uma feature

dataset["Title"] = pd.Series(dataset_title)                                      #insere essa nova featura no dataset

dataset["Title"].head()
g = sns.countplot(x="Title",data=dataset)

g = plt.setp(g.get_xticklabels(), rotation=45)
Counter(dataset["Title"])
#os títulos menos comuns (usados na sociedade e não no dataset) vou chamar de Rare

dataset["Title"] = dataset["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

#os títulos mais comuns vamos separar por homem e mulher, sendo de homem separados o master de mr pois tem significados diferentes e podem influenciar

dataset["Title"] = dataset["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})

dataset["Title"] = dataset["Title"].astype(int)

Counter(dataset["Title"])
g = sns.countplot(x="Title",data=dataset)

g = plt.setp(g.get_xticklabels(), rotation=45)
dataset.drop(labels = ["Name"], axis = 1, inplace = True)
#criamos uma feature chamada de Fsize que representa o tamanho da família no navio, ou seja num de (irmãos + esposas) + (pais + filhos) + a pessoa

dataset["Fsize"] = dataset["SibSp"] + dataset["Parch"] + 1
#quebrar o feature Fsize para obtermos outros features baseados no tamanho delas

dataset['Single'] = dataset['Fsize'].map(lambda s: 1 if s == 1 else 0)

dataset['SmallF'] = dataset['Fsize'].map(lambda s: 1 if  s == 2  else 0)

dataset['MedF'] = dataset['Fsize'].map(lambda s: 1 if 3 <= s <= 4 else 0)

dataset['LargeF'] = dataset['Fsize'].map(lambda s: 1 if s >= 5 else 0)
#gera o dummies dos títulos e se embarcou para transformar em variáveis com valores numéricas

dataset = pd.get_dummies(dataset, columns = ["Title"])

dataset = pd.get_dummies(dataset, columns = ["Embarked"], prefix="Em")
dataset.head()
print(dataset["Cabin"].describe())
#cria feature baseado se a pessoa tem cabine ou não. verifica que se for do tipo NaN (que é considerado um float) recebe 0 (não tem cabine) e se for diferente de NaN tem cabine e recebe 1

dataset['Has_Cabin'] = dataset["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
dataset['Has_Cabin'].head()
#substitui o nome da cabine pela primeira letra e se for nulo coloca X no lugar

dataset["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in dataset['Cabin'] ])
g = sns.catplot(y="Survived",x="Cabin",data=dataset,kind="bar",order=['A','B','C','D','E','F','G','T','X'])

g = g.set_ylabels("Survival Probability")
dataset = pd.get_dummies(dataset, columns = ["Cabin"],prefix="Cabin")
#percorre o feature Ticket, se não for dígito retira . / e pega o prefixo. Caso seja um dígito substitui por X e aloca em Ticket

Ticket = []

for i in list(dataset.Ticket):

    if not i.isdigit() :

        Ticket.append(i.replace(".","").replace("/","").strip().split(' ')[0])

    else:

        Ticket.append("X")

        

dataset["Ticket"] = Ticket #substitui pela lista Ticket criada

dataset["Ticket"].head()
dataset = pd.get_dummies(dataset, columns = ["Ticket"], prefix="T")
dataset.head()
dataset["Pclass"] = dataset["Pclass"].astype("category") #transforma os números 0,1,2 e em catergorias para gerar o dummies

dataset = pd.get_dummies(dataset, columns = ["Pclass"],prefix="Pc")
dataset.head()
print(dataset.isnull().sum())
#separando o dataset em treino e teste

train_original = dataset[:train_len]

test_original = dataset[train_len:]

test_original.drop(labels=["Survived"],axis = 1,inplace=True)
skf = StratifiedKFold(n_splits=5, random_state=2020, shuffle=True)

X = train_original.drop('Survived', axis=1).values

y = train_original['Survived'].values

score_val =[]

score_test = []



def treinamento(classificador):

    for train_index, val_index in skf.split(X,y): #treino recebe 4 partes e o teste em 1 parte (que sempre será distinta) e intera no for 5 vezes 

        

        X_train,X_val = X[train_index],X[val_index]    #recebe a parte que treinará o modelo e a parte que testará o modelo treinado (validação)

        y_train,y_val = y[train_index],y[val_index]    #recebe a as respostas correspondetes ao treino e a as respostas correspondetes ao teste (validação)



        model = classificador                          #recebe o algoritmo machine learn 

        model.fit(X_train, y_train)                    #treina o modelo

        

        y_test = model.predict_proba(X_val)            #recebe a predição do teste para comparar com o y_val = respostas da parte de teste

        y_test = y_test[:,1]                           #pega somente a coluna que contem as % de ter sobrevivido ou não

        RAC = roc_auc_score(y_val, y_test)             #calcula a área sobre a curva ROC

        score_val.append(RAC)                          #guarda o valor obtido para fazer a média no final

        print("Teste = {}".format(RAC))

        

        #uso da parte de teste do dataset original para gerar a predição que será enviada no site

        y_pred = model.predict_proba(test_original)    #roda o modelo em cima do dataset teste original

        y_pred = y_pred[:, 1]                          #pega toda as linhas da coluna de 'Survived'

        score_test.append(y_pred)                      #guarda o valor obtido da predição para submeter na competição



    print("Média do teste = {}".format(np.mean(score_val)))

    

    return score_test
"""

É feito a média do resultuado (merge) para que eu consiga percorrer 100% no dataset,

já que eu dividio o meu dataset em 5 partes diferentes, treinando com cada uma delas.

Além disso, o modelo variará menos do que enviar somente o melhor modelo dos 5 gerados.

"""

final_pred_RD = np.mean(treinamento(RandomForestClassifier(random_state=2020)), axis=0) 
final_pred_LR = np.mean(treinamento(LogisticRegression(random_state=2020)), axis=0)
pegarID = pd.read_csv('../input/data-train-competicao-ml-1-titanic/test.csv')

identificador = pegarID['PassengerId']

resultado_RD = pd.concat([identificador, pd.DataFrame(final_pred_RD, columns=['Survived'])], axis=1)

resultado_RD.head()
resultado_LR = pd.concat([identificador, pd.DataFrame(final_pred_LR, columns=['Survived'])], axis=1)

resultado_LR.head()
resultado_ensemble = resultado_LR.copy()

resultado_ensemble['Survived'] = (resultado_RD['Survived'] +resultado_LR['Survived'])/2

resultado_ensemble.head()
def posprocess(x):

    if x<=0.15:

        return 0

    elif x>=0.9:

        return 1

    else:

        return x
resultado_ensemble['Survived'] = resultado_ensemble['Survived'].apply(posprocess)

resultado_ensemble.head()
resultado_ensemble.to_csv('submission_esem_pos.csv', index=False)