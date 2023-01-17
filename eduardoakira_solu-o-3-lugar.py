import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from pylab import rcParams

# Definir tamanho padrão para os gráficos

rcParams['figure.figsize'] = 17, 4



from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score
# Carregando os datasets de treino e teste

train = pd.read_csv('../input/data-train-competicao-ml-1-titanic/train.csv')

test = pd.read_csv('../input/data-train-competicao-ml-1-titanic/test.csv')

identificador = test['PassengerId']
## Join train and test datasets in order to obtain the same number of features during categorical conversion

train_len = len(train) #guarda o tamanho do train para divisão do dataset depois

dataset =  pd.concat(objs=[train, test], axis=0).reset_index(drop=True)
# Analise dos datasets

len(train)
len(test)
len(dataset)
# Fill empty and NaNs values with NaN

dataset = dataset.fillna(np.nan)



# Check for Null values

dataset.isnull().sum()
# Infos

dataset.info()
dataset.head()
dataset.describe()
# -----Analisando dados numericos-----

# Correlation matrix between numerical values (SibSp Parch Age and Fare values) and Survived 

g = sns.heatmap(dataset[["Survived","SibSp","Parch","Age","Fare"]].corr(), annot=True, fmt = ".2f", cmap = "coolwarm")
# Explore SibSp feature vs Survived

g = sns.catplot(x="SibSp",y="Survived",data=dataset,kind="bar", height= 6 , palette = "muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")
# Explore Parch feature vs Survived

g  = sns.catplot(x="Parch",y="Survived",data=dataset,kind="bar", height= 6, palette = "muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")
# Explore Age vs Survived

g = sns.FacetGrid(dataset, col='Survived')

g = g.map(sns.distplot, "Age")
# Explore Age distribution 

g = sns.kdeplot(dataset["Age"][(dataset["Survived"] == 0) & (dataset["Age"].notnull())], color="Red", shade = True)

g = sns.kdeplot(dataset["Age"][(dataset["Survived"] == 1) & (dataset["Age"].notnull())], ax =g, color="Blue", shade= True)

g.set_xlabel("Age")

g.set_ylabel("Frequency")

g = g.legend(["Not Survived","Survived"])
# -----Analisando dados categoricos-----

# Explorar Sex vs Survived

g = sns.barplot(x="Sex",y="Survived",data=dataset)

g = g.set_ylabel("Survival Probability")
dataset[["Sex","Survived"]].groupby('Sex').mean()
# Explore Pclass vs Survived

g = sns.catplot(x="Pclass",y="Survived",data=dataset,kind="bar", height= 6 , palette = "muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")
# Explore Pclass vs Survived by Sex

g = sns.catplot(x="Pclass", y="Survived", hue="Sex", data=dataset, height= 6, kind="bar", palette="muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")
# Detectar valores nulos de embark

dataset["Embarked"].isnull().sum()
#Fill Embarked nan values of dataset set with 'S' most frequent value

dataset["Embarked"] = dataset["Embarked"].fillna("S")
# Explore Embarked vs Survived 

g = sns.catplot(x="Embarked", y="Survived",  data=dataset, height= 6, kind="bar", palette="muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")
# Explore Pclass vs Embarked 

g = sns.catplot("Pclass", col="Embarked",  data=dataset, height=6, kind="count", palette="muted")

g.despine(left=True)

g = g.set_ylabels("Count")
#----- Preenchendo valores null -----

# Explore Age vs Sex, Parch , Pclass and SibSP

g = sns.catplot(y="Age",x="Sex",data=dataset,kind="box")

g = sns.catplot(y="Age",x="Sex",hue="Pclass", data=dataset,kind="box")

g = sns.catplot(y="Age",x="Parch", data=dataset,kind="box")

g = sns.catplot(y="Age",x="SibSp", data=dataset,kind="box")
# convert Sex into categorical value 0 for male and 1 for female

dataset["Sex"] = dataset["Sex"].map({"male": 0, "female":1})
# correlation map pra saber qual das categorias usar pra determinar age

g = sns.heatmap(dataset[["Age","Sex","SibSp","Parch","Pclass"]].corr(),cmap="BrBG",annot=True)
# Filling missing value of Age 



## Fill Age with the median age of similar rows according to Pclass, Parch and SibSp

# Index of NaN age rows

index_NaN_age = list(dataset["Age"][dataset["Age"].isnull()].index)



for i in index_NaN_age :

    age_med = dataset["Age"].median()

    age_pred = dataset["Age"][((dataset['SibSp'] == dataset.iloc[i]["SibSp"]) & (dataset['Parch'] == dataset.iloc[i]["Parch"]) & (dataset['Pclass'] == dataset.iloc[i]["Pclass"]))].median()

    if not np.isnan(age_pred) :

        dataset['Age'].iloc[i] = age_pred

    else :

        dataset['Age'].iloc[i] = age_med
# ----- Feature engineering -----

# Gives the length of the name

dataset['Name_length'] = dataset['Name'].apply(len)
# Tamanho do nome vs survived 

g = sns.catplot(x="Survived", y = "Name_length",data = dataset, kind="box")
dataset["Name"].head()
# Get Title from Name

dataset_title = [i.split(",")[1].split(".")[0].strip() for i in dataset["Name"]]

dataset["Title"] = pd.Series(dataset_title)

dataset["Title"].head()
g = sns.countplot(x="Title",data=dataset)

g = plt.setp(g.get_xticklabels(), rotation=45) 
# Convert to categorical values Title 

dataset["Title"] = dataset["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

dataset["Title"] = dataset["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})

dataset["Title"] = dataset["Title"].astype(int)
g = sns.countplot(dataset["Title"])

g = g.set_xticklabels(["Master","Miss/Ms/Mme/Mlle/Mrs","Mr","Rare"])
g = sns.catplot(x="Title",y="Survived",data=dataset,kind="bar")

g = g.set_xticklabels(["Master","Miss-Mrs","Mr","Rare"])

g = g.set_ylabels("survival probability")
# Drop Name variable

dataset.drop(labels = ["Name"], axis = 1, inplace = True)
# Create a family size descriptor from SibSp and Parch

dataset["Fsize"] = dataset["SibSp"] + dataset["Parch"] + 1
g = sns.catplot(x="Fsize",y="Survived",data = dataset, kind = "point")

g = g.set_ylabels("Survival Probability")
# Create new feature of family size

dataset['Single'] = dataset['Fsize'].map(lambda s: 1 if s == 1 else 0)

dataset['SmallF'] = dataset['Fsize'].map(lambda s: 1 if  s == 2  else 0)

dataset['MedF'] = dataset['Fsize'].map(lambda s: 1 if 3 <= s <= 4 else 0)

dataset['LargeF'] = dataset['Fsize'].map(lambda s: 1 if s >= 5 else 0)
g = sns.catplot(x="Single",y="Survived",data=dataset,kind="bar")

g = g.set_ylabels("Survival Probability")

g = sns.catplot(x="SmallF",y="Survived",data=dataset,kind="bar")

g = g.set_ylabels("Survival Probability")

g = sns.catplot(x="MedF",y="Survived",data=dataset,kind="bar")

g = g.set_ylabels("Survival Probability")

g = sns.catplot(x="LargeF",y="Survived",data=dataset,kind="bar")

g = g.set_ylabels("Survival Probability")
# convert to indicator values Title and Embarked 

dataset = pd.get_dummies(dataset, columns = ["Title"])

dataset = pd.get_dummies(dataset, columns = ["Embarked"], prefix="Em")
dataset.head()
dataset["Cabin"].head()
dataset["Cabin"].describe()
dataset["Cabin"].isnull().sum()
# Replace the Cabin number by the type of cabin 'X' if not

dataset["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in dataset['Cabin'] ])
g = sns.countplot(dataset["Cabin"],order=['A','B','C','D','E','F','G','T','X'])
g = sns.catplot(y="Survived",x="Cabin",data=dataset,kind="bar",order=['A','B','C','D','E','F','G','T','X'])

g = g.set_ylabels("Survival Probability")
dataset = pd.get_dummies(dataset, columns = ["Cabin"],prefix="Cabin")
dataset["Ticket"]
## Treat Ticket by extracting the ticket prefix. When there is no prefix it returns X. 



Ticket = []

for i in list(dataset.Ticket):

    if not i.isdigit() :

        Ticket.append(i.replace(".","").replace("/","").strip().split(' ')[0]) #Take prefix

    else:

        Ticket.append("X")

        

dataset["Ticket"] = Ticket

dataset["Ticket"].head()
dataset = pd.get_dummies(dataset, columns = ["Ticket"], prefix="T")
# Create categorical values for Pclass

dataset["Pclass"] = dataset["Pclass"].astype("category")

dataset = pd.get_dummies(dataset, columns = ["Pclass"],prefix="Pc")
#----- Treinamento do modelo -----

treino = dataset[:train_len]

teste = dataset[train_len:]

teste.drop(labels=["Survived"],axis = 1,inplace=True)



skf = StratifiedKFold(n_splits=5, random_state=420, shuffle=True)

X = treino.drop('Survived', axis=1).values

y = treino['Survived'].values

media_treino =[]

media_teste = []

media_validacao =[]
contador = 1

for indice_treino, indice_validacao in skf.split(X,y):    

    X_treino = X[indice_treino]

    y_treino = y[indice_treino]    

    X_validacao = X[indice_validacao]

    y_validacao = y[indice_validacao]

    modelo = GradientBoostingClassifier(random_state=420)

    modelo.fit(X_treino, y_treino)

    

    y_pred = modelo.predict_proba(X_treino)

    y_pred = y_pred[:,1]

    score_treino = roc_auc_score(y_treino, y_pred)  

    print("Treino número {} : {}".format(contador, score_treino))



    y_validacao_pred = modelo.predict_proba(X_validacao)

    y_validacao_pred = y_validacao_pred[:,1]

    score_validacao = roc_auc_score(y_validacao, y_validacao_pred)

    print("Validacao número {} : {} \n".format(contador, score_validacao))



    contador += 1



    X_teste = teste

    y_pred_teste = modelo.predict_proba(X_teste)

    y_pred_teste = y_pred_teste[:, 1]  



    media_treino.append(score_treino)  

    media_validacao.append(score_validacao)

    media_teste.append(y_pred_teste)



print("Media de todos treinos {}:".format(np.mean(media_treino)))

print("Media de todas validações {}:".format(np.mean(media_validacao)))
mediafinal_pred = np.mean(media_teste, axis=0)
resultado = pd.concat([identificador, pd.DataFrame(mediafinal_pred, columns=['Survived'])], axis=1)

resultado
# gerando arquivos para submissão na competição

resultado.to_csv('submission.csv', index=False)