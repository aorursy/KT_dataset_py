# Importing libraries
# Importando bibliotecas

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
# Reading data files
# Lendo arquivos de dados

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
full_data = [train, test]
train.info()
train.head(3)
# Verifying Null
# Verificando Null

print('Train columns with null values:\n', train.isnull().sum())
print("-"*10)

print('Test columns with null values:\n', test.isnull().sum())
print("-"*10)
print(train[["Pclass", "Survived"]].groupby(["Pclass"]).mean())
print(train[["Sex", "Survived"]].groupby(["Sex"]).mean())
# Transforming "Sex" (Male = 0, Female = 1)
# Transformando "Sex" 

for a in full_data:
    a["Sex"] = a["Sex"].map({"male" : 0, "female" : 1})
# Missing values receive 0. Non missing values receive 1.
# Valores missing recebem 0. Valores não missing recebem 1.

for a in full_data:
    a["Cabin"] = a["Cabin"].fillna(0)
    
for a in full_data:
    a.loc[a["Cabin"] != 0, "Cabin"] = 1
print(train[["Cabin", "Survived"]].groupby(["Cabin"]).mean())
# Extacting name title
# Extraindo título do nome

for a in full_data:
    a['Title'] = a.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
sns.pointplot(x = "Age", y = "Title", data = train)
sns.pointplot(x = "Age", y = "Title", data = test)
# Grupping titles
# Agrupando titulos

for a in full_data:
    a["CatTitle"] = a["Title"]
    a['CatTitle'] = a['CatTitle'].replace(['Lady', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
    a['CatTitle'] = a['CatTitle'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
    a['CatTitle'] = a['CatTitle'].replace('Mlle', 'Miss')
    a['CatTitle'] = a['CatTitle'].replace('Ms', 'Miss')
    a['CatTitle'] = a['CatTitle'].replace('Mme', 'Mrs')

train[['CatTitle', 'Survived']].groupby(['CatTitle']).mean()
for a in full_data:
    a["CatTitle"] = a["CatTitle"].map({"Master" : 0,
                                    "Miss" : 1,
                                    "Mr" : 2,
                                    "Mrs" : 3,
                                    "Rare" : 4,
                                    "Royal" : 5})
# Filling Null with average age of each title
# Preenchendo Null com idade média de cada título

mr_age = train.loc[train["Title"] == "Mr"].mean()["Age"]
mrs_age = train.loc[train["Title"] == "Mrs"].mean()["Age"]
miss_age = train.loc[train["Title"] == "Miss"].mean()["Age"]
master_age = train.loc[train["Title"] == "Master"].mean()["Age"]
ms_age = train.loc[train["Title"] == "Ms"].mean()["Age"]
for a in full_data:
    a.loc[(a["Title"] == "Mr") & (a.Age.isnull()), ["Age"]] = mr_age
    a.loc[(a["Title"] == "Mrs") & (a.Age.isnull()), ["Age"]] = mrs_age
    a.loc[(a["Title"] == "Miss") & (a.Age.isnull()), ["Age"]] = miss_age
    a.loc[(a["Title"] == "Master") & (a.Age.isnull()), ["Age"]] = master_age
    a.loc[(a["Title"] == "Ms") & (a.Age.isnull()), ["Age"]] = ms_age
    a.loc[(a["Title"] == "Dr") & (a.Age.isnull()), ["Age"]] = a["Age"].mean()
for a in full_data:
    a["CatAge"] = pd.cut(a["Age"], 6)
print(train[["CatAge", "Survived"]].groupby(["CatAge"]).mean())
# Transformando Age

age_bins = [0, 13.683, 26.947, 40.21, 53.473, 66.737, 100]
age_brackets = ["0", "1", "2", "3", "4", "5"]

for a in full_data:
    a["CatAge"] = pd.cut(a["Age"], age_bins, labels = age_brackets)
print(train[["SibSp", "Survived"]].groupby(["SibSp"]).mean())
print(train[["Parch", "Survived"]].groupby(["Parch"]).mean())
# Creating variables "FamilySize" and "isAlone"
# Criando variaveis "FamilySize" e "isAlone"

for a in full_data:
    a["FamilySize"] = (a["SibSp"].values + a["Parch"].values + 1)
    a.loc[a["FamilySize"] != 1, "isAlone"] = 0
    a.loc[a["FamilySize"] == 1, "isAlone"] = 1
    a["isAlone"] = a["isAlone"].astype(int)
print(train[["FamilySize", "Survived"]].groupby(["FamilySize"]).mean())
print(train[["isAlone", "Survived"]].groupby(["isAlone"]).mean())
sns.pointplot(x="Pclass", y="Fare", data=train)
test[["Pclass"]].loc[test["Fare"].isnull()]
# Filling Null with average Fare when Pclas = 3
# Preenchendo NulL com Fare médio em Pclass = 3

test["Fare"] = test["Fare"].fillna(train.groupby(["Pclass"]).mean()["Fare"][3])
# Grupping "Fare"
# Agrupando "Fare"

for a in full_data:
    a["CatFare"] = pd.qcut(a["Fare"], 4)
print(train[["CatFare", "Survived"]].groupby(["CatFare"]).mean())
for a in full_data:
    a.loc[(a['Fare'] <= 7.91), 'Fare'] = 0
    a.loc[(a["Fare"] > 7.91) & (a["Fare"] <= 14.454), "Fare"] = 1
    a.loc[(a["Fare"] > 14.454) & (a["Fare"] <= 31.0), "Fare"] = 2
    a.loc[(a["Fare"] > 31.0), "Fare"] = 3
    a["Fare"] = a["Fare"].astype(int)
sns.pointplot(x="Embarked", y="Fare", data=train)
train.loc[train["Embarked"].isnull()]
# Filling Null with "Embarked" category most popular in "Fare" = 3
# Preenchendo Null com a categoria "Embarked" mais popular de "Fare" = 3

train["Embarked"] = train.Embarked.fillna("C")
print(train[["Embarked", "Survived"]].groupby(["Embarked"]).mean())
for a in full_data:
    a["CatEmbarked"] = a["Embarked"].map({"C" : 0, "Q" : 1, "S" : 2}).astype(int)
# Removing uselless columns
# Removendo colunas sem uso

train = train.drop(["Name", "Ticket", "Age", "Embarked", "CatFare", "Title"], axis = 1)
test = test.drop(["Name", "Ticket", "Age", "Embarked", "CatFare", "Title"], axis = 1)
train.head(5)
atributos1 = ["Pclass"]
atributos2 = ["Sex"]
atributos3 = ["SibSp"]
atributos4 = ["Parch"]
atributos5 = ["Fare"]
atributos6 = ["CatAge"]
atributos7 = ["FamilySize"]
atributos8 = ["isAlone"]
atributos9 = ["CatEmbarked"]
atributos10 = ["Cabin"]
atributos11 = ["CatTitle"]

top6 = ["Sex", "CatTitle", "Pclass", "FamilySize", "Fare", "Cabin"]
top7 = ["Sex", "CatTitle", "Pclass", "FamilySize", "Fare", "Cabin", "CatEmbarked"]
top8 = ["Sex", "CatTitle", "Pclass", "FamilySize", "Fare", "Cabin", "CatEmbarked", "SibSp"]
top9 = ["Sex", "CatTitle", "Pclass", "FamilySize", "Fare", "Cabin", "CatEmbarked", "SibSp", "Parch"]
top10 = ["Sex", "CatTitle", "Pclass", "FamilySize", "Fare", "Cabin", "CatEmbarked", "SibSp", "Parch", "CatAge"]
top11 = ["Sex", "CatTitle", "Pclass", "FamilySize", "Fare", "Cabin", "CatEmbarked", "SibSp", "Parch", "CatAge", "isAlone"]
Ytr =  train[["Survived"]].values

Xtr1 = train[atributos1].values
Xte1 = test[atributos1].values

Xtr2 = train[atributos2].values
Xte2 = test[atributos2].values

Xtr3 = train[atributos3].values
Xte3 = test[atributos3].values

Xtr4 = train[atributos4].values
Xte4 = test[atributos4].values

Xtr5 = train[atributos5].values
Xte5 = test[atributos5].values

Xtr6 = train[atributos6].values
Xte6 = test[atributos6].values

Xtr7 = train[atributos7].values
Xte7 = test[atributos7].values

Xtr8 = train[atributos8].values
Xte8 = test[atributos8].values

Xtr9 = train[atributos9].values
Xte9 = test[atributos9].values

Xtr10 = train[atributos10].values
Xte10 = test[atributos10].values

Xtr11 = train[atributos11].values
Xte11 = test[atributos11].values

Xtr14 = train[top6].values
Xte14 = test[top6].values

Xtr15 = train[top7].values
Xte15 = test[top7].values

Xtr16 = train[top8].values
Xte16 = test[top8].values

Xtr17 = train[top9].values
Xte17 = test[top9].values

Xtr18 = train[top10].values
Xte18 = test[top10].values

Xtr19 = train[top11].values
Xte19 = test[top11].values
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
# Function to run models
# Função para executar os modelos

def runModel(xydict):
    
    classifiers = [
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    GradientBoostingClassifier()]    
    
    model_dict = {}
        
    for key in xydict:
        
        X = xydict[key][0]
        y = xydict[key][1]
        
        acc_dict = {}
        
        for clf in classifiers:
            
            acc_list = []
                
            name = clf.__class__.__name__

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.22, random_state = 0)
                
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            
            acc_dict[name] = acc 
            
        model_dict[key] = acc_dict
    
    return model_dict
attributes = {"Pclass" : [Xtr1, Ytr],
        "Sex" : [Xtr2, Ytr],
        "SibSp" : [Xtr3, Ytr],
        "Parch" : [Xtr4, Ytr],
        "Fare" : [Xtr5, Ytr],
        "CatAge" : [Xtr6, Ytr],
        "FamilySize" : [Xtr7, Ytr],
        "isAlone" : [Xtr8, Ytr],
        "CatEmbarked" : [Xtr9, Ytr],
        "Cabin" : [Xtr10, Ytr],
        "CatTitle" : [Xtr11, Ytr]}
df = pd.DataFrame(runModel(attributes))
df
sco = df.T["DecisionTreeClassifier"]+df.T["RandomForestClassifier"]+df.T["GradientBoostingClassifier"]
sco = pd.DataFrame(sco/3, columns=["Score"])
sco.sort_values("Score", ascending=False)
plt.figure(figsize=(12,6))
sns.barplot(x = "Score", y = sco.index, data = sco, orient="h",  
            order = ["Sex", "CatTitle", "Cabin", "Pclass", "FamilySize", "Fare", 
                     "CatEmbarked", "SibSp", "CatAge", "Parch", "isAlone"])
plt.xlim((0.55, 0.85))
plt.grid(True)
plt.title("Variable x Score")
plt.xticks(np.arange(0.55, 0.86, step=0.05))
plt.show()
top = {"Top 6" : [Xtr14, Ytr],
        "Top 7" : [Xtr15, Ytr],
        "Top 8" : [Xtr16, Ytr],
        "Top 9" : [Xtr17, Ytr],
        "Top 10" : [Xtr18, Ytr],
        "Top 11" : [Xtr19, Ytr]}
#Run model 50 times and return average
#Executa modelo 50 vezes e retorna a média

modeldfsum = pd.DataFrame(runModel(top))

for i in range(50):
    model = runModel(top)
    modeldf = pd.DataFrame(model)
    modeldfsum += modeldf
    
model = modeldfsum / 51

model["Classifier"] = model.index
model = model.melt('Classifier', var_name='Col',  value_name='Score')
plt.figure(figsize=(12,7))
g = sns.barplot(x="Score", y="Classifier", hue='Col', data=model, palette="Paired", orient='h')
plt.xlim((0.78, 0.88))
plt.grid(True, axis='x')
plt.title("Variables x Score")
plt.xticks(np.arange(0.78, 0.88, step=0.01))
plt.show()
# final_model = RandomForestClassifier()
# final_model.fit(Xtr19, Ytr)
# Y_pred = final_model.predict(Xte19)

# submission = pd.DataFrame({
#        "PassengerId": test["PassengerId"],
#        "Survived": Y_pred})

# submission.to_csv('submission.csv', index=False)