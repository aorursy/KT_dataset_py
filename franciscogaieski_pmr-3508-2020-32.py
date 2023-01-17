import pandas as pd

import numpy as np
adultDF = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv", na_values="?", index_col=['Id'])

adultDF.head()
print('Shape do DataFrame:',adultDF.shape)
import seaborn as sns



import matplotlib.pyplot as plt
dfAnalysis = adultDF.copy()



# Importando o LabelEncoder

from sklearn.preprocessing import LabelEncoder



# Instanciando o LabelEncoder

le = LabelEncoder()



# Modificando o nosso dataframe, transformando a variável de classe em 0s e 1s

dfAnalysis['income'] = le.fit_transform(dfAnalysis['income'])
# Cálculo da matriz de correlação

corr = dfAnalysis.corr()



# Geração de uma máscara para o triângulo superior

mask = np.triu(np.ones_like(corr, dtype=bool))



# Configuração da figura em matplotlib

f, ax = plt.subplots(figsize=(11, 9))



# Geração de um mapa de cores divergente

cmap = sns.diverging_palette(230, 20, as_cmap=True)



# Desenho do mapa de calor com a máscara e a proporção correta

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=0.5, vmin = -0.5, center=0, annot=True,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
sns.catplot(x="income", y="age", kind="violin", data=adultDF)
sns.catplot(x="income", y="education.num", kind="violin", data=adultDF)
sns.catplot(x="income", y="capital.gain", kind="violin", data=adultDF)
sns.catplot(x="income", y="capital.loss", kind="violin", data=adultDF)
sns.catplot(x="income", y="hours.per.week", kind="violin", data=adultDF)
sns.catplot(x="workclass", col="income", col_wrap=4, data=adultDF, kind="count", aspect=2)
sns.catplot(x="marital.status", col="income", col_wrap=4, data=adultDF, kind="count", aspect = 2)
sns.catplot(x="occupation", col="income", col_wrap=4, data=adultDF, kind="count", aspect=4)
sns.catplot(x="relationship", col="income", col_wrap=4, data=adultDF, kind="count", aspect=1)
sns.catplot(x="race", col="income", col_wrap=4, data=adultDF, kind="count", aspect=1.4)
sns.catplot(x="sex", col="income", col_wrap=4, data=adultDF, kind="count")
adultDF['sex'].value_counts()
sns.catplot(x="income", col="native.country", col_wrap=4, data=adultDF, kind="count")
adultDF['native.country'].value_counts()
adultDF.isnull().sum().sort_values(ascending = False).head(5)
adultDF = adultDF.dropna()
adultDF.isnull().sum().sort_values(ascending = False).head(5)
dfAnalysis = dfAnalysis.dropna()



# Instanciando o LabelEncoder

le = LabelEncoder()



# Transformando as variáveis categóricas em numéricas

categoricColumn = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex']

for column in categoricColumn:

    dfAnalysis[column] = le.fit_transform(dfAnalysis[column])
dfAnalysis.head()
# Cálculo da matriz de correlação

corr = dfAnalysis.corr()



# Geração de uma máscara para o triângulo superior

mask = np.triu(np.ones_like(corr, dtype=bool))



# Configuração da figura em matplotlib

f, ax = plt.subplots(figsize=(11, 9))



# Geração de um mapa de cores divergente

cmap = sns.diverging_palette(230, 20, as_cmap=True)



# Desenho do mapa de calor com a máscara e a proporção correta

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=0.5, vmin = -0.5, center=0, annot=True,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
choosenFeatures = ["age", "education.num", "sex", "capital.gain", "capital.loss", "hours.per.week", "income"]



adultTrain = dfAnalysis[choosenFeatures].copy()
adultTrain.head()
XAdultTrain = adultTrain.drop(["income"], axis = 1)

YAdultTrain = adultTrain["income"]
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV
#Configurações para o GridSearch

kRange = list(range(5, 31))

pOptions = list(range(1,3))

gridParameters = dict(n_neighbors=kRange, p=pOptions)
knnTreino = KNeighborsClassifier(n_neighbors=5)



grid = GridSearchCV(knnTreino, gridParameters, cv=10, scoring='accuracy', n_jobs = -2)  
grid.fit(XAdultTrain, YAdultTrain)

print(grid.best_estimator_)

print(grid.best_score_)
knnFinal = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=None, n_neighbors=26, p=1, weights='uniform')



knnFinal.fit(XAdultTrain, YAdultTrain)
#Importação da base de teste

adultTest = pd.read_csv("/kaggle/input/adult-pmr3508/test_data.csv", na_values="?", index_col=['Id'])

adultTest.shape
#Preenchimento das linhas com valores faltantes

adultTest.isnull().sum().sort_values(ascending = False).head(5)
columnWithNull = ['occupation', 'workclass', 'native.country']

for column in columnWithNull:

    value = adultTest[column].describe().top

    adultTest[column] = adultTest[column].fillna(value)
adultTest.isnull().sum().sort_values(ascending = False).head(5)
# Instanciando o LabelEncoder

le = LabelEncoder()



# Transformando as variáveis categóricas em numéricas

for column in categoricColumn:

    adultTest[column] = le.fit_transform(adultTest[column])
choosenFeatures = ["age", "education.num", "sex", "capital.gain", "capital.loss", "hours.per.week"]



adultTest = adultTest[choosenFeatures].copy()

adultTest.head()
YAdultTest = knnFinal.predict(adultTest)
finalArray = []



for i in range(len(YAdultTest)):

    if (YAdultTest[i] == 0):

        finalArray.append('<=50K')

    else:

        finalArray.append('>50K')

        

#transformação do array em DataFrame

finalDF = pd.DataFrame({'income': finalArray})
finalDF.to_csv("submission.csv", index = True, index_label = 'Id')