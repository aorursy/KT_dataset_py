import numpy as np

import sklearn as sk

import pandas as pd

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn import preprocessing

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

import missingno as msno
adult = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv",

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")



adult = adult.rename(columns={"age":"Age","workclass":"Workclass","education":"Education",

                              "education.num":"Education-Num","marital.status":"Marital-Status",

                              "occupation":"Occupation","relationship":"Relationship","race":"Race","sex":"Sex",

                             "capital.gain":"Capital-Gain","capital.loss":"Capital-Loss",

                              "hours.per.week":"Hours-Per-Week","native.country":"Native-Country",

                              "income":"Income"})
print("Forma do DataFrame: ",adult.shape)
adult.head()
#Imprime as regioes em que há dados faltantes

msno.matrix(adult)
#Imprime o valor total de dados faltantes e sua porcentagem.

total = adult.isnull().sum().sort_values(ascending = False)

percent = ((adult.isnull().sum()/adult.isnull().count())*100).sort_values(ascending = False)

missing_data = pd.concat([total, percent], axis = 1, keys = ['Total', '%'])

missing_data.head()
print("Occupation: \n")

print(adult["Occupation"].describe(),"\n")

adult["Occupation"].value_counts().plot(kind="bar")
print("Workclass: \n")

print(adult["Workclass"].describe(), "\n")

adult["Workclass"].value_counts().plot(kind="bar")
print("Native-Country: \n")

print(adult["Native-Country"].describe(), "\n")

adult["Native-Country"].value_counts().plot(kind="bar")
value = adult['Workclass'].describe().top

adult['Workclass'] = adult['Workclass'].fillna(value)



value = adult['Native-Country'].describe().top

adult['Native-Country'] = adult['Native-Country'].fillna(value)
#Imprime o valor total de dados faltantes e sua porcentagem.

total = adult.isnull().sum().sort_values(ascending = False)

percent = ((adult.isnull().sum()/adult.isnull().count())*100).sort_values(ascending = False)

missing_data = pd.concat([total, percent], axis = 1, keys = ['Total', '%'])

missing_data.head()
adult = adult.dropna()



#Imprime o valor total de dados faltantes e sua porcentagem.

total = adult.isnull().sum().sort_values(ascending = False)

percent = ((adult.isnull().sum()/adult.isnull().count())*100).sort_values(ascending = False)

missing_data = pd.concat([total, percent], axis = 1, keys = ['Total', '%'])

missing_data.head()
adult = adult.drop(["fnlwgt"], axis=1)
adult.head()
adult_menor = adult.loc[adult['Income'] == '<=50K']

adult_maior = adult.loc[adult['Income'] == '>50K']
pd.concat([adult_menor["Age"].value_counts(), adult_maior["Age"].value_counts()],

          keys=["<=50K", ">50K"], axis=1).plot(kind='bar', xlabel='Age', ylabel='num', figsize=(20, 10))
pd.concat([adult_menor["Workclass"].value_counts(), adult_maior["Workclass"].value_counts()],

          keys=["<=50K", ">50K"], axis=1).plot(kind='bar', xlabel='Age', ylabel='num', figsize=(20, 10))
pd.concat([adult_menor["Education"].value_counts(), adult_maior["Education"].value_counts()],

          keys=["<=50K", ">50K"], axis=1).plot(kind='bar', xlabel='Age', ylabel='num', figsize=(20, 10))
pd.concat([adult_menor["Education-Num"].value_counts(), adult_maior["Education-Num"].value_counts()],

          keys=["<=50K", ">50K"], axis=1).plot(kind='bar', xlabel='Age', ylabel='num', figsize=(20, 10))
pd.concat([adult_menor["Marital-Status"].value_counts(), adult_maior["Marital-Status"].value_counts()],

          keys=["<=50K", ">50K"], axis=1).plot(kind='bar', xlabel='Age', ylabel='num', figsize=(20, 10))
pd.concat([adult_menor["Occupation"].value_counts(), adult_maior["Occupation"].value_counts()],

          keys=["<=50K", ">50K"], axis=1).plot(kind='bar', xlabel='Age', ylabel='num', figsize=(20, 10))
pd.concat([adult_menor["Relationship"].value_counts(), adult_maior["Relationship"].value_counts()],

          keys=["<=50K", ">50K"], axis=1).plot(kind='bar', xlabel='Age', ylabel='num', figsize=(20, 10))
pd.concat([adult_menor["Race"].value_counts(), adult_maior["Race"].value_counts()],

          keys=["<=50K", ">50K"], axis=1).plot(kind='bar', xlabel='Age', ylabel='num', figsize=(20, 10))
pd.concat([adult_menor["Sex"].value_counts(), adult_maior["Sex"].value_counts()],

          keys=["<=50K", ">50K"], axis=1).plot(kind='bar', xlabel='Age', ylabel='num', figsize=(20, 10))
pd.concat([adult_menor["Capital-Gain"].value_counts(), adult_maior["Capital-Gain"].value_counts()],

          keys=["<=50K", ">50K"], axis=1).plot(kind='bar', xlabel='Age', ylabel='num', figsize=(20, 10))
pd.concat([adult_menor["Capital-Loss"].value_counts(), adult_maior["Capital-Loss"].value_counts()],

          keys=["<=50K", ">50K"], axis=1).plot(kind='bar', xlabel='Age', ylabel='num', figsize=(20, 10))
pd.concat([adult_menor["Hours-Per-Week"].value_counts(), adult_maior["Hours-Per-Week"].value_counts()],

          keys=["<=50K", ">50K"], axis=1).plot(kind='bar', xlabel='Age', ylabel='num', figsize=(20, 10))
pd.concat([adult_menor["Native-Country"].value_counts(), adult_maior["Native-Country"].value_counts()],

          keys=["<=50K", ">50K"], axis=1).plot(kind='bar', xlabel='Age', ylabel='num', figsize=(20, 10))
adult_analize = adult



# Transformar variável de classe (Income) em numérica

from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()



adult_analize['Income'] = le.fit_transform(adult_analize['Income'])





#Tirando a classe "Id"

adult_analize = adult_analize.drop(["Id"], axis=1)
sns.heatmap(adult_analize.corr().round(2), vmin=-1., vmax=1., cmap = plt.cm.RdYlGn_r, annot=True )
adult = adult.drop(["Native-Country"], axis=1)

adult = adult.drop(["Workclass"], axis=1)

adult = adult.drop(["Race"], axis=1)

adult = adult.drop(["Hours-Per-Week"], axis=1)

adult = adult.drop(["Id"], axis=1)

adult.head()
# Encoda os dados STRING em classificação ordinal

stringfeatures = ["Marital-Status","Education","Occupation", "Relationship", "Sex"]



numadult = adult

numadult[stringfeatures] = numadult[stringfeatures].apply(preprocessing.LabelEncoder().fit_transform) 



# Normalização

normalizar = ["Age", "Education-Num", "Marital-Status","Occupation","Relationship",

            "Sex"]



for col in normalizar:

    numadult[col] = (numadult[col] - numadult[col].min()) / (numadult[col].max() - numadult[col].min())
numadult = adult

numadult.head()
y_train = numadult.pop("Income")

x_train = numadult

x_train.head()
maxk = 0

maxscore = 0

knn = None

maxknn = None

for k in range(26,27):

    knn = KNeighborsClassifier(k,n_jobs=-1)

    scores = cross_val_score(knn, x_train, y_train, cv=10)

    print(f"{k} {scores.mean():.6f} {scores.max():.6f}")

    if scores.mean() > maxscore:

        maxscore = scores.mean()

        maxk = k

        maxknn = knn
# Criar um kNN para esse número de vizinhos



knn = KNeighborsClassifier(n_neighbors=26)
knn.fit(x_train, y_train)
adult_test = pd.read_csv("/kaggle/input/adult-pmr3508/test_data.csv",

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")



adult_test = adult_test.rename(columns={"age":"Age","workclass":"Workclass","education":"Education",

                              "education.num":"Education-Num","marital.status":"Marital-Status",

                              "occupation":"Occupation","relationship":"Relationship","race":"Race","sex":"Sex",

                             "capital.gain":"Capital-Gain","capital.loss":"Capital-Loss",

                              "hours.per.week":"Hours-Per-Week","native.country":"Native-Country",

                              "income":"Income"})
adult_test = adult_test.drop(["Native-Country"], axis=1)

adult_test = adult_test.drop(["Workclass"], axis=1)

adult_test = adult_test.drop(["Race"], axis=1)

adult_test = adult_test.drop(["Hours-Per-Week"], axis=1)

adult_test = adult_test.drop(["Id"], axis=1)

adult_test = adult_test.drop(["fnlwgt"], axis=1)
adult_test.head()
value = adult_test['Occupation'].describe().top

adult_test['Occupation'] = adult_test['Occupation'].fillna(value)
# Encoda os dados STRING em classificação ordinal

stringfeatures = ["Marital-Status","Education","Occupation", "Relationship", "Sex"]



x_test = adult_test

x_test[stringfeatures] = x_test[stringfeatures].apply(preprocessing.LabelEncoder().fit_transform) 



# Normalização

normalizar = ["Age", "Education-Num", "Marital-Status","Occupation","Relationship",

            "Sex"]



for col in normalizar:

    x_test[col] = (x_test[col] - x_test[col].min()) / (x_test[col].max() - x_test[col].min())
prediction = knn.predict(x_test)
prediction
# Substituindo os valores 0 e 1 para os valores iniciais para a variável Income



subs = {0: '<=50K', 1: '>50K'}

prediction_str = np.array([subs[i] for i in prediction], dtype=object)
prediction_str
submission = pd.DataFrame()

submission[0] = adult_test.index

submission[1] = prediction_str

submission.columns = ['Id', 'Income']





submission.head()
submission.to_csv('submission.csv', index=False)