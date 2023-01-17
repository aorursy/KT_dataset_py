#3I0203 UI1
# Importovanie potrebných balíčkov

import pandas as pd

import numpy as np

import random as rnd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OrdinalEncoder

from sklearn.impute import SimpleImputer

from sklearn.compose import make_column_transformer

from sklearn.pipeline import make_pipeline

from sklearn.metrics import accuracy_score
#Načítanie súborov

train_df = pd.read_csv('../input/titanic/train.csv')

Xtest_df = pd.read_csv('../input/titanic/test.csv')
#Informatívny výpis na zistenie dátového typu a množstva chýbajúcich hodnôt

train_df.info()
#Odstránenie stĺpca s číslom kabíny z dôvodu veľkého množstva chýbajúcich hodnôt a ich náročného doplnenia tak, aby bola zachovaná istá logika 

#Odstránenie stĺpca s informaciou o čísle palubného lístka 

#Odstránenie stĺpca s Menami cestujúcich - táto informácia by mohla byť užitočná, ale vyźadovala by zložitejšie predspracovanie

train_df = train_df.drop('Name', axis=1,)

train_df = train_df.drop('Ticket', axis=1,)

train_df = train_df.drop('Cabin', axis=1,)
#Doplnienie chýbajúcich hodnôt do stĺpcov Age a Embarked

train_df["Embarked"] = train_df["Embarked"].fillna("C")

train_df["Age"] = train_df["Age"].fillna(train_df["Age"].median())
#Overenie doplnenia - výpis počtu nulových hodnôt

checklist = list(train_df.columns.values)

for check in checklist:

    print (check,": ",sum(pd.isnull(train_df[check])))
#Výpis niekoľkých vzoriek upravenej tréningovej dátovej množiny

train_df.head()
#Rozdelenie dátovej množiny na tréningové a testovacie dáta

df_train, df_test = train_test_split(train_df, test_size=0.1,

stratify=train_df["Survived"], random_state=3)
#Rozdelenie numerical a categorical vstupov

categorical_inputs = ["Sex","Embarked"]  

numeric_inputs = ["Pclass","Age","SibSp","Parch","Fare","PassengerId"] 

output = "Survived"
#Pipeline, Encoder 

input_preproc = make_column_transformer(

    (make_pipeline(

        SimpleImputer(strategy="most_frequent"),

        OrdinalEncoder()),

     categorical_inputs),    

    (make_pipeline(

        SimpleImputer(),

        StandardScaler()),

     numeric_inputs)

)  

output_enc = OrdinalEncoder()
#Ladenie a predspracovanie

X_train = input_preproc.fit_transform(df_train[categorical_inputs+numeric_inputs])

Y_train = df_train[output].values.reshape(-1)



X_test = input_preproc.transform(df_test[categorical_inputs+numeric_inputs])

Y_test = df_test[output].values.reshape(-1)
#Vytvorenie modelu K Nearest Neighbour (KNN)

model = KNeighborsClassifier(n_neighbors=30, weights='distance', algorithm='auto', leaf_size=10, p=2)

model.fit(X_train, Y_train)
#Testovanie s použitím testovacích dát

y_test = model.predict(X_test)
#Ohodnotenie presnosti modelu - krížová validácie

cm = pd.crosstab(Y_test, y_test,

                 rownames=['actual'],

                 colnames=['predicted'])

print(cm)



acc = accuracy_score(Y_test, y_test)

print("Accuracy = {}".format(acc))
#Úprava testovacích dát

Xtest_df = Xtest_df.drop('Name', axis=1,)

Xtest_df = Xtest_df.drop('Ticket', axis=1,)

Xtest_df = Xtest_df.drop('Cabin', axis=1,)
#Doplnienie chýbajúcich hodnôt do stĺpcov Age a Embarked

Xtest_df["Embarked"] = train_df["Embarked"].fillna("C")

Xtest_df["Age"] = train_df["Age"].fillna(train_df["Age"].median())
#Výpis niekoľkých vzoriek upravenej testovacej dátovej množiny

Xtest_df.head()
#Ladenie a predspracovanie

X_testt = input_preproc.transform(Xtest_df[categorical_inputs+numeric_inputs])

Y_testt = Xtest_df.values.reshape(-1)
#predikcia

y_testt = model.predict(X_testt)
#odoslanie výsledkov

submission = pd.DataFrame({

        "PassengerId": Xtest_df["PassengerId"],

        "Survived": y_testt

    })

submission.to_csv('vysledok.csv', index=False)