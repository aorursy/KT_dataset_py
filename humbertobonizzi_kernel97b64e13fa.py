import sys



#carregemento das bibliotecas de suporte

import pandas as pd

import numpy as np



#Bibliotecas de treino dos modelos

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB

from sklearn.svm import LinearSVC, SVC

from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.model_selection import cross_val_predict, RandomizedSearchCV, cross_val_score

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
df = pd.read_csv('/kaggle/input/titanic/train.csv')

df_teste = pd.read_csv('/kaggle/input/titanic/test.csv')

df_resposta = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
df.tail()
df.columns
df.describe()
df.info()
x_treino = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

y_treino = df['Survived']



x_teste = df_teste[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

y_teste = df_resposta['Survived']
colunas_numericas = ['Age', 'SibSp', 'Parch', 'Fare']

transfomador_numerico = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='median')),

    ('scaler', StandardScaler())

])
colunas_categoricas_numero = [ 'Pclass' ]

transformador_categorico_numero = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])
colunas_categoricas_texto = ['Sex', 'Embarked']

transformador_categorico_texto = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])
preprocessador = ColumnTransformer(transformers=[

    ('num', transfomador_numerico, colunas_numericas),

    ('cat_texto', transformador_categorico_texto, colunas_categoricas_texto),

    ('cat_num', transformador_categorico_numero, colunas_categoricas_numero)

    

])
classificador = Pipeline(steps=[

    ('preprocessador', preprocessador)

    #('classificador', RandomForestClassifier(random_state=42))

])
base_tratada_treino = classificador.fit_transform(x_treino)

base_tratada_teste = classificador.transform(x_teste)
clf = VotingClassifier(estimators = [('logistic', LogisticRegression()), 

                                     ('SVM', LinearSVC(random_state=42, max_iter=3000)),

                                     ('random_forest', RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)),

                                     ('adaboost', AdaBoostClassifier(n_estimators=100)),

                                     ('gradient', GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=42))

                                    ])
clf.fit(base_tratada_treino, y_treino)
y_pred = clf.predict(base_tratada_teste)
scores = cross_val_score(clf, base_tratada_treino, y_treino, cv=10, scoring='f1_macro')

print("scores: ", scores)

print("média: ", scores.mean())
print(classification_report(y_teste, y_pred))