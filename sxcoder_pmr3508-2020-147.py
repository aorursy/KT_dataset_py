import sklearn

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
# Definição do nome das colunas e visualização dos primeiros dados do dataset

col_names = ["Age", "Workclass", "Final Weight", "Education", "Education-Num"

            ,"Marital Status", "Occupation", "Relationship", "Race", "Sex",

            "Capital Gain", "Capital Loss", "Hours per week", "Country", "Income"]



# Os dados faltantes serão identificados como ?

df = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv",

        sep=',',

        engine='python',

        na_values="?",

        index_col=['Id'])



df.columns = col_names



df.head(10)
df.shape
df.info()
df.describe()
# Transformar variável de classe (Income) em numérica

from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()



df['Income'] = le.fit_transform(df['Income'])
df['Income']
sns.heatmap(df.corr().round(2), vmin=-1., vmax=1., cmap = plt.cm.RdYlGn_r, annot=True )
sns.catplot(y="Workclass", x="Income", kind="bar", data=df)
sns.catplot(y="Education", x="Income", kind="bar", data=df)
sns.catplot(y="Marital Status", x="Income", kind="bar", data=df)
sns.catplot(y="Occupation", x="Income", kind="bar", data=df)
sns.catplot(y="Relationship", x="Income", kind="bar", data=df)
sns.catplot(y="Race", x="Income", kind="bar", data=df)
sns.catplot(y="Sex", x="Income", kind="bar", data=df)
sns.catplot(y="Country", x="Income", kind="bar", data=df)
df["Country"].value_counts()
# Remoção de dados duplicados



df.drop_duplicates(keep="first", inplace=True)



# Remoção de colunas que segundo a análise podem ser desconsideradas



df = df.drop(["Final Weight", "Country"], axis=1)
df.head()
# Separação dos dados



# Variável de classe

Y_train = df.pop('Income')



# Variáveis Independentes

X_train = df
# Seleciona variáveis categóricas

categorical_cols = list(X_train.select_dtypes(exclude=[np.number]).columns.values)



# Seleciona variáveis esparsas

sparse_cols = ['Capital Gain', 'Capital Loss']



# Seleciona variáveis numéricas

numerical_cols = list(X_train.select_dtypes(include=[np.number]).columns.values)



numerical_cols.remove('Capital Gain')

numerical_cols.remove('Capital Loss')
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import Pipeline



categorical_pipeline = Pipeline(steps = [

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(drop='if_binary'))

])
from sklearn.preprocessing import StandardScaler



numerical_pipeline = Pipeline(steps=[

    ('scaler', StandardScaler())

])
from sklearn.preprocessing import RobustScaler



sparse_pipeline = Pipeline(steps=[

    ('scaler', RobustScaler())

])
# Junção das pipelines em apenas um transformador

from sklearn.compose import ColumnTransformer



preprocessor = ColumnTransformer(transformers=[

    ('num', numerical_pipeline, numerical_cols),

    ('spr', sparse_pipeline, sparse_cols),

    ('cat', categorical_pipeline, categorical_cols)

])



X_train = preprocessor.fit_transform(X_train)
from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsClassifier



# Quantidade de vizinhos que serão testadas:

neighbors = [5,10,15,20,25,30,35,40]



for n in neighbors:

    score = cross_val_score(KNeighborsClassifier(n_neighbors=n), X_train, Y_train, cv=13, scoring="accuracy").mean()

    

    print("Número de vizinhos: ", n," | Acurácia: ", score)
# Criar um kNN para esse número de vizinhos



knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X_train, Y_train)
test_data = pd.read_csv("/kaggle/input/adult-pmr3508/test_data.csv",

        sep=r'\s*,\s*',

        engine='python',

        na_values="?",

        index_col=['Id'])



# Definição do nome das colunas e visualização dos primeiros dados do dataset

col_names_test = ["Age", "Workclass", "Final Weight", "Education", "Education-Num"

            ,"Marital Status", "Occupation", "Relationship", "Race", "Sex",

            "Capital Gain", "Capital Loss", "Hours per week", "Country"]



test_data.columns = col_names_test 
X_test = test_data.drop(["Final Weight", "Country"], axis=1)
X_test = preprocessor.transform(X_test)
prediction = knn.predict(X_test)
prediction
# Substituindo os valores 0 e 1 para os valores iniciais para a variável Income



subs = {0: '<=50K', 1: '>50K'}

prediction_str = np.array([subs[i] for i in prediction], dtype=object)
prediction_str
submission = pd.DataFrame()

submission[0] = test_data.index

submission[1] = prediction_str

submission.columns = ['Id', 'Income']





submission.head()
submission.to_csv('submission.csv', index=False)