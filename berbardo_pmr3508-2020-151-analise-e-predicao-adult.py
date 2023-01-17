import numpy as np

import pandas as pd
!pip install scikit-learn==0.23.0



from numpy.ma import MaskedArray

import sklearn.utils.fixes



sklearn.utils.fixes.MaskedArray = MaskedArray
df = pd.read_csv("../input/adult-pmr3508/train_data.csv", index_col=['Id'], na_values="?")
df.head()
df.info()
df.describe()
import matplotlib.pyplot as plt

import seaborn as sns
# Copia "df" para "df_analysis"

df_analysis = df.copy()
# Importando o LabelEncoder

from sklearn.preprocessing import LabelEncoder



# Instanciando o LabelEncoder

le = LabelEncoder()



# Modificando o nosso dataframe, transformando a variável de classe em 0s e 1s

df_analysis['income'] = le.fit_transform(df_analysis['income'])
df_analysis['income']
mask = np.triu(np.ones_like(df_analysis.corr(), dtype=np.bool))



plt.figure(figsize=(10,10))



sns.heatmap(df_analysis.corr(), mask=mask, square = True, annot=True, vmin=-1, vmax=1, cmap='autumn')

plt.show()
sns.distplot(df_analysis['age']);
sns.catplot(x="income", y="hours.per.week", data=df_analysis);
sns.catplot(x="income", y="hours.per.week", kind="boxen", data=df_analysis);
sns.catplot(x="income", y="education.num", kind="boxen", data=df_analysis);
sns.catplot(x="income", y="age", kind="boxen", data=df_analysis);
sns.catplot(x="income", y="capital.gain", kind="boxen", data=df_analysis);
sns.catplot(x="income", y="capital.loss", kind="boxen", data=df_analysis);
sns.catplot(x="income", y="capital.gain", data=df_analysis);
sns.catplot(x="income", y="capital.loss", data=df_analysis);
df_analysis.describe()
sns.catplot(y="sex", x="income", kind="bar", data=df_analysis);
sns.catplot(y="race", x="income", kind="bar", data=df_analysis);
sns.catplot(y="workclass", x="income", kind="bar", data=df_analysis);
sns.catplot(y="marital.status", x="income", kind="bar", data=df_analysis);
sns.catplot(y="occupation", x="income", kind="bar", data=df_analysis);
sns.catplot(y="native.country", x="income", kind="bar", data=df_analysis);
df_analysis["native.country"].value_counts()
df.drop_duplicates(keep='first', inplace=True)
df = df.drop(['fnlwgt', 'native.country', 'education'], axis=1)
df.head()
# Removendo a nossa variável de classe

Y_train = df.pop('income')



X_train = df
X_train.head()
# Seleciona as variáveis numéricas

numerical_cols = list(X_train.select_dtypes(include=[np.number]).columns.values)



# Remove as variáveis numéricas esparsas

numerical_cols.remove('capital.gain')

numerical_cols.remove('capital.loss')



# Seleciona as variáveis numéricas esparsas

sparse_cols = ['capital.gain', 'capital.loss']



# Seleciona as variáveis categóricas

categorical_cols = list(X_train.select_dtypes(exclude=[np.number]).columns.values)



# Mostrando as diferentes seleções

print("Colunas numéricas: ", numerical_cols)

print("Colunas esparsas: ", sparse_cols)

print("Colunas categóricas: ", categorical_cols)
from sklearn.impute import SimpleImputer



# Inicializa nosso Imputer

simple_imputer = SimpleImputer(strategy='most_frequent')
# Cria um array com um dado faltante

array = np.array([["Female"],

         ["Male"],

         [np.nan],

         ["Female"]], dtype=object)



# Preenche o dado faltante com o Imputer

new_array = simple_imputer.fit_transform(array)



print(new_array)
from sklearn.preprocessing import OneHotEncoder



# Inicializa nosso Encoder

one_hot = OneHotEncoder(sparse=False)
# Cria um array com dados categóricos

array = np.array([["Female"],

         ["Male"],

         ["Female"],

         ["Female"]], dtype=object)



# Transforma o nosso array

new_array = one_hot.fit_transform(array)



new_array
from sklearn.pipeline import Pipeline



# Cria a nossa pipeline categórica

categorical_pipeline = Pipeline(steps = [

    ('imputer', SimpleImputer(strategy = 'most_frequent')),

    ('onehot', OneHotEncoder(drop='if_binary'))

])
# Cria um array com dados categóricos

array = np.array([["Female"],

         ["Male"],

         [np.nan],

         ["Female"]], dtype=object)



# Transforma o nosso array

new_array = categorical_pipeline.fit_transform(array)
from sklearn.impute import KNNImputer



# Cria o nosso KNNImputer com 5 vizinhos

knn_imputer = KNNImputer(n_neighbors=5)
# Cria o nosso array com dados faltantes

array = [[1, 2, np.nan], [3, 4, 3], [np.nan, 6, 5], [8, 8, 7]]



# Preenche os dados faltantes

new_array = knn_imputer.fit_transform(array)



new_array
from sklearn.preprocessing import StandardScaler



# Cria o nosso StandardScaler

scaler = StandardScaler()
# Cria um array numérico

array = [[-3, 0], [0, 0], [3, 1], [0, 1]]



# Normaliza nosso array

new_array = scaler.fit_transform(array)



new_array
# Cria a nossa pipeline numérica

numerical_pipeline = Pipeline(steps = [

    ('imputer', KNNImputer(n_neighbors=10, weights="uniform")),

    ('scaler', StandardScaler())

])
from sklearn.preprocessing import RobustScaler



sparse_pipeline = Pipeline(steps = [

    ('imputer', KNNImputer(n_neighbors=10, weights="uniform")),

    ('scaler', RobustScaler())

])
from sklearn.compose import ColumnTransformer



# Cria o nosso Pré-Processador



# Cada pipeline está associada a suas respectivas colunas no datast

preprocessor = ColumnTransformer(transformers = [

    ('num', numerical_pipeline, numerical_cols),

    ('spr', sparse_pipeline, sparse_cols),

    ('cat', categorical_pipeline, categorical_cols)

])
X_train = preprocessor.fit_transform(X_train)
from sklearn.neighbors import KNeighborsClassifier



# Instancia nosso classificador

knn = KNeighborsClassifier(n_neighbors=20)
from sklearn.model_selection import cross_val_score



score = cross_val_score(knn, X_train, Y_train, cv = 5, scoring="accuracy")

print("Acurácia com cross validation:", score.mean())
# Quantidades de vizinhos a serem testadas

neighbors = [15, 20, 25, 30, 35]



# Dicionário para guardar as pontuações de cada hiperparâmetro

neighbors_scores = {}



for n_neighbors in neighbors:

    # Calcula a média de acurácia de cada classificador

    score = cross_val_score(KNeighborsClassifier(n_neighbors=n_neighbors), X_train, Y_train, cv = 5, scoring="accuracy").mean()

    

    # Guarda essa acurácia

    neighbors_scores[n_neighbors] = score



# Obtém a quantidade de vizinhos com o melhor desempenho

best_n = max(neighbors_scores, key=neighbors_scores.get)



print("Melhor hiperparâmetro: ", best_n)

print("Melhor acurácia: ", neighbors_scores[best_n])
# Importa o Bayes Search:

from skopt import BayesSearchCV



from sklearn.model_selection import ShuffleSplit



# Importa o espaço de busca inteiro

from skopt.space import Integer



cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)





# Cria o Bayes Search:

bayes_search_cv = BayesSearchCV(estimator = KNeighborsClassifier(),

                              search_spaces = {'n_neighbors': Integer(10, 50),}, # Vizinhos de 10 a 50

                              cv = cv,

                              n_iter = 12, random_state=42)



# Realizando a otimização por BayesSearch:

bayes_search_cv.fit(X_train, Y_train)



best_param = bayes_search_cv.best_params_['n_neighbors']



print('Melhor quantidade de vizinhos: {}'.format(bayes_search_cv.best_params_['n_neighbors']))

print('Desempenho do melhor modelo: {}'.format(round(bayes_search_cv.best_score_,5)))
knn = KNeighborsClassifier(n_neighbors=26)
knn.fit(X_train, Y_train)
test_data = pd.read_csv("../input/adult-pmr3508/test_data.csv", index_col=['Id'], na_values="?")
X_test = test_data.drop(['fnlwgt', 'native.country', 'education'], axis=1)
X_test = preprocessor.transform(X_test)
predictions = knn.predict(X_test)
predictions
submission = pd.DataFrame()
submission[0] = test_data.index

submission[1] = predictions

submission.columns = ['Id','income']
submission.head()
submission.to_csv('submission.csv',index = False)