import numpy as np #biblioteca para lidar com arrays

import pandas as pd #biblioteca para trabalhar com dataframes

import matplotlib.pyplot as plt #biblioteca para gerar gráficos

%matplotlib inline 

import seaborn as sns #biblioteca gráfica com mais estilos

import sklearn #biblioteca com algoritmos de aprendizado de máquina
df = pd.read_csv("../input/adult-pmr3508/train_data.csv", index_col=['Id'], na_values="?")

df_corr = df.copy() 
df.head()
df = df.drop(['education', 'relationship'], axis=1)
df.head()
df.describe()
sns.catplot(x="age", y="income", kind="violin", split=True, data=df, height=7, aspect=1)
sns.catplot(x="fnlwgt", y="income", kind="violin", split=True, data=df, height=7, aspect=1)
sns.catplot(x="education.num", y="income", kind="violin", split=True, data=df, height=7, aspect=1)
sns.catplot(x="capital.gain", y="income", kind="violin", split=True, data=df, height=7, aspect=1)
sns.catplot(x="capital.loss", y="income", kind="violin", split=True, data=df, height=7, aspect=1)
sns.catplot(x="hours.per.week", y="income", kind="violin", split=True, data=df, height=7, aspect=1)
sns.catplot(x="workclass", hue="income", kind="count", data=df, height=7, aspect=1)

plt.xticks(rotation=45)
sns.catplot(x="marital.status", hue="income", kind="count", data=df, height=7, aspect=1)

plt.xticks(rotation=45)
sns.catplot(x="occupation", hue="income", kind="count", data=df, height=7, aspect=1)

plt.xticks(rotation=45)
sns.catplot(x="race", hue="income", kind="count", data=df, height=7, aspect=1)

plt.xticks(rotation=45)
sns.catplot(x="sex", hue="income", kind="count", data=df, height=7, aspect=1)

plt.xticks(rotation=45)
sns.catplot(x="native.country", hue="income", kind="count", data=df, height=7, aspect=2.7)

plt.xticks(rotation=45)
df = df.drop(['native.country'], axis=1)

df.head()
Y_train = df.pop('income')

X_train = df



X_train.head()
Y_train.head()
X_train.info()
for column in ['workclass', 'occupation']:

    X_train[column].fillna(X_train[column].mode()[0], inplace=True)



X_train.info()
categorical_cols = list(X_train.select_dtypes(exclude=[np.number]).columns.values)

print(categorical_cols)
for column in categorical_cols:

  X_train[column] = X_train[column].astype('category')



X_train.info()
for column in categorical_cols:

  X_train[column] = X_train[column].cat.codes



X_train.head()
from sklearn.preprocessing import LabelEncoder



# Instanciando o LabelEncoder

le = LabelEncoder()



# Modificando o nosso dataframe, transformando a variável de classe em 0s e 1s

df_corr['income'] = le.fit_transform(df_corr['income'])
mask = np.triu(np.ones_like(df_corr.corr(), dtype=np.bool))



plt.figure(figsize=(10,10))



sns.heatmap(df_corr.corr(), mask=mask, square = True, annot=True, vmin=-1, vmax=1, cmap='winter')

plt.show()
X_train = X_train.drop(['fnlwgt'], axis=1)



X_train.head()
plt.figure(figsize=(10,10))

sns.distplot(X_train['age'])

plt.show()
X_train["age"].value_counts()
plt.figure(figsize=(10,10))

sns.distplot(X_train['workclass'],kde=False)

plt.show()
X_train["workclass"].value_counts()
plt.figure(figsize=(10,10))

sns.distplot(X_train['education.num'])

plt.show()
X_train["education.num"].value_counts()
plt.figure(figsize=(10,10))

sns.distplot(X_train['marital.status'])

plt.show()
X_train["marital.status"].value_counts()
plt.figure(figsize=(10,10))

sns.distplot(X_train['occupation'])

plt.show()
X_train["occupation"].value_counts()
plt.figure(figsize=(10,10))

sns.distplot(X_train['race'],kde=False)

plt.show()
X_train["race"].value_counts()
plt.figure(figsize=(10,10))

sns.distplot(X_train['sex'])

plt.show()
X_train["sex"].value_counts()
plt.figure(figsize=(10,10))

sns.distplot(X_train['capital.gain'],kde=False)

plt.show()
X_train["capital.gain"].value_counts()
plt.figure(figsize=(10,10))

sns.distplot(X_train['capital.loss'],kde=False)

plt.show()
X_train["capital.loss"].value_counts()
plt.figure(figsize=(10,10))

sns.distplot(X_train['hours.per.week'])

plt.show()
X_train["hours.per.week"].value_counts()
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import RobustScaler

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline



numerical_cols = list(X_train.select_dtypes(include=[np.number]).columns.values)



numerical_cols.remove('capital.gain')

numerical_cols.remove('capital.loss')



sparse_cols = ['capital.gain', 'capital.loss']



numerical_pipeline = Pipeline(steps = [

    ('scaler', StandardScaler())

])



sparse_pipeline = Pipeline(steps = [

    ('scaler', RobustScaler())

])



preprocessor = ColumnTransformer(transformers = [

    ('num', numerical_pipeline, numerical_cols),

    ('spr', sparse_pipeline, sparse_cols),

])



X_train = preprocessor.fit_transform(X_train)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score



knn = KNeighborsClassifier(n_neighbors=20)



score = cross_val_score(knn, X_train, Y_train, cv = 5, scoring="accuracy")

print("Acurácia com cross validation:", score.mean())
# Importando o Random Search com cross-validation

from sklearn.model_selection import RandomizedSearchCV



# Definindo o Random Search CV. Vamos fornecer o argumento n_iter, que fala quantas configurações de hparams testar:

random_search_cv = RandomizedSearchCV(estimator = KNeighborsClassifier(),

                              param_distributions = {'n_neighbors': range(1,50)}, # Testando comprimentos máximos de 1 a 50

                              scoring='accuracy', 

                              cv = 5,

                              n_iter = 12)



# Realizando a otimização por GridSearch para os dados de cancer de mama:

random_search_cv.fit(X_train,Y_train)



#Vamos ver informações relevantes:

print('Melhor número de vizinhos: {}'.format(random_search_cv.best_params_['n_neighbors']))

print('Melhor acurácia: {}'.format(round(random_search_cv.best_score_,5)))
knn = KNeighborsClassifier(n_neighbors=17)
knn.fit(X_train, Y_train)
df_t = pd.read_csv("../input/adult-pmr3508/test_data.csv", index_col=['Id'], na_values="?")

df_t.head()
X_test = df_t



X_test = X_test.drop(['fnlwgt','education','relationship','native.country'], axis=1)



X_test.head()
for column in ['workclass', 'occupation']:

  X_test[column].fillna(X_test[column].mode()[0], inplace=True)



for column in categorical_cols:

  X_test[column] = X_test[column].astype('category')



for column in categorical_cols:

  X_test[column] = X_test[column].cat.codes



X_test = preprocessor.fit_transform(X_test)
predictions = knn.predict(X_test)
submission = pd.DataFrame()
submission[0] = df_t.index

submission[1] = predictions

submission.columns = ['Id','income']



submission.head()
submission.to_csv('submission.csv',index = False)