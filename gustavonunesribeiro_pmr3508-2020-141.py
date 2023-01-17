import pandas as pd

import numpy as np
import sklearn
import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv("../input/adult-pmr3508/train_data.csv",index_col=['Id'],na_values = '?')
train.head()
train.info()
train.describe()
sns.catplot(x="age", y="income", kind="violin", data=train, aspect=2)
sns.catplot(x = "fnlwgt",y = "income", kind = "violin", data = train, aspect = 2)
sns.catplot(x = "education.num",y = "income", kind = "violin", data = train, aspect = 2)
sns.catplot(x = "capital.gain",y = "income", kind = "violin", data = train, aspect = 2)
sns.catplot(x = "capital.loss",y = "income", kind = "violin", data = train, aspect = 2)
sns.catplot(x = "hours.per.week",y = "income", kind = "violin", data = train, aspect = 2)
sns.catplot( x = "workclass", hue = "income", kind = "count", data = train, height=7, aspect=1)

plt.xticks(rotation=30)
sns.catplot( x = "education", hue = "income", kind = "count", data = train, height=7, aspect=2)

plt.xticks(rotation=30)
sns.catplot( x = "marital.status", hue = "income", kind = "count", data = train, height=7, aspect=1)

plt.xticks(rotation=30)
sns.catplot( x = "occupation", hue = "income", kind = "count", data = train, height=7, aspect=2)

plt.xticks(rotation=30)
sns.catplot( x = "race", hue = "income", kind = "count", data = train, height=7, aspect=2)

plt.xticks(rotation=30)
sns.catplot( x = "sex", hue = "income", kind = "count", data = train, height=7, aspect=2)

plt.xticks(rotation=30)
sns.catplot( x = "relationship", hue = "income", kind = "count", data = train, height=7, aspect=2)

plt.xticks(rotation=30)
train["native.country"].value_counts()
train_corr = train.copy()
from sklearn.preprocessing import LabelEncoder



# Instanciando o LabelEncoder

le = LabelEncoder()



# Modificando o nosso dataframe, transformando a variável de classe em 0s e 1s

train_corr['income'] = le.fit_transform(train_corr['income'])
plt.figure(figsize=(10,10))



sns.heatmap(train_corr.corr(), square = True, annot=True, vmin=-0.5, vmax=1, cmap='cool')

plt.show()
train.drop_duplicates(keep='first', inplace=True)
train = train.drop(['fnlwgt', 'native.country','education'], axis = 1)
train.head()
y_train = train.pop("income")

x_train = train
x_train.head()
numerical_cols = list(x_train.select_dtypes(include=[np.number]).columns.values)



numerical_cols.remove('capital.gain')

numerical_cols.remove('capital.loss')



sparse_cols = ['capital.gain','capital.loss']



categorical_cols = list(x_train.select_dtypes(exclude=[np.number]).columns.values)



print("Colunas numéricas: ", numerical_cols)

print("Colunas esparsas: ", sparse_cols)

print("Colunas categóricas: ", categorical_cols)
from sklearn.impute import SimpleImputer



simple_imputer = SimpleImputer(strategy='most_frequent')
from sklearn.preprocessing import OneHotEncoder



one_hot = OneHotEncoder(sparse=False)
from sklearn.pipeline import Pipeline
categorical_pipeline = Pipeline(steps = [

    ('imputer', SimpleImputer(strategy = 'most_frequent')),

    ('onehot', OneHotEncoder(drop='if_binary'))])
from sklearn.preprocessing import StandardScaler



numerical_pipeline = Pipeline(steps = [

    ('scaler', StandardScaler())

])
from sklearn.preprocessing import RobustScaler



sparse_pipeline = Pipeline(steps = [

    ('scaler', RobustScaler())

])
from sklearn.compose import ColumnTransformer



# Cria o nosso Pré-Processador



# Cada pipeline está associada a suas respectivas colunas no dataset

preprocessor = ColumnTransformer(transformers = [

    ('num', numerical_pipeline, numerical_cols),

    ('spr', sparse_pipeline, sparse_cols),

    ('cat', categorical_pipeline, categorical_cols)

])
x_train = preprocessor.fit_transform(x_train)
from sklearn.neighbors import KNeighborsClassifier



from sklearn.model_selection import cross_val_score



knn = KNeighborsClassifier(n_neighbors=10)



score = cross_val_score(knn, x_train, y_train, cv = 5, scoring="accuracy")

print("Acurácia com cross validation:", score.mean())
# Quantidades de vizinhos a serem testadas

neighbors = [15, 20, 25]



# Dicionário para guardar as pontuações de cada hiperparâmetro

neighbors_scores = {}



for n_neighbors in neighbors:

    # Calcula a média de acurácia de cada classificador

    score = cross_val_score(KNeighborsClassifier(n_neighbors=n_neighbors), x_train, y_train, cv = 5, scoring="accuracy").mean()

    

    # Guarda essa acurácia

    neighbors_scores[n_neighbors] = score



# Obtém a quantidade de vizinhos com o melhor desempenho

best_n = max(neighbors_scores, key=neighbors_scores.get)



print("Melhor hiperparâmetro: ", best_n)

print("Melhor acurácia: ", neighbors_scores[best_n])
knn = KNeighborsClassifier(n_neighbors=20)



knn.fit(x_train, y_train)
test_data = pd.read_csv('../input/adult-pmr3508/test_data.csv',index_col=['Id'], na_values="?")

test_data.head()
x_test = test_data.drop(['fnlwgt','education','native.country'], axis=1)



x_test = preprocessor.transform(x_test)



predictions = knn.predict(x_test)
predictions
submission = pd.DataFrame ()
submission[0] = test_data.index

submission[1] = predictions

submission.columns = ['Id','income']
submission.head()
submission.to_csv('submission.csv',index = False)