import pandas as pd

import sklearn

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score

from sklearn import preprocessing

adult_df=pd.read_csv("../input/adult-pmr3508/train_data.csv", index_col=['Id'], na_values="?")

adult_df.columns = ["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Marital Status","Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss","Hours per week", "Country", "Target"]
adult_df
df_quantitativo = adult_df[['Age','fnlwgt','Education-Num','Capital Gain','Capital Loss','Hours per week','Target']].copy()

df_quantitativo_auxiliar = pd.DataFrame()

df_quantitativo_auxiliar['Max'] = df_quantitativo.max()

df_quantitativo_auxiliar['Min'] = df_quantitativo.min()

df_quantitativo_auxiliar['Media'] = df_quantitativo.mean()

df_quantitativo_auxiliar['Desvio Padrao'] = df_quantitativo.std()

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

df_quantitativo['Target']= label_encoder.fit_transform(df_quantitativo['Target'])



df_quantitativo_auxiliar

df_quantitativo.describe()
df_qualitativo = adult_df[['Workclass','Education','Marital Status','Occupation','Relationship','Race','Sex','Country']].copy()

for column in df_qualitativo.columns:

    print(df_qualitativo[column].drop_duplicates().values)
df_qualitativo.describe()
var_quant_corr=df_quantitativo.corr()

print(var_quant_corr['Target'].sort_values())

plt.figure(figsize=(10,8))

sns.heatmap(var_quant_corr, vmin=-1, vmax=1, annot=True, cmap="Blues")

plt.show()
plt.figure(figsize=(10,8))

plt.hist(x=df_quantitativo['Age'])

plt.show()

plt.figure(figsize=(10,8))

sns.boxplot(x='Target', y='Age', data=df_quantitativo, palette='Blues')

plt.title('Age')
plt.figure(figsize=(10,8))

plt.hist(x=df_quantitativo['fnlwgt'])

plt.show()

plt.figure(figsize=(10,8))

sns.boxplot(x='Target', y='fnlwgt', data=df_quantitativo, palette='Blues')

plt.title('fnlwgt')
plt.figure(figsize=(10,8))

plt.hist(x=df_quantitativo['Education-Num'])

plt.show()

plt.figure(figsize=(10,8))

sns.boxplot(x='Target', y='Education-Num', data=df_quantitativo, palette='Blues')

plt.title('Education-Num')
plt.figure(figsize=(10,8))

plt.hist(x=df_quantitativo['Capital Gain'])

plt.show()

plt.figure(figsize=(10,8))

sns.boxplot(x='Target', y='Capital Gain', data=df_quantitativo, palette='Blues')

plt.title('Capital Gain')
plt.figure(figsize=(10,8))

plt.hist(x=df_quantitativo['Capital Loss'])

plt.show()

plt.figure(figsize=(10,8))

sns.boxplot(x='Target', y='Capital Loss', data=df_quantitativo, palette='Blues')

plt.title('Capital Loss')
plt.figure(figsize=(10,8))

plt.hist(x=df_quantitativo['Hours per week'])

plt.show()

plt.figure(figsize=(10,8))

sns.boxplot(x='Target', y='Hours per week', data=df_quantitativo, palette='Blues')

plt.title('Hours per week')
plt.figure(figsize=(10,8))

sns.countplot(y='Workclass', data=df_qualitativo, palette="Greens_d")
plt.figure(figsize=(10,8))

sns.countplot(y='Education', data=df_qualitativo, palette="Greens_d")
plt.figure(figsize=(10,8))

sns.countplot(y='Marital Status', data=df_qualitativo, palette="Greens_d")
plt.figure(figsize=(10,8))

sns.countplot(y='Occupation', data=df_qualitativo, palette="Greens_d")
plt.figure(figsize=(10,8))

sns.countplot(y='Relationship', data=df_qualitativo, palette="Greens_d")
plt.figure(figsize=(10,8))

sns.countplot(y='Race', data=df_qualitativo, palette="Greens_d")
plt.figure(figsize=(10,8))

sns.countplot(y='Sex', data=df_qualitativo, palette="Greens_d")
plt.figure(figsize=(10,8))

sns.countplot(y='Country', data=df_qualitativo, palette="Greens_d")
adult_df.drop_duplicates(keep='first', inplace=True)

def fillna(col):

    col.fillna(col.value_counts().index[0], inplace=True)

    return col

df_adultna=adult_df.apply(lambda col:fillna(col))

df_adultna
df_adultna = df_adultna.drop(['fnlwgt',"Education","Marital Status","Country"],axis=1)

df_adultna
Y_test = df_adultna.pop('Target')

X_test = df_adultna
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder



#será usado no pré processamento dos dados qualitativos

pipeline_categ = Pipeline(steps = [

    ('onehot', OneHotEncoder(sparse=False))

])

from sklearn.preprocessing import StandardScaler

from sklearn.impute import KNNImputer

from sklearn.preprocessing import RobustScaler



#será usado no pré processamento dos dados numéricos não esparços, de forma a normalizar os dados

pipeline_num = Pipeline(steps = [

    ('scaler', StandardScaler())

])



#será usado para as colunas de ganhos e perdas de capital, dado que os dados são esparços

pipeline_robust = Pipeline(steps = [

    ('imputer', KNNImputer(n_neighbors=10, weights="uniform")),

    ('scaler', RobustScaler())

])
from sklearn.compose import ColumnTransformer



colunasNumericas = list(X_test.select_dtypes(include = [np.number]).columns.values)

colunasCategoricas = list(X_test.select_dtypes(exclude = [np.number]).columns.values)





#dados dispersos

colunasNumericas.remove('Capital Gain')

colunasNumericas.remove('Capital Loss')



preprocessador = ColumnTransformer(transformers = [

    ('numerico', pipeline_num, colunasNumericas),

    ('categorico', pipeline_categ, colunasCategoricas),

    ('robust', pipeline_robust, ['Capital Gain', 'Capital Loss'])

])





#realiza-se o pré processamento

X_test = preprocessador.fit_transform(X_test)

X_test
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

import time



############################################ Primeiro Loop ############################################################

best_accuracy = 0

best_n = 1



for n in [1,10,50,100,1000]:

    inicio = time.time()

    knn = KNeighborsClassifier(n_neighbors=n)

    scores = cross_val_score(knn, X_test, Y_test, cv=10)

    best_score=np.mean(scores)

    fim = time.time()

    if best_accuracy < best_score:

        best_accuracy = best_score

        best_n = n

        

    print("Tempo de execução para n={} : {} s".format(n,fim-inicio) )

    print("Melhor score para n = {}: {}".format(n,best_score))

    

print("Melhor acuracia do primeiro loop: {} \n Melhor n: {} ".format(best_accuracy,best_n))
######################################### Segundo Loop ########################################################################

best_accuracy = 0.86

best_n = 10



#for n in range(10,50): esse for eu executei no meu computador e o log pode ser visto abaixo, aqui no kaggle não tem condições de fazer desse jeito

for n in range(25,30):

    inicio = time.time()

    knn = KNeighborsClassifier(n_neighbors=n)

    scores = cross_val_score(knn, X_test, Y_test, cv=10)

    best_score=np.mean(scores)

    fim = time.time()

    if best_accuracy < best_score:

        best_accuracy = best_score

        best_n = n

        

    print("Tempo de execução para n={} : {} s".format(n,fim-inicio) )

    print("Melhor score para n = {}: {}".format(n,best_score))

    

print("Melhor acuracia do segundo loop: {} \n Melhor n: {} ".format(best_accuracy,best_n))

knn = KNeighborsClassifier(n_neighbors=26)

knn.fit(X_test, Y_test)
teste = pd.read_csv("../input/adult-pmr3508/test_data.csv", index_col=['Id'], na_values="?")

teste.columns = ["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Marital Status","Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss","Hours per week", "Country"]

teste_index = teste.index

#teste = teste.dropna()

def fillna(col):

    col.fillna(col.value_counts().index[0], inplace=True)

    return col

teste=teste.apply(lambda col:fillna(col))



X_teste_submissao = teste.drop(['fnlwgt', 'Country', 'Education','Marital Status'], axis=1)

X_teste_submissao



X_teste_submissao = preprocessador.fit_transform(X_teste_submissao)

predict = knn.predict(X_teste_submissao)



predict
submissao = pd.DataFrame()

submissao[0] = teste_index

submissao[1] = predict

submissao.columns = ['Id','income']

submissao.to_csv('submission.csv',index = False)