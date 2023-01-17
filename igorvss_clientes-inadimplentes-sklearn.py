!wget -O gold.parquet https://www.dropbox.com/s/3m0xqogz5gi2moy/gold.parquet?dl=1
!ls /content
import pandas as pd
import numpy as np

work_dir = "/content"

#leitura dos dados de entrada
df_bruto = pd.read_parquet(work_dir +"/gold.parquet", engine="pyarrow")
df_bruto.shape
# não iremos utilizar estas duas colunas no treinamento e no teste
df_preparado = df_bruto.drop(['key', 'timestamp'], axis=1)


#Não remover a data da ultima compra, usar como varialvel no dataframe

# iremos remover a feature ultima_compra, 
# mas fica como exercício você aproveitá-la no conjunto de features
#df_preparado = df_preparado.drop(['ultima_compra'], axis=1)
df_bruto
# Substitui valores nulos por 0 nas colunas numéricas
colunas_numericas = ['pedidos_4_meses','pedidos_8_meses','pedidos_12_meses','itens_4_meses','itens_8_meses','itens_12_meses']
df_preparado[colunas_numericas] = df_preparado[colunas_numericas].fillna(value=0)

# transformar colunas categóricas em numéricas
df_preparado = pd.get_dummies(df_preparado, columns=["city", "state", "cnae_id"])

#Os algoritimos de predição não conseguem trabalhar com datas no formato bruto
#então criei uma nova coluna para dizer se possui ou não ultima compra
df_preparado['realizou_ultima_compra'] = 0
for idx, row in df_preparado.iterrows():
  if row['ultima_compra'] == None:
    df_preparado.loc[idx,'realizou_ultima_compra'] = 0
  elif row['ultima_compra'] != None:
    df_preparado.loc[idx,'realizou_ultima_compra'] = 1

df_preparado['realizou_ultima_compra'] = df_preparado['realizou_ultima_compra'].fillna(value=0)
df_preparado = df_preparado.drop(['ultima_compra'], axis=1)
#Aumntei a amostra de teste para oferecer mais casos no treinamento do modelo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# seleciona as tuplas com rótulos
df_to_train = df_preparado[df_preparado["defaulting"].notnull()]

# remove a coluna defaulting dos dados de treinamento para não gerar overfiting
X = df_to_train.drop('defaulting', axis=1)

# Transforma a variável a predizer de boolean para inteiro
le = LabelEncoder()
y = le.fit_transform(df_to_train.defaulting.values)

# Divisão em conjunto de treinamento e validação
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=1)

print(X_train.shape)
print(y_train.shape)
print(X_valid.shape)
print(y_valid.shape)
df_to_train.shape
#Alterado ao algorítimo anterior para utlizar regressão linear
from sklearn import linear_model
clf = linear_model.LinearRegression()
clf.fit(X_train,y_train)

#predict
y_pred = clf.predict(X_valid)
pd.DataFrame(clf.coef_).describe()
df_test = df_preparado[df_preparado["defaulting"].isnull()]
df_test.shape
X_test = df_test.drop('defaulting', axis=1)
y_test = clf.predict(X_test)
y_test

#Os algoritimos de de regressão não retorna 0 ou 1 como resposta. Tem que ser interpretado
output = df_test.assign(inadimplente=y_test)
output = output.loc[:, ['client_id','inadimplente']]
output.describe()
#Defini como 0.53 o valor que devo considerar para classificar como inadimplentes
#Esse valor teve o melhor resultado apos rodar varioas vez o modelo para varios valores
#Comecei com valores da media e fui aumentando e diminuindo

for idx, row in output.iterrows():
  if row['inadimplente'] >= 0.53:
    output.loc[idx,'inadimplente'] = int(1)
  else:
    output.loc[idx,'inadimplente'] = int(0)


#resultado do y_valid comprova a melhora no modelo.

import sklearn.metrics as metrics

for idx, item in enumerate(y_pred):
  if item >= 0.53:
    y_pred[idx] = int(1)
  else:
    y_pred[idx] = int(0)

print("ROC AUC:",metrics.roc_auc_score(y_valid, y_pred))
print("Acurácia:",metrics.accuracy_score(y_valid, y_pred))
print("F1 score:",metrics.f1_score(y_valid, y_pred))
from datetime import date
from google.colab import files
from datetime import datetime

out = 'ouput_sklearn_' + datetime.now().strftime("%m_%d_%Y") + '.csv'
output.to_csv(work_dir +"/"+out, index=False)
files.download(out) 