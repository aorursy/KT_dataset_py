!wget -O gold.parquet https://www.dropbox.com/s/3m0xqogz5gi2moy/gold.parquet?dl=1
import pandas as pd
import numpy as np

work_dir = "/content"

#leitura dos dados de entrada
df_bruto = pd.read_parquet(work_dir +"/gold.parquet", engine="pyarrow")
df_bruto.shape

# não iremos utilizar estas duas colunas no treinamento e no teste
df_preparado = df_bruto.drop(["key", "timestamp"], axis=1)


# Substitui valores nulos por 0 nas colunas numéricas
colunas_numericas = ['pedidos_4_meses','pedidos_8_meses','pedidos_12_meses','itens_4_meses','itens_8_meses','itens_12_meses']
df_preparado[colunas_numericas] = df_preparado[colunas_numericas].fillna(value=0)

# transformar colunas categóricas em numéricas
df_preparado = pd.get_dummies(df_preparado, columns=["city", "state", "cnae_id"])
now = pd.to_datetime("now");
df_preparado['fez_ultima_compra'] = None

for idx, row in df_preparado.iterrows():
  if row['ultima_compra']== None and int(row['pedidos_4_meses']) == 0 and int(row['pedidos_8_meses']) == 0 and int(row['pedidos_12_meses']) == 0:
    df_preparado.loc[idx,'ultima_compra'] = 0
    df_preparado.loc[idx,'fez_ultima_compra'] = 0
  elif row['ultima_compra']!= None: #and int(row['pedidos_4_meses']) > 0.0 : #and int(row['pedidos_8_meses']) > 0 and int(row['pedidos_12_meses']) > 0:
    compra = pd.to_datetime(row['ultima_compra'], infer_datetime_format=True, format="%Y-%m-%d", errors='coerce')
    dif = now - compra
    df_preparado.loc[idx,'ultima_compra'] = int(dif/np.timedelta64(1, 'h'))
    df_preparado.loc[idx,'fez_ultima_compra'] = 1
  #else:
    #df_preparado.loc[idx,'ultima_compra'] = None
    #df_preparado.loc[idx,'fez_ultima_compra'] = 0

df_preparado['fez_ultima_compra'] = df_preparado['fez_ultima_compra'].fillna(value=0)
df_preparado['ultima_compra'] = df_preparado['ultima_compra'].fillna(value=999999999)

#df_preparado = pd.get_dummies(df_preparado, columns=["fez_ultima_compra", "ultima_compra"])

df_preparado.head(20)
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
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=1)

print(X_train.shape)
print(y_train.shape)
print(X_valid.shape)
print(y_valid.shape)
from sklearn.tree import DecisionTreeClassifier
from sklearn import linear_model, datasets
from sklearn import svm, datasets
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

#clf = DecisionTreeClassifier()
#clf = linear_model.LinearRegression()
#clf = svm.SVC(gamma = 0.001, C = 100)
#clf = KNeighborsClassifier(n_neighbors=6)
clf = linear_model.BayesianRidge()

#train model
#clf.fit(X,y)
clf.fit(X,y)

#predict
y_pred = clf.predict(X_valid)
#pd.DataFrame(clf.coef_).describe()
df_test = df_preparado[df_preparado["defaulting"].isnull()]
X_test = df_test.drop('defaulting', axis=1)
y_test = clf.predict(X_test)

output = df_test.assign(inadimplente=y_test)
output = output.loc[:, ['client_id','inadimplente']]
#output.describe()
from datetime import date
from google.colab import files
from datetime import datetime

taxa = 0.56
temp = output.copy()
for idx, row in temp.iterrows():
  if row['inadimplente'] >= taxa:
    temp.loc[idx,'inadimplente'] = int(1)
  else:
    temp.loc[idx,'inadimplente'] = int(0)

#out = 'ouput_sklearn_' + datetime.now().strftime("%m_%d_%Y") + '.csv'
#temp.to_csv(work_dir +"/"+out, index=False)
#files.download(out) 
y_pred_temp = np.array(y_pred, copy=True)
import sklearn.metrics as metrics
#taxa = 0.56 - RESJUSTES MANUAIS
for idx, item in enumerate(y_pred):
  if item >= taxa:
    y_pred[idx] = int(1)
  else:
    y_pred[idx] = int(0)

print("ROC AUC:",metrics.roc_auc_score(y_valid, y_pred))
print("Acurácia:",metrics.accuracy_score(y_valid, y_pred))
print("F1 score:",metrics.f1_score(y_valid, y_pred))

y_pred = np.array(y_pred_temp, copy=True)