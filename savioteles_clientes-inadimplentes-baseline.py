!wget -O gold.parquet https://www.dropbox.com/s/3m0xqogz5gi2moy/gold.parquet?dl=1
!ls /content
import pandas as pd

work_dir = "/content"

#leitura dos dados de entrada
df_bruto = pd.read_parquet(work_dir +"/gold.parquet", engine="pyarrow")
df_bruto.shape
df_bruto.info()
# não iremos utilizar estas duas colunas no treinamento e no teste
df_preparado = df_bruto.drop(['key', 'timestamp'], axis=1)

# iremos remover a feature ultima_compra, 
# mas fica como exercício você aproveitá-la no conjunto de features
df_preparado = df_preparado.drop(['ultima_compra'], axis=1)
df_bruto
# Substitui valores nulos por 0 nas colunas numéricas
colunas_numericas = ['pedidos_4_meses','pedidos_8_meses','pedidos_12_meses','itens_4_meses','itens_8_meses','itens_12_meses']
df_preparado[colunas_numericas] = df_preparado[colunas_numericas].fillna(value=0)

# transformar colunas categóricas em numéricas
df_preparado = pd.get_dummies(df_preparado, columns=["city", "state", "cnae_id"])
df_preparado.head()

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
df_to_train.defaulting.values
y
X_valid
from sklearn.tree import DecisionTreeClassifier

# Cria um classificador
clf = DecisionTreeClassifier()

# Treina a Árvore de Decisão
clf = clf.fit(X_train,y_train)

# Prediz a resposta para o dataset de validação
y_pred = clf.predict(X_valid)
y_valid.shape
y_pred.shape
import sklearn.metrics as metrics
print("ROC AUC:",metrics.roc_auc_score(y_valid, y_pred))
print("Acurácia:",metrics.accuracy_score(y_valid, y_pred))
print("F1 score:",metrics.f1_score(y_valid, y_pred))
df_test = df_preparado[df_preparado["defaulting"].isnull()]
df_test.shape
X_test = df_test.drop('defaulting', axis=1)
y_test = clf.predict(X_test)
y_test
output = df_test.assign(inadimplente=y_test)
output = output.loc[:, ['client_id','inadimplente']]
output.head()

output.to_csv(work_dir +"/ouput_sklearn.csv", index=False)
from google.colab import files
files.download('ouput_sklearn.csv') 