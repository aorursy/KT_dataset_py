# Import dos módulos
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import sys

from datetime import datetime

#graficos
import matplotlib.pyplot as plt
import seaborn as sns

#cross-validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

#modelos de ML
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC, NuSVC, LinearSVC

#metricas
import sklearn.metrics as metrics

#normalização
from sklearn.preprocessing import Normalizer

#dados de treino
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#resample
from sklearn.utils import resample

#
import pickle

#funções 
import funcoes as f
#!wget -O gold.parquet https://www.dropbox.com/s/3m0xqogz5gi2moy/gold.parquet?dl=1
#!ls /content
work_dir = "content"
work_dir2 = 'RESULTADO_ML'

#leitura dos dados de entrada
df_bruto = pd.read_parquet(work_dir +"/gold.parquet", engine="pyarrow")
df_bruto.shape
f.relatorio(df_bruto)
df_bruto.info()
df_bruto.head(3)
# não iremos utilizar estas duas colunas no treinamento e no teste
df_preparado = df_bruto.drop(['key', 'timestamp'], axis=1)

#criando o atributo dias_ultima_compra
df_preparado['dias_ultima_compra'] = df_preparado['ultima_compra'].apply(lambda x : f.diff_days(datetime.now().date(), x))

#excluindo o atributo ultima_compra
df_preparado = df_preparado.drop(['ultima_compra'], axis=1)
df_preparado.head(1)
df_preparado.groupby('cnae_id').size().sort_values(ascending = False).to_csv("cnae.csv")

#tratando valores nulos de cnae_id
df_preparado['cnae_id'] = df_preparado['cnae_id'].replace(np.nan,'0')
#há valores cnae_id que são identicos em numeros, mas diferem na máscara

df_preparado['cnae_id_mod']  = df_preparado['cnae_id'].map(f.somenteNumeros)
display(df_preparado.groupby('cnae_id').size().sort_values(ascending = False)       
        , df_preparado.groupby('cnae_id_mod').size().sort_values(ascending = False))
#retirando caracteres (/-.) de cnae_id

df_preparado['cnae_id']  = df_preparado['cnae_id'].map(f.somenteNumeros)


df_preparado = df_preparado.drop(['cnae_id_mod'], axis=1)
df_preparado.sort_values(by = 'cnae_id', ascending  = False).head(1)
df_preparado.groupby('cnae_id').size().sort_values(ascending = False)
# Substitui valores nulos por NÃO INFORMADO nas colunas categóricas
colunas_cat =['city', 'state']
df_preparado[colunas_cat] = df_preparado[colunas_cat].fillna(value='NÃO INFORMADO')
# Substitui valores nulos por 0 nas colunas numéricas
colunas_numericas = ['pedidos_4_meses','pedidos_8_meses','pedidos_12_meses','itens_4_meses','itens_8_meses','itens_12_meses']
df_preparado[colunas_numericas] = df_preparado[colunas_numericas].fillna(value=0)

f.relatorio(df_preparado)
# não iremos utilizar estas duas colunas no treinamento e no teste
df_preparado = df_preparado.drop(['city', 'state'], axis=1)
# transformar colunas categóricas em numéricas
#df_preparado = pd.get_dummies(df_preparado, columns=["city", "state", "cnae_id"])


df_preparado = pd.get_dummies(df_preparado, columns=["cnae_id"])
#df_preparado.head(1)
df_preparado.columns
#MELHORIA: AMPLIANDO OS DADOS DE TREINO
# na primeira tentativa, foi considerado como dados de treino todo o conjunto de registros,
# e dados de validação os 20% do conjunto de registros


## seleciona as tuplas com rótulos
#df_to_train = df_preparado[df_preparado["defaulting"].notnull()]
#
## remove a coluna defaulting dos dados de treinamento para não gerar overfiting
#X = df_to_train.drop('defaulting', axis=1)
#
## Transforma a variável a predizer de boolean para inteiro
#le = LabelEncoder()
#y = le.fit_transform(df_to_train.defaulting.values)
#
## Divisão em conjunto de treinamento e validação (0.2)
#X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=1)
#MELHORIA: AMPLIANDO OS DADOS DE TREINO

#X_train  = X
#y_train  = y

#print(X_train.shape)
#print(y_train.shape)
#print(X_valid.shape)
#print(y_valid.shape)
display("total: ", df_preparado.shape, "defaulting", df_preparado.groupby("defaulting").size())
# seleciona as tuplas com rótulos
df = df_preparado[df_preparado["defaulting"].notnull()]

# Transforma a variável a predizer de boolean para inteiro
le = LabelEncoder()
df['defaulting_int'] = le.fit_transform(df.defaulting.values)

# remove a coluna defaulting dos dados de treinamento para não gerar overfiting
df = df.drop('defaulting', axis=1)

#separando as classes
df_majoritario = df[df.defaulting_int ==0]
df_minoritario = df[df.defaulting_int ==1]


# Fazer Upsample classe minoritária
df_minotirario_upsampled = resample(df_minoritario,
                                 replace = True,     # sample with replacement
                                 n_samples = len(df_majoritario),    # to match majority class
                                 random_state = 123) # reproducible results

# Fazer undersample classe majoritária
df_majoritario_undersampled = resample(df_majoritario,
                                 replace=False,     # sample with replacement
                                 n_samples=len(df_minoritario),    # to match majority class
                                 random_state=123) # reproducible results

# Combinar as classes novamente (undersampled)
df_undersampled = pd.concat([df_minoritario, df_majoritario_undersampled])

# Combinar as classes novamente (upersampled)
df_upersampled = pd.concat([df_majoritario, df_minotirario_upsampled])


print(df_undersampled.shape)
print(df_upersampled.shape)

#pergundar qual tipo de sampler usar
#tipo_resample = input("Digite 1 para undersampler ou 2 para upersample")
tipo_resample = '2'
if (tipo_resample == '1'):
    df_resampled = df_undersampled
else:
    df_resampled = df_upersampled


##REPROGRAMANDO VARIAVEIS
X = df_resampled.drop('defaulting_int', axis=1)
Y = df_resampled['defaulting_int']

f.resumo(X, 5)
#MELHORIA: NORMALIZAÇÃO
#OBS.: foi implementado, em alguns cenários, a normalização dos dados, porém sem muito ganho de acurácia.

## Gerando os dados normalizados
#scaler = Normalizer().fit(X)
#normalizedX = scaler.transform(X)
#
## Sumarizando os dados transformados
#print("Dados Originais: \n\n", X.values)
#print("\nDados Normalizados: \n\n", normalizedX)
#X = normalizedX
#y = np.array(Y)

## Divisão em conjunto de treinamento e validação
#X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.2, random_state=1)
#
#
#print(X_train.shape)
#print(y_train.shape)
#print(X_valid.shape)
#print(y_valid.shape)
## Divisão em conjunto de treinamento e validação
X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.2, random_state=1)
# Preparando a lista de modelos

from sklearn.linear_model import LogisticRegression

modelos = []
modelos.append(('LinearDiscriminantAnalysis', LinearDiscriminantAnalysis()))
modelos.append(('Decision_Tree_Classifier', DecisionTreeClassifier()))

#ensemblers
modelos.append(('Random_Forest_Classifier', RandomForestClassifier()))
modelos.append(('AdaBoost_Classifier', AdaBoostClassifier()))

#nearest neighbors
modelos.append(('KNeighbors_Classifier', KNeighborsClassifier()))

#support vector classification
modelos.append(('SVC', SVC()))


modelos.append(('Logistic Regression', LogisticRegression()))

# Avaliando cada modelo em um loop
resultados = []
nomes = []

# Definindo os valores para o número de folds
num_folds = 10
seed = 7

for nome, modelo in modelos:
    nomes.append(nome)

    # cross validation
    kfold = KFold(n_splits = num_folds, random_state = seed)
    cv_results = cross_val_score(modelo, X_train, y_train, cv = kfold, scoring = 'accuracy')    
    #cv_results = cross_val_score(modelo, X, Y, cv = kfold, scoring = 'accuracy')    
    resultados.append(cv_results)
    msg = "Score cross-validation de %s: Média: %f, Desvio padrão: (%f)" % (nome, cv_results.mean(), cv_results.std())
    print(msg)   

    #Treinando o modelo
    modelo = modelo.fit(X_train,y_train) 

    # Salvando o modelo
    arquivo = work_dir2 +"/"+nome+".sav"
    pickle.dump(modelo, open(arquivo, 'wb'))
    msg2 = "Modelo "+nome+" salvo!"
    print(msg2)

#dados de teste

df_test = df_preparado[df_preparado["defaulting"].isnull()]
X_test = df_test.drop('defaulting', axis=1)
df_test.shape
df_test
for nome, modelo in modelos:
    arquivo = work_dir2 +"/"+nome+".sav"
  # Carregando o arquivo
    modelo_carregado = pickle.load(open(arquivo, 'rb'))
    msg3 = "Modelo "+nome+" carregado!"
    print(msg3)

  # Print do resultado
  # Fazendo previsões com os dados de validação
    y_pred = modelo.predict(X_valid)  
 
  # Resultado
    print("Métricas de "+nome)
    print("ROC AUC:",metrics.roc_auc_score(y_valid, y_pred)) 
    print("Acurácia:",metrics.accuracy_score(y_valid, y_pred)) 
    print("F1 score:",metrics.f1_score(y_valid, y_pred))
    print("************************************")
    
  # Fazendo previsões com os dados de teste
    y_test = modelo.predict(X_test)

  #Resultado final
    output = df_test.assign(inadimplente = y_test)
    output = output.loc[:, ['client_id','inadimplente']]

  #Salvando o resultado final
    output.to_csv(work_dir2 +"/"+nome+".csv", index=False)

#from google.colab import files

#for nome in nomes:
#    files.download(nome+".csv") 
