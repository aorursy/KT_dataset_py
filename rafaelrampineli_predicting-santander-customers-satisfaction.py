# Importando as bibliotecas necessárias

# Biblioteca para manipulação de Dados
import pandas as pd
# Biblioteca para plot de gráficos
import matplotlib.pyplot as plt
%matplotlib inline
# 
import numpy as np

# Utilizado para realizar o balanceamento da classificação (TARGET) dos dados para evitar resultado tendêncioso
#!pip install imblearn
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")
# Carregando o dataset utilizando o Pandas
dataset_train = pd.read_csv("/kaggle/input/santander-customers/Santander_Customers.csv", sep = ",")
# Visualizando os primeiros registros do dataset_train
dataset_train.head(5)
dataset_train.info()
pd.set_option('display.max_columns', None)  
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)
dataset_train.describe()
def CleanData(dataset):
    # Removendo colunas que possuem somentes valores = 0 (irrelevantes para o modelo)
    dataset = dataset.loc[:, (dataset !=0).any(axis=0)]
    
    # Variável var3 apresenta um valor Min muito disperso com base no describe acima podendo representar valores unknown. 
    # Realizando o ajuste da variável aplicando o valor encontrado na mediana.
    dataset = dataset.replace({'var3': -999999}, 2)
    
    return dataset
dataset_train = CleanData(dataset_train)
dataset_train.describe()
# Verificando se existem valores NULL
dataset_train.isnull().values.any()
# Separando a coluna ID do dataset e controlando em outro dataset.
dataset_train_ID = dataset_train[['ID']]
dataset_train = dataset_train.drop(['ID'], axis=1)
dataset_train['TARGET'].value_counts().plot(kind = 'bar', figsize=(6,6))
plt.title('Analisando os dados TARGET')
plt.xlabel('Target')
plt.ylabel('Quantidade Registros')
plt.show()
dataset_train.groupby('TARGET').size()
columns = dataset_train.columns

for i in columns:
    dataset_train[i].plot(kind = 'hist')
    plt.title('Histograma Analisando os dados: ' + i)
    plt.xlabel(i)
    plt.ylabel('Distribuição')
    
    plt.show()
    
columns = dataset_train.columns

for i in columns:
    dataset_train[i].plot(kind = 'box')
    plt.title('BoxPlots Analisando os dados: ' + i)
    plt.xlabel(i)
    plt.ylabel('Escala')
    
    plt.show()
# Criando uma Matrix de Correlação para identificar a correlação entre as variáveis
correlations = dataset_train.corr()

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin = -1, vmax = 1) #Mostrar as correlações
fig.colorbar(cax)
plt.show()
dataset_train.corr()['TARGET']
def NormalizingData(dataset):
    normalized_vars = (dataset.drop(['TARGET'], axis=1) - dataset.drop(['TARGET'], axis=1).mean())/dataset.drop(['TARGET'], axis=1).std()

    df_normalized = pd.concat([ dataset[['TARGET']], normalized_vars], axis=1)
    
    return df_normalized
dataset_train_Normalized = NormalizingData(dataset_train)
dataset_train_Normalized.head(5)
def fun_BalancingData(dataset, var_target):
    
    # Split da variavel target e variaveis preditoras
    x_train = dataset.drop([var_target], axis=1)
    y_train = dataset[var_target]
    
    smt = SMOTE()
    
    # Separando os dados em features e target
    features_train, target_train = smt.fit_sample(x_train, y_train)
    
    target_train_DF = pd.DataFrame(target_train)
    # atribuindo o nome da coluna
    target_train_DF.columns = ['TARGET']
    
    features_train_DF = pd.DataFrame(features_train)
    features_train_DF.columns = x_train.columns
    
    return pd.concat( [target_train_DF, features_train_DF], axis=1 )
dataset_train_Balanced = fun_BalancingData(dataset_train_Normalized, 'TARGET')
from sklearn.ensemble import ExtraTreesClassifier

# Atribuindo a quantidade de variáveis que desejamos selecionar, para reaproveitamento do valor em códigos futuros
n_varsImportants = 45

array_dataset = dataset_train_Balanced.values

features_x = array_dataset[:,1:] # Seleciona todas as variáveis preditoras
target_y = array_dataset[:,0:1] # Seleciona somente a TARGET

model = ExtraTreesClassifier()
seletor = model.fit(features_x,target_y.ravel())

# Trasnsformando em valor serial para plotar no gráfico
features_important = pd.Series(model.feature_importances_, index= dataset_train.drop(['TARGET'], axis=1).columns.values).sort_values(ascending=False)

features_important[:n_varsImportants].plot(kind='bar',title='Top 45 Features Most Important by ExtraTreesClassifier',figsize=(12,8))
plt.show()
# Função utilizada para aplicar a seleção das N variáveis mais relevantes de acordo com o ExtraTreesClassifier
def Select_Top_Features(DF, IndexColNames):
    DF_feature = DF.loc[:, DF.columns.isin(IndexColNames)]
    
    return pd.concat( [DF['TARGET'], DF_feature], axis=1 )
Columns_Feature_Select = features_important[:n_varsImportants].index

dataset_train_feature_selection = Select_Top_Features(dataset_train_Balanced, Columns_Feature_Select)
dataset_train_feature_selection.describe()
!pip install pyspark
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql.functions import *

from pyspark.ml.linalg import Vectors
from pyspark.sql import Row
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import PCA
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Spark Session - usada quando se trabalha com Dataframes no Spark
spSession = SparkSession.builder.master("local").appName("Santander-Customers").getOrCreate()

sqlContext = SQLContext(spSession)

DF = sqlContext.createDataFrame(dataset_train_feature_selection)
# Criando um LabeledPoint (target, Vector[features])
def transformaVar(row):
    listvalues = []
    # Seleciona todas as variáveis preditoras exceto a TARGET
    for i in row[1:]:
        listvalues.append(i)

    obj = (row["TARGET"], Vectors.dense(listvalues))
    return obj
# Transforma o Dataframe do Spark em RDD e aplica a função map
RDDTransf = DF.rdd.map(transformaVar)
RDDTransf.collect()
# Transforma o RDD de volta a um DataFrame do Spark
DF2 = spSession.createDataFrame(RDDTransf, ["label", "features"])
type(DF2)
DF2.select("features","label").show(5)
# Aplicando a redução de Dimensionalidade com PCA
# Embora algumas etapas acima eu tenha selecionado as top 45 variáveis mais relevantes com base no resultado do ExtraTreesClassifier
# Estou aplicando o PCA para reduzir as 45 variáveis em componentes para aplicação do conceito e prática operacional.
pca = PCA(k = 5, inputCol = "features", outputCol = "pcaFeatures")

# Criando o modelo
pcaModel = pca.fit(DF2)
pcaResult = pcaModel.transform(DF2).select("label","features")
pcaResult.show(truncate = False)
# Indexação é pré-requisito para Decision Trees com apache Spark
# RandomForest é um algoritmo de Decision Tree

# Utiliza a coluna "label" para criar uma nova coluna Indexada
stringIndexer = StringIndexer(inputCol = "label", outputCol = "indexed")
si_model = stringIndexer.fit(pcaResult)
FinalResult = si_model.transform(pcaResult)
FinalResult.collect()
# Split dados-Treino e Dados-Teste
(df_treino, df_teste) = FinalResult.randomSplit([0.7, 0.3]) # 70% Treino 30% Teste
df_treino.count()
df_teste.count()
# Criando o modelo utilizando RandomForest
# O parâmetro "featurescol" espera um dado do tipo vetor, por esse motivo na etapa de pré-processamento foi gerado um densevector.
rfClassifer = RandomForestClassifier(labelCol = "indexed", featuresCol = "features")
modelo = rfClassifer.fit(df_treino)
# Previsões com dados de teste
predictions = modelo.transform(df_teste)

# Selecionando os campos necessários
# prediction: Previsão feita pelo modelo
# indexed e label são os target (dado que o label possui apenas 2 valores [0 e 1])
# Features representa os componentes utilizados durante a criação do modelo
predictions.select("prediction", "indexed", "label", "features").collect()
# Avaliando a acurácia/assertividade do modelo criado.

# Informa que a Coluna "prediction" representa a previsão feita pelo modelo,
# que a coluna "indexed" representa a coluna target que queriamos prever,
evaluator = MulticlassClassificationEvaluator(predictionCol = "prediction", labelCol = "indexed", metricName = "accuracy")
# Aplica a classificação criada ao conjunto de dados predictions
evaluator.evaluate(predictions) 

# Existem várias formas de melhorar a acurácia do modelo criado: Ajustes de # componentes dos PCA, Incluir ou remover variáveis... 
# Gerando uma Confusion Matrix para avaliar as classificações corretas e incorretas.
predictions.groupBy("indexed", "prediction").count().show()