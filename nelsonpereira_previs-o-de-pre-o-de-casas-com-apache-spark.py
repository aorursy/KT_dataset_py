!pip install pyspark
#Importar as Bibliotecas necessárias

import os

import pandas as pd

import numpy as np



from pyspark import SparkConf, SparkContext

from pyspark.sql import SparkSession, SQLContext



from pyspark.sql.types import *

import pyspark.sql.functions as F

from pyspark.sql.functions import udf, col



from pyspark.ml.regression import LinearRegression

from pyspark.mllib.evaluation import RegressionMetrics



from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, CrossValidatorModel

from pyspark.ml.feature import VectorAssembler, StandardScaler

from pyspark.ml.evaluation import RegressionEvaluator
import seaborn as sns

import matplotlib.pyplot as plt
#Visualização

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"



pd.set_option('display.max_columns', 200)

pd.set_option('display.max_colwidth', 400)



from matplotlib import rcParams

sns.set(context='notebook', style='whitegrid', rc={'figure.figsize': (18,4)})

rcParams['figure.figsize'] = 18,4



%matplotlib inline

%config InlineBackend.figure_format = 'retina'
# definindo semente aleatória para reprodutibilidade do notebook

rnd_seed=23

np.random.seed=rnd_seed

np.random.set_state=rnd_seed
spark = SparkSession.builder.master("local[2]").appName("Regressao-Linear-Casas-California").getOrCreate()
spark
sc = spark.sparkContext

sc
sqlContext = SQLContext(spark.sparkContext)

sqlContext
CASAS_DATA = '../input/cal_housing.data'
# define o esquema, correspondente a uma linha no arquivo de dados csv.

schema = StructType([

    StructField("long", FloatType(), nullable=True),

    StructField("lat", FloatType(), nullable=True),

    StructField("medage", FloatType(), nullable=True),

    StructField("totrooms", FloatType(), nullable=True),

    StructField("totbdrms", FloatType(), nullable=True),

    StructField("pop", FloatType(), nullable=True),

    StructField("houshlds", FloatType(), nullable=True),

    StructField("medinc", FloatType(), nullable=True),

    StructField("medhv", FloatType(), nullable=True)]

)

# Carregar dados das casas

casas_df = spark.read.csv(path=CASAS_DATA, schema=schema).cache()
# Inspecionando as 5 primeiras linhas

casas_df.take(5)
# Mostrar as primeiras cinco linhas

casas_df.show(5)
# mostra as colunas do Data Frame

casas_df.columns
# mostra o esquema do Data Frame

casas_df.printSchema()
# executar uma seleção de amostra

casas_df.select('pop','totbdrms').show(10)
# Agrupar por housingmedianage e ver a distribuição

resultado_df = casas_df.groupBy("medage").count().sort("medage", ascending=False)
resultado_df.show(10)
resultado_df.toPandas().plot.bar(x='medage', figsize=(14,6))
(casas_df.describe().select(

                    "summary",

                    F.round("medage", 4).alias("medage"),

                    F.round("totrooms", 4).alias("totrooms"),

                    F.round("totbdrms", 4).alias("totbdrms"),

                    F.round("pop", 4).alias("pop"),

                    F.round("houshlds", 4).alias("houshlds"),

                    F.round("medinc", 4).alias("medinc"),

                    F.round("medhv", 4).alias("medhv"))

                    .show())
# Ajustando os valores de `medianHouseValue`

casas_df = casas_df.withColumn("medhv", col("medhv")/100000)
# Exibir as 2 primeiras linhas de 'df'

casas_df.show(2)
casas_df.columns
# Adicionando novas colunas ao data frame 'df'

casas_df = (casas_df.withColumn("rmsperhh", F.round(col("totrooms")/col("houshlds"), 2))

                       .withColumn("popperhh", F.round(col("pop")/col("houshlds"), 2))

                       .withColumn("bdrmsperrm", F.round(col("totbdrms")/col("totrooms"), 2)))
# Inspecionando o resultado

casas_df.show(5)
# Reordenação e seleção das colunas

casas_df = casas_df.select("medhv", 

                              "totbdrms", 

                              "pop", 

                              "houshlds", 

                              "medinc", 

                              "rmsperhh", 

                              "popperhh", 

                              "bdrmsperrm")
featureCols = ["totbdrms", "pop", "houshlds", "medinc", "rmsperhh", "popperhh", "bdrmsperrm"]
# coloca recursos em uma coluna de vetor de recursos

assembler = VectorAssembler(inputCols=featureCols, outputCol="features") 
assembled_df = assembler.transform(casas_df)
assembled_df.show(10, truncate=False)
# Inicialize o `standardScaler`

standardScaler = StandardScaler(inputCol="features", outputCol="features_escaladas")
# Ajuste o DataFrame ao redimensionador

escalar_df = standardScaler.fit(assembled_df).transform(assembled_df)
# Inspecionando o resultado

escalar_df.select("features", "features_escaladas").show(10, truncate=False)
# Divida os dados em conjuntos de treino e teste

treino_data, teste_data = escalar_df.randomSplit([.8,.2], seed=rnd_seed)
treino_data.columns
#Inicializar 'lr'

lr = (LinearRegression(featuresCol='features_escaladas', labelCol="medhv", predictionCol='predmedhv', 

                               maxIter=10, regParam=0.3, elasticNetParam=0.8, standardization=False))
# ajustando os dados ao modelo

linearModel = lr.fit(treino_data)
# Coeficientes para o modelo

linearModel.coefficients
featureCols
# Interceptação para o modelo

linearModel.intercept
coeff_df = pd.DataFrame({"Feature": ["Intercept"] + featureCols, "Co-efficients": np.insert(linearModel.coefficients.toArray(), 0, linearModel.intercept)})

coeff_df = coeff_df[["Feature", "Co-efficients"]]
coeff_df
# geração de previsões

predictions = linearModel.transform(teste_data)
# Extração das previsões e os rótulos corretos "conhecidos"

predandlabels = predictions.select("predmedhv", "medhv")
predandlabels.show()
# Obter o RMSE

print("RMSE: {0}".format(linearModel.summary.rootMeanSquaredError))
print("MAE: {0}".format(linearModel.summary.meanAbsoluteError))
# Obter o R2

print("R2: {0}".format(linearModel.summary.r2))
evaluator = RegressionEvaluator(predictionCol="predmedhv", labelCol='medhv', metricName='rmse')

print("RMSE: {0}".format(evaluator.evaluate(predandlabels)))
evaluator = RegressionEvaluator(predictionCol="predmedhv", labelCol='medhv', metricName='mae')

print("MAE: {0}".format(evaluator.evaluate(predandlabels)))
evaluator = RegressionEvaluator(predictionCol="predmedhv", labelCol='medhv', metricName='r2')

print("R2: {0}".format(evaluator.evaluate(predandlabels)))
# mllib é antigo, portanto os métodos estão disponíveis no rdd

metrics = RegressionMetrics(predandlabels.rdd)
print("RMSE: {0}".format(metrics.rootMeanSquaredError))
print("MAE: {0}".format(metrics.meanAbsoluteError))
print("R2: {0}".format(metrics.r2))
# Para o serviço do PySpark

spark.stop()