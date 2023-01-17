import os

print(os.listdir("../input"))
!pip install pyspark
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('Competencia_DeteccionTrafico').getOrCreate()
from pyspark.sql.functions import *





concurso_train_df = spark.read.format('csv').option("header", "false").load("/kaggle/input/competencia-deteccion/Train.txt")

concurso_test_df = spark.read.format('csv').option("header", "false").load("/kaggle/input/competencia-deteccion/Test.txt")

concurso_train_df.toPandas().head(2)
print("Número de registros del conjunto de entrenamiento : ",concurso_train_df.count())

print("Número de registros del conjunto de pruebas : ",concurso_test_df.count())
concurso_train_df = concurso_train_df.withColumn('_c41', regexp_replace('_c41', '^(?!normal).*', 'ataque'))

concurso_train_df.toPandas().head(2)

print("numero de categorias : ",concurso_train_df.select("_c41").distinct().count())
cols = concurso_train_df.columns



from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler

categoricalColumns = ["_c0","_c1","_c2","_c3","_c4","_c5","_c6","_c7","_c8","_c9","_c10",\

    "_c11","_c12","_c13","_c14","_c15","_c16","_c17","_c18","_c19","_c20",\

    "_c21","_c22","_c23","_c24","_c25","_c26","_c27","_c28","_c29","_c30",\

    "_c31","_c32","_c33","_c34","_c35","_c36","_c37","_c38","_c39","_c40",\

    "_c42"]

stages = []



for categoricalCol in categoricalColumns:

    stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index', stringOrderType = 'alphabetAsc')

    encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"],handleInvalid='keep',dropLast=True)

    stages += [stringIndexer, encoder]

label_stringIdx = StringIndexer(inputCol = '_c41', outputCol = 'label')

stages += [label_stringIdx]

numericCols = []

assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols

assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")

stages += [assembler]
from pyspark.ml import Pipeline

pipeline = Pipeline(stages = stages)

pipelineModel = pipeline.fit(concurso_train_df)

concurso_train_df = pipelineModel.transform(concurso_train_df)

selectedCols = ['label', 'features'] + cols

concurso_train_df = concurso_train_df.select(selectedCols)

concurso_train_df.toPandas().head(3)
concurso_train_df = concurso_train_df.withColumnRenamed("_c0","duration")

concurso_train_df = concurso_train_df.withColumnRenamed("_c1","protocol_type")

concurso_train_df = concurso_train_df.withColumnRenamed("_c2","service")

concurso_train_df = concurso_train_df.withColumnRenamed("_c3","flag")

concurso_train_df = concurso_train_df.withColumnRenamed("_c4","src_bytes")

concurso_train_df = concurso_train_df.withColumnRenamed("_c5","dst_bytes")

concurso_train_df = concurso_train_df.withColumnRenamed("_c6","land")

concurso_train_df = concurso_train_df.withColumnRenamed("_c7","wrong_fragment")

concurso_train_df = concurso_train_df.withColumnRenamed("_c8","urgent")

concurso_train_df = concurso_train_df.withColumnRenamed("_c9","hot")

concurso_train_df = concurso_train_df.withColumnRenamed("_c10","num_failed_logins")

concurso_train_df = concurso_train_df.withColumnRenamed("_c11","logged_in")

concurso_train_df = concurso_train_df.withColumnRenamed("_c12","num_compromised")

concurso_train_df = concurso_train_df.withColumnRenamed("_c13","root_shell")

concurso_train_df = concurso_train_df.withColumnRenamed("_c14","su_attempted")

concurso_train_df = concurso_train_df.withColumnRenamed("_c15","num_root")

concurso_train_df = concurso_train_df.withColumnRenamed("_c16","num_file_creations")

concurso_train_df = concurso_train_df.withColumnRenamed("_c17","num_shells")

concurso_train_df = concurso_train_df.withColumnRenamed("_c18","num_access_files")

concurso_train_df = concurso_train_df.withColumnRenamed("_c19","num_outbound_cmds")

concurso_train_df = concurso_train_df.withColumnRenamed("_c20","is_host_login")

concurso_train_df = concurso_train_df.withColumnRenamed("_c21","is_guest_login")

concurso_train_df = concurso_train_df.withColumnRenamed("_c22","count")

concurso_train_df = concurso_train_df.withColumnRenamed("_c23","srv_count")

concurso_train_df = concurso_train_df.withColumnRenamed("_c24","serror_rate")

concurso_train_df = concurso_train_df.withColumnRenamed("_c25","srv_serror_rate")

concurso_train_df = concurso_train_df.withColumnRenamed("_c26","rerror_rate")

concurso_train_df = concurso_train_df.withColumnRenamed("_c27","srv_rerror_rate")

concurso_train_df = concurso_train_df.withColumnRenamed("_c28","same_srv_rate")

concurso_train_df = concurso_train_df.withColumnRenamed("_c29","diff_srv_rate")

concurso_train_df = concurso_train_df.withColumnRenamed("_c30","srv_diff_host_rate")

concurso_train_df = concurso_train_df.withColumnRenamed("_c31","dst_host_count")

concurso_train_df = concurso_train_df.withColumnRenamed("_c32","dst_host_srv_count")

concurso_train_df = concurso_train_df.withColumnRenamed("_c33","dst_host_same_srv_rate")

concurso_train_df = concurso_train_df.withColumnRenamed("_c34","dst_host_diff_srv_rate")

concurso_train_df = concurso_train_df.withColumnRenamed("_c35","dst_host_same_src_port_rate")

concurso_train_df = concurso_train_df.withColumnRenamed("_c36","dst_host_srv_diff_host_rate")

concurso_train_df = concurso_train_df.withColumnRenamed("_c37","dst_host_serror_rate")

concurso_train_df = concurso_train_df.withColumnRenamed("_c38","dst_host_srv_serror_rate")

concurso_train_df = concurso_train_df.withColumnRenamed("_c39","dst_host_rerror_rate")

concurso_train_df = concurso_train_df.withColumnRenamed("_c40","dst_host_srv_rerror_rate")

concurso_train_df = concurso_train_df.withColumnRenamed("_c41","attack")

concurso_train_df = concurso_train_df.withColumnRenamed("_c42","last_flag")
train, test = concurso_train_df.randomSplit([0.7, 0.3], seed = 2018)

print("Training Dataset Count: " + str(train.count()))

print("Test Dataset Count: " + str(test.count()))
from pyspark.ml.classification import LogisticRegression



lr = LogisticRegression(featuresCol = 'features', labelCol = 'label', maxIter=10)

modelo_ml_deteccion = lr.fit(train)
predictions = modelo_ml_deteccion.transform(test)

predictions.toPandas().head(2)
predictions.groupBy("prediction").count()

print("Tráfico clasificado como ataque :" + " " + str(predictions.filter(predictions.prediction == "0.0").count()))

print("Tráfico clasificado como normal :" + " " + str(predictions.filter(predictions.prediction == "1.0").count()))



accuracy = predictions.filter(predictions.label == predictions.prediction).count() / float(predictions.count())

print("Precisión del modelo", accuracy)
from pyspark.ml.feature import IndexToString



converter = IndexToString(inputCol="prediction", outputCol="prediccionLiteral", labels=["normal","ataque"])

predictions_literal = converter.transform(predictions)

predictions_literal.toPandas().head(5)