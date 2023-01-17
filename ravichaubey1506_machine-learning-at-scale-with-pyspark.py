!pip install pyspark
#Initializing PySpark

from pyspark import SparkContext, SparkConf



#Spark Config

conf = SparkConf().setAppName("sample_app")

sc = SparkContext(conf=conf)
from pyspark.sql import SQLContext

from pyspark.sql import DataFrameNaFunctions



from pyspark.ml import Pipeline

from pyspark.ml.classification import DecisionTreeClassifier



from pyspark.ml.feature import Binarizer

from pyspark.ml.feature import VectorAssembler, StringIndexer, VectorIndexer
sqlContext = SQLContext(sc)

df = sqlContext.read.load('../input/san-diego-daily-weather-data/daily_weather.csv', 

                          format='com.databricks.spark.csv', 

                          header='true',inferSchema='true')
print(df.columns)
featureColumns = ['air_pressure_9am','air_temp_9am','avg_wind_direction_9am','avg_wind_speed_9am',

        'max_wind_direction_9am','max_wind_speed_9am','rain_accumulation_9am',

        'rain_duration_9am']
df = df.drop('number')

df = df.na.drop()
print(df.count(),",",len(df.columns))
binarizer = Binarizer(threshold=24.99999,

                     inputCol = "relative_humidity_3pm",

                     outputCol = "label")



binarizedDF = binarizer.transform(df)
binarizedDF.select('relative_humidity_3pm','label').show(5)
binarizedDF.toPandas().head(2)
assembler = VectorAssembler(inputCols=featureColumns,

                           outputCol = 'features')



assembled = assembler.transform(binarizedDF)
assembled.select('features').show(2)
(trainingData, testData) = assembled.randomSplit([0.8,0.2], seed=13234)
trainingData.count(),testData.count()
dt = DecisionTreeClassifier(labelCol='label',featuresCol='features',maxDepth=5,

                           minInstancesPerNode = 20, impurity = 'gini')
pipeline = Pipeline(stages=[dt])

model = pipeline.fit(trainingData)
predictions = model.transform(testData)
predictions.select('prediction','label').show(5)
predictions.select('prediction','label').write.save('low_humidity_prediction.csv', 

                          format='com.databricks.spark.csv', 

                          header='true',inferSchema='true')
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from pyspark.mllib.evaluation import MulticlassMetrics
predictions = sqlContext.read.load('./low_humidity_prediction.csv', 

                          format='com.databricks.spark.csv', 

                          header='true',inferSchema='true')
predictions.show(5)
evaluator = MulticlassClassificationEvaluator(labelCol='label',

                                             predictionCol = 'prediction',

                                             metricName = 'accuracy')
acc = evaluator.evaluate(predictions)

print(acc)
predictions.rdd.take(2)
predictions.rdd.map(tuple).take(5)
metrics = MulticlassMetrics(predictions.rdd.map(tuple))


metrics.confusionMatrix().toArray().T
from pyspark.sql import SQLContext

from pyspark.ml.clustering import KMeans

from pyspark.ml.feature import VectorAssembler

from pyspark.ml.feature import StandardScaler

%matplotlib inline
from itertools import cycle, islice

from math import sqrt

from numpy import array

from pandas.plotting import parallel_coordinates

from pyspark.ml.clustering import KMeans as KM

from pyspark.mllib.linalg import DenseVector

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd



def computeCost(featuresAndPrediction, model):

    allClusterCenters = [DenseVector(c) for c in model.clusterCenters()]

    arrayCollection   = featuresAndPrediction.rdd.map(array)



    def error(point, predictedCluster):

        center = allClusterCenters[predictedCluster]

        z      = point - center

        return sqrt((z*z).sum())

    

    return arrayCollection.map(lambda row: error(row[0], row[1])).reduce(lambda x, y: x + y)





def elbow(elbowset, clusters):

    wsseList = []	

    for k in clusters:

        print("Training for cluster size {} ".format(k))

        kmeans = KM(k = k, seed = 1)

        model = kmeans.fit(elbowset)

        transformed = model.transform(elbowset)

        featuresAndPrediction = transformed.select("features", "prediction")



        W = computeCost(featuresAndPrediction, model)

        print("......................WSSE = {} ".format(W))



        wsseList.append(W)

    return wsseList



def elbow_plot(wsseList, clusters):

    wsseDF = pd.DataFrame({'WSSE' : wsseList, 'k' : clusters })

    wsseDF.plot(y='WSSE', x='k', figsize=(15,10), grid=True, marker='o')



def pd_centers(featuresUsed, centers):

    colNames = list(featuresUsed)

    colNames.append('prediction')



    # Zip with a column called 'prediction' (index)

    Z = [np.append(A, index) for index, A in enumerate(centers)]



    # Convert to pandas for plotting

    P = pd.DataFrame(Z, columns=colNames)

    P['prediction'] = P['prediction'].astype(int)

    return P



def parallel_plot(data, P):

    my_colors = list(islice(cycle(['b', 'r', 'g', 'y', 'k']), None, len(P)))

    plt.figure(figsize=(15,8)).gca().axes.set_ylim([-3,+3])

    parallel_coordinates(data, 'prediction', color = my_colors, marker='o')
df = sqlContext.read.load('../input/san-diego-daily-weather-data/minute_weather.csv', 

                          format='com.databricks.spark.csv', 

                          header='true',inferSchema='true')
df.count()
filteredDF = df.filter((df.rowID % 10 == 0))

filteredDF.count()
filteredDF.describe().toPandas().T
filteredDF.filter(filteredDF.rain_accumulation == 0).count()
filteredDF.filter(filteredDF.rain_duration == 0).count()
workingDF = filteredDF.drop('rain_accumulation').drop('rain_duration').drop('hpwren_timestamp')
before = workingDF.count()

workingDF = workingDF.na.drop()

after = workingDF.count()

print(before - after)
workingDF.columns
featuresUsed = ['air_pressure','air_temp','avg_wind_direction','avg_wind_speed','max_wind_direction',

                'max_wind_speed','relative_humidity']

assembler = VectorAssembler(inputCols=featuresUsed,outputCol='features_unscaled')

assembled = assembler.transform(workingDF)
assembled.show(2)
scaler = StandardScaler(inputCol='features_unscaled',outputCol='features',withMean=True,withStd=True)

scalerModel = scaler.fit(assembled)

scalerData = scalerModel.transform(assembled)
scalerData.show(2)
scalerData = scalerData.select("features","rowID")



elbowset = scalerData.filter(scalerData.rowID % 3 == 0).select("features")

elbowset.persist()
clusters = range(2,31)



wsseList = elbow(elbowset,clusters)
elbow_plot(wsseList,clusters)
scaledDataFeat = scalerData.select('features')

scaledDataFeat.persist()
kmeans = KMeans(k = 12, seed=1)

model = kmeans.fit(scaledDataFeat)



transformed = model.transform(scaledDataFeat)
centers = model.clusterCenters()

centers
P = pd_centers(featuresUsed,centers)

P.head()
parallel_plot(P[P['relative_humidity']<-0.5],P)
parallel_plot(P[P['air_temp']>0.5],P)
parallel_plot(P[(P['relative_humidity']>0.5) & (P['air_temp']<0.5) ],P)
parallel_plot(P.iloc[[2]],P)