print("Starting with a csv of joined data from the original data and additional snapshot, this notebook checks if rMPI can be predicted by a collection of socioeconomic statistics, then starts a PCA analysis that could help construct a Kiva estimate of need/poverty aka Kestimate. During merging and cleaning, regional and country-level socioeconomic stats were looked up using the region name or latitude and longitude. Missing data were decreased by assigning a rounded latitute and longitude when it was missing but could  region was known.")

print("First will fix the count of male / female applicants, and sample 500k rows, about half the data, to save memory and training time")
#Reading in the data and fixing the counts of female/male loan applicants
import numpy as np 
import pandas as pd 
import random

data = pd.read_csv("../input/cleaned-combined-kiva/cleaned_combined_kiva.csv")
#Sampling 500k rows to save training time / memory
sample = random.sample(range(len(data)), 500000)
data = data.iloc[sample, :]

#Adjusting dtypes so they match
#Some empty spaces causing type issues in PPP variables, and filling NaNs with empty strings for string variables.
data['rPPP90'] = pd.to_numeric(data['rPPP90'].replace(" ",""))
data['rPPP95'] = pd.to_numeric(data['rPPP95'].replace(" ",""))
data['rPPP00'] = pd.to_numeric(data['rPPP00'].replace(" ",""))
data['rPPP05'] = pd.to_numeric(data['rPPP05'].replace(" ",""))
cols = data.columns.to_series().groupby(data.dtypes).groups
cols = {k.name: v for k, v in cols.items()}
for col in cols['object']:
    data[col] = data[col].fillna("")

def parsegender(v):
    genderlist = []
    if isinstance(v['borrower_genders'], str):
        genderlist = v['borrower_genders'].split(',')
    numfemale = 0
    nummale = 0
    for gender in genderlist:
        if len(gender.strip()) == 6:
            numfemale += 1
        if len(gender.strip()) == 4:
            nummale += 1
    return numfemale, nummale


df = data.apply(parsegender, axis=1)
data['femalec'] = df.apply(lambda x: x[0])
data['malec'] = df.apply(lambda x: x[1])
del data['borrower_genders']
del data['Unnamed: 0']

print(data['femalec'].describe())
print(data['malec'].describe())
#Borrowed this missing data check from another kernel
def mdata(data1):
    mtotal = data1.isnull().sum().sort_values(ascending=False)
    mpercent = (data1.isnull().sum() / data1.isnull().count()).sort_values(ascending=False)
    return pd.concat([mtotal, mpercent], axis=1, keys=['Total', 'Percent'])
print(mdata(data))


#A bit of a messy approach to improve location granularity by extracting region from town names.
#It takes a long time to run so I am skipping it.
'''
def addregionfromloc(v):
    if (v['region'] == '' or not isinstance(v['region'], str)) and (\
                    isinstance(v['town_name'], str) and len(v['town_name']) > 1):
    
        temp = mpi_locations.fillna('zzz').apply(lambda x: x['region'] in v['town_name'], axis=1)
        temp = temp[temp == True]
        for item in temp.iteritems():
        # Limit length problem with match to strings<5 like region 'Bo' or common word Centre
            if len(mpi_locations.iloc[item[0], 3]) > 5 and (mpi_locations.iloc[item[0], 3] != "Centre") and (\
                    mpi_locations.iloc[item[0], 3] != "Northern") and (\
                    mpi_locations.iloc[item[0], 3] != "Southern") and (\
                    mpi_locations.iloc[item[0], 3] != "Central"):
                print('ADDED region from town', mpi_locations.iloc[item[0], 3], ' from', v['town_name'])
                return mpi_locations.iloc[item[0], 3]

        return v['region']

data['region'] = data.apply(addregionfromloc, axis=1)
print(mdata(data))
'''
print("Switching to spark for ML")
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark import SparkContext, SparkConf

sc = (SparkSession.builder
                  .appName('Kiva')
                  .enableHiveSupport()
                  .config("spark.executor.memory", "2G")
                  .config("spark.driver.memory","17G")
                  .config("spark.executor.cores","7")
                  .config("spark.python.worker.memory","2G")
                  .config("spark.driver.maxResultSize","0")
                  .config("spark.sql.crossJoin.enabled", "true")
                  .config("spark.serializer","org.apache.spark.serializer.KryoSerializer")
                  .config("spark.default.parallelism","2")
                  .getOrCreate())
sc.sparkContext.setLogLevel('INFO')
data = sc.createDataFrame(data)
data.printSchema()
print("This cell shows that rMPI can be accurately predicted using the mix of country- and region-level stats in the data. This makes sense since rMPI is calculated using similar data. This random forest regression model could be very helpful to Kiva in estimating missing MPI values using existing data.")
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.util import MLUtils
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

kiva = data.na.drop(subset=["rMPI",'cHDI','cLE','cEDavg','GNI','rPop05','rPPP05'])
featurecolumns = ['cHDI','cLE','cEDavg','GNI','rPop05','rPPP05']
(training, test) = kiva.randomSplit([0.7, 0.3])

assembler = VectorAssembler(inputCols=featurecolumns, outputCol='features')
rf = RandomForestRegressor(featuresCol="features", labelCol='rMPI', numTrees=12, featureSubsetStrategy="auto", maxDepth=20)

pipeline = Pipeline(stages=[assembler, rf])

model = pipeline.fit(training)
predictions = model.transform(test)

predictions.select("prediction", "rMPI", "features").show(25)

evaluator = RegressionEvaluator(
    labelCol="rMPI", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse) 
print("Here we extract principal components from socioeconomic variables from the previous cell, plus loan and gender variables to meet Kiva's goal of incorporating those aspects into a metric.\
      So these 12 variables explain a lot about the loan, country- and regional poverty, and the number of applications + their gender. The PCA finds a way to capture 85% of all the variation across these traits in 5 components.")
from pyspark.ml.feature import PCA
from pyspark.ml.feature import StandardScaler

featurecolumns = ['femalec', 'malec', 'loan_amount', 'term_in_months', 'cHDI','cLE','cEDavg','cEDexp', 'GNI', 'rPop05', 'rPPP00', 'rPPP05']
kiva = data.na.drop(subset=featurecolumns)

assembler = VectorAssembler(inputCols=featurecolumns, outputCol='features')
scaler = StandardScaler(inputCol='features', outputCol='zfeatures', withStd=True, withMean=True)
pca = PCA(k=5, inputCol="zfeatures", outputCol="pcaFeatures")

pcapipe = Pipeline(stages=[assembler, scaler, pca])

pcamodel = pcapipe.fit(kiva)
pcatrained = pcamodel.stages[2]

print('Explained variance: ', pcatrained.explainedVariance)
print('Total explained variance: ', sum(pcatrained.explainedVariance))

print('Principal components matrix:')
print(pcatrained.pc)
print('The PCA could be taken further in order to construct an MPI style metric that factors in data about the loan, gender, etc. -- which could be called the Kestimate')