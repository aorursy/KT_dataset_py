!pip install pyspark
!pip install scikit-learn
from pyspark import SparkConf
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.sql import DataFrameReader
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from pyspark.ml import Pipeline
from pyspark.ml.feature import RFormula
from pyspark.ml.regression import RandomForestRegressor
from pyspark.mllib.evaluation import RegressionMetrics
from pyspark.ml import PipelineModel
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.feature import RFormula
from pyspark.ml.regression import LinearRegression
from pyspark.mllib.evaluation import RegressionMetrics
from pylab import *
import matplotlib.pyplot as plt
import numpy as np
import datetime
##logFile = "E:\\anacanda/Lib/site-packages/pyspark/bin/README.md"  # Should be some file on your system
spark = SparkSession.builder.appName("SimpleApp").getOrCreate()


##Set data storage path. This is where data is sotred on the blob attached to the cluster.
dataDir = '../output/kaggle/working/'; # The last backslash is needed;
dataDir2 = '../output/kaggle/working2/'; # The last backslash is needed;

sqlContext = SQLContext(spark)
## READ IN TRIP DATA FRAME FROM CSV

trip_fare_join='../input/newyork-taxi-demand/yellow_tripdata_2016-01.csv'
trip_fare = spark.read.csv(path=trip_fare_join, header=True, inferSchema=True)

trip_fare.printSchema()

dataframe1 = trip_fare.withColumnRenamed('  pickup_longitude', 'pickup_longitude')
dataframe2 = dataframe1.withColumnRenamed('   pickup_latitude', 'pickup_latitude')
dataframe3 = dataframe2.withColumnRenamed(' dropoff_longitude', 'dropoff_longitude')
dataframe4 = dataframe3.withColumnRenamed('  dropoff_latitude', 'dropoff_latitude')



## READ IN FARE DATA FRAME FROM CSV



dataDir3 ='../output/kaggle/working3/'; # The last backslash is needed;

dataframe4path = dataDir3 + "dataframe4";

dataframe4.write.mode("overwrite").parquet(dataframe4path)

dataframe4 = pd.read_parquet(dataframe4path)   # Dataframe as pd. convert Spark's PD to Pandas's PD

dataframe4.head(5)

print('Data Shape',dataframe4.shape)
dataframe4.info()
dataframe4.describe().transpose()
# Remove coordinate outliers
dataframe4 = dataframe4[dataframe4['pickup_longitude'] <= -73.75]
dataframe4 = dataframe4[dataframe4['pickup_longitude'] >= -74.03]
dataframe4 = dataframe4[dataframe4['pickup_latitude'] <= 40.85]
dataframe4 = dataframe4[dataframe4['pickup_latitude'] >= 40.63]
dataframe4 = dataframe4[dataframe4['dropoff_longitude'] <= -73.75]
dataframe4 = dataframe4[dataframe4['dropoff_longitude'] >= -74.03]
dataframe4 = dataframe4[dataframe4['dropoff_latitude'] <= 40.85]
dataframe4 = dataframe4[dataframe4['dropoff_latitude'] >= 40.63]

dataframe4.describe().transpose()
#check for missing values in dataframe4 data
dataframe4.isnull().sum().sort_values(ascending=False)

#highest fare is $250
#highest tip is $125

dataframe4 = dataframe4.drop(dataframe4[(dataframe4['fare_amount']==0)].index, axis=0)
dataframe4 = dataframe4.drop(dataframe4[(dataframe4['fare_amount']>250)].index, axis=0)
dataframe4 = dataframe4.drop(dataframe4[(dataframe4['fare_amount']<0)].index, axis=0)

dataframe4 = dataframe4.drop(dataframe4[(dataframe4['tip_amount']>125)].index, axis=0)
dataframe4 = dataframe4.drop(dataframe4[(dataframe4['tip_amount']<0)].index, axis=0)



# import seaborn as sns

# x=dataframe4['fare_amount']
# y=dataframe4['tip_amount']


# sns.scatterplot(x,y)
dataframe4['fare_amount'].sort_values(ascending=False)

dataframe4['tip_amount'].sort_values(ascending=False)

dataframe4.trip_distance[(dataframe4.trip_distance==0)].count()
dataframe4=dataframe4.drop(dataframe4[(dataframe4['trip_distance']==0)].index,axis=0)
dataframe4=dataframe4.drop(dataframe4[(dataframe4['trip_distance']>21.000000)].index,axis=0)

dataframe4[(dataframe4.pickup_latitude != dataframe4.dropoff_latitude) &
              (dataframe4.pickup_longitude != dataframe4.dropoff_longitude) &
              (dataframe4.trip_distance == 0)].count()

# dataframe4 = dataframe4.drop(dataframe4[(dataframe4.pickup_latitude!=dataframe4.dropoff_latitude)&
#                                         (dataframe4.pickup_longitude!=dataframe4.dropoff_longitude)&
#                                         (dataframe4.trip_distance == 0)].index, axis = 0)

dataframe4[(dataframe4['trip_distance']==0)&(dataframe4['fare_amount']==0)].count()
dataframe4[(dataframe4['trip_distance']==0)&((dataframe4['fare_amount']==0)|(dataframe4['fare_amount']!=0))].count()

dataframe4[(dataframe4['trip_distance']==10)].count()
dataframe4[(dataframe4['trip_distance']==21.000000)].count()
import seaborn as sns

x=dataframe4['trip_distance']
y=dataframe4['fare_amount']


sns.scatterplot(x,y)
dataframe4[(dataframe4['trip_distance']==1)&((dataframe4['fare_amount']>100)|(dataframe4['fare_amount']>100))].count()

import seaborn as sns

x=dataframe4['trip_distance']
y=dataframe4['passenger_count']


sns.scatterplot(x,y)
dataframe4[(dataframe4['trip_distance']==1)&((dataframe4['passenger_count']>3)|(dataframe4['passenger_count']>6))].count()

plt.figure(figsize=(10,7))
plt.hist(dataframe4['passenger_count'],bins=15)
plt.xlabel('No of Passanger')
plt.ylabel('Frequency')
dataframe4=dataframe4.drop(dataframe4[(dataframe4['passenger_count']==0)].index,axis=0)
dataframe4=dataframe4.drop(dataframe4[(dataframe4['passenger_count']>6)].index,axis=0)


dataframe4[(dataframe4['passenger_count']==0)].count()

dataframe4[(dataframe4['passenger_count']==2)].count()

dataframe4[(dataframe4['passenger_count']==3)].count()

dataframe4[(dataframe4['passenger_count']==4)].count()

dataframe4[(dataframe4['passenger_count']==5)].count()

dataframe4[(dataframe4['passenger_count']==6)].count()

import seaborn as sns

corrMat = dataframe4[::].corr(); 
ax = plt.subplots(figsize=(13, 12))
ax = sns.heatmap(corrMat,vmin=-1, vmax=1, annot=True, square = True,linewidths=2);
dataframe4 = dataframe4.drop(columns=['trip_distance'])

dataframe4 = dataframe4.drop(columns=['payment_type'])
dataframe4 = dataframe4.drop(columns=['passenger_count'])
dataframe4 = dataframe4.drop(columns=['improvement_surcharge'])

dataframe4 = dataframe4.drop(columns=['extra'])

dataframe4 = dataframe4.drop(columns=['mta_tax'])

dataframe4 = dataframe4.drop(columns=['VendorID'])

dataframe4=dataframe4.drop(dataframe4[(dataframe4['total_amount']>255)].index,axis=0)

dataframe4 = dataframe4.drop(columns=['store_and_fwd_flag'])

dataframe4 = dataframe4.drop(columns=['tpep_dropoff_datetime'])

def add_pickupdatetime_info(dataset):
    #Convert to datetime format
    dataset['tpep_pickup_datetime'] = pd.to_datetime(dataset['tpep_pickup_datetime'],format="%Y-%m-%d %H:%M:%S")
    
    dataset['pickup_hour'] = dataset.tpep_pickup_datetime.dt.hour
    dataset['pickup_day'] = dataset.tpep_pickup_datetime.dt.day
    dataset['pickup_month'] = dataset.tpep_pickup_datetime.dt.month
    dataset['pickup_weekday'] = dataset.tpep_pickup_datetime.dt.weekday
    dataset['pickup_year'] = dataset.tpep_pickup_datetime.dt.year
    
    return dataset
dataframe4 = add_pickupdatetime_info(dataframe4)
dataframe4.describe()
plt.figure(figsize=(10,7))
plt.hist(train_data['hour'],bins=50)
plt.xlabel('Hour')
plt.ylabel('Frequency')
dataframe4_copy_sklearn = dataframe4.copy()
y = dataframe4_copy_sklearn['fare_amount']

x = dataframe4_copy_sklearn.drop(columns=['fare_amount'])
from sklearn.pipeline import Pipeline
data_pipeline = Pipeline([('rob_scale', RobustScaler())])
dataframe4.info()
dataframe4_copy_sklearn.drop(columns=['tpep_pickup_datetime'],inplace=True)
x_scaled = data_pipeline.fit_transform(x)
x_scaled = pd.DataFrame(x_scaled,columns=x.columns,index=x.index)
import seaborn as sns

sns.distplot(a=dataframe4_copy_sklearn.total_amount)
import seaborn as sns

sns.distplot(a=dataframe4_copy_sklearn.fare_amount)
# Mean distribution
mu = x_scaled['total_amount'].mean()

# Std distribution
sigma = x_scaled['total_amount'].std()
num_bins = 100

# Histogram 
fig = plt.figure(figsize=(8.5, 5))
n, bins, patches = plt.hist(x_scaled['total_amount'], num_bins, normed=1,
                           edgecolor = 'black', lw = 1, alpha = .40)
# Normal Distribution
y = mlab.normpdf(bins, mu, sigma)
plt.plot(bins, y, 'r--', linewidth=2)
plt.xlabel('total_amount')
plt.ylabel('Probability density')

# Adding a title
plt.title(r'$\mathrm{Trip\ duration\ skewed \ to \ the \ right:}\ \mu=%.3f,\ \sigma=%.3f$'%(mu,sigma))
plt.grid(True)
#fig.tight_layout()
plt.show()

# Statistical summary
x_scaled.describe()[['total_amount']].transpose()



# dataframe4_copy_sklearn.to_parquet(dataframe4_copy_sklearn,path=dataDir4, engine='pyarrow', compression='none', index=None, partition_cols=None)

def write_parquet_file():
    df = dataframe4_copy_sklearn
    df.to_parquet('/kaggle/working/x.parquet')
    
write_parquet_file()



dataDir4 ='/kaggle/working/x.parquet';

x = spark.read.parquet(dataDir4)   # Dataframe as pd. convert Spark's PD to Pandas's PD

x.createOrReplaceTempView("fareamount1")

## USING SQL: MERGE TRIP AND FARE DATA-SETS TO CREATE A JOINED DATA-FRAME
## ELIMINATE SOME COLUMNS, AND FILTER ROWS WTIH VALUES OF SOME COLUMNS
sqlStatement = """SELECT
  fa.total_amount, fa.tolls_amount,fa.dropoff_latitude,
  fa.fare_amount,fa.pickup_latitude,fa.dropoff_longitude,
  fa.tip_amount,fa.pickup_hour,fa.pickup_year,fa.pickup_month,fa.pickup_day,fa.pickup_weekday,fa.pickup_longitude
  FROM fareamount1 fa 
  WHERE fa.tip_amount >= 0 AND fa.tip_amount <= 125 
  AND fa.fare_amount >= 1 AND fa.fare_amount <= 250
  AND fa.tip_amount < fa.fare_amount"""
  
fare_amountDF = spark.sql(sqlStatement)

# REGISTER JOINED TRIP-FARE DF IN SQL-CONTEXT
fare_amountDF.createOrReplaceTempView("fareamount2")

## SHOW WHICH TABLES ARE REGISTERED IN SQL-CONTEXT
spark.sql("show tables").show()


# SAMPLE 10% OF DATA, SPLIT INTO TRAIINING AND VALIDATION AND SAVE IN BLOB
trip_fare_featSampled = fare_amountDF.sample(False, 0.1, seed=1234)
trainfilename = dataDir + "TrainData";
trip_fare_featSampled.repartition(10).write.mode("overwrite").parquet(trainfilename)
## READ IN DATA FRAME FROM CSV
taxi_df = spark.read.parquet(trainfilename)
## CREATE A CLEANED DATA-FRAME BY DROPPING SOME UN-NECESSARY COLUMNS & FILTERING FOR UNDESIRED VALUES OR OUTLIERS
taxi_df_cleaned = taxi_df.drop('RatecodeID')\
    .filter("fare_amount >= 1 AND fare_amount < 100" )

## PERSIST AND MATERIALIZE DF IN MEMORY
taxi_df_cleaned.persist()

## REGISTER DATA-FRAME AS A TEMP-TABLE IN SQL-CONTEXT
taxi_df_cleaned.createOrReplaceTempView("taxi_df")
taxi_df_cleaned.printSchema()

spark.sql("show tables").show()

#%%sql -q -o sqlResultsPD
#SELECT fare_amount, passenger_count, tip_amount FROM taxi_train WHERE passenger_count > 0 AND passenger_count < 7 AND fare_amount > 0 AND fare_amount < 100 AND tip_amount > 0 AND tip_amount < 15

sqlResultsPDtest = """SELECT fare_amount,
passenger_count,
tip_amount FROM taxi_train
WHERE passenger_count > 0 AND passenger_count < 7 AND fare_amount > 0 AND fare_amount < 100 AND tip_amount > 0 AND tip_amount < 15"""
sqlResultsPD = spark.sql(sqlResultsPDtest)

sqlResultsPD.createOrReplaceTempView("trip_test_final")

spark.sql("show tables").show()

trip_fare_fit_final = sqlResultsPD

trainfilename2 = dataDir2 + "TrainDatafinal";

trip_fare_fit_final.write.mode("overwrite").parquet(trainfilename2)



trip_fare_fit_final_df = pd.read_parquet(trainfilename2)
#%%local
%matplotlib inline
## %%local creates a pandas data-frame on the head node memory, from spark data-frame,
##which can then be used for plotting. Here, sampling data is a good idea, depending on the memory of the head node

# TIP BY PAYMENT TYPE AND PASSENGER COUNT
ax1 = trip_fare_fit_final_df[['tip_amount']].plot(kind='hist', bins=25, facecolor='lightblue')
ax1.set_title('Tip amount distribution')
ax1.set_xlabel('Tip Amount ($)'); ax1.set_ylabel('Counts');
plt.figure(figsize=(4,4)); plt.suptitle(''); plt.show()

# TIP BY PASSENGER COUNT
ax2 = trip_fare_fit_final_df.boxplot(column=['tip_amount'], by=['passenger_count'])
ax2.set_title('Tip amount by Passenger count')
ax2.set_xlabel('Passenger count'); ax2.set_ylabel('Tip Amount ($)');
plt.figure(figsize=(4,4)); plt.suptitle(''); plt.show()

# TIP AMOUNT BY FARE AMOUNT, POINTS ARE SCALED BY PASSENGER COUNT
ax = trip_fare_fit_final_df.plot(kind='scatter', x= 'fare_amount', y = 'tip_amount', c='blue', alpha = 0.10, s=2.5*(trip_fare_fit_final_df.passenger_count))
ax.set_title('Tip amount by Fare amount')
ax.set_xlabel('Fare Amount ($)'); ax.set_ylabel('Tip Amount ($)');
plt.axis([-2, 80, -2, 20])
plt.figure(figsize=(4,4)); plt.suptitle(''); plt.show()

### CREATE FOUR BUCKETS FOR TRAFFIC TIMES
sqlStatement = """SELECT
  total_amount,tolls_amount,dropoff_latitude,
  fare_amount,pickup_latitude,dropoff_longitude,
  tip_amount,pickup_hour,pickup_year,pickup_month,pickup_day,pickup_weekday,pickup_longitude,
  CASE
    WHEN (pickup_hour <= 6 OR pickup_hour >= 20) THEN 'Night'
    WHEN (pickup_hour >= 7 AND pickup_hour <= 10) THEN 'AMRush' 
    WHEN (pickup_hour >= 11 AND pickup_hour <= 15) THEN 'Afternoon'
    WHEN (pickup_hour >= 16 AND pickup_hour <= 19) THEN 'PMRush'
    END as TrafficTimeBins
    FROM taxi_df"""

taxi_df = spark.sql(sqlStatement)
taxi_df.show()
from pyspark.ml import Pipeline

# DEFINE THE TRANSFORMATIONS THAT NEEDS TO BE APPLIED TO SOME OF THE FEATURES
sI4 = StringIndexer(inputCol="TrafficTimeBins", outputCol="TrafficTimeBinsIndex");

# APPLY TRANSFORMATIONS
encodedFinal = Pipeline(stages=[sI4]).fit(taxi_df).transform(taxi_df);

encodedFinal.show()
trainingFraction = 0.75; testingFraction = (1-trainingFraction);
seed = 1234;

# SPLIT SAMPLED DATA-FRAME INTO TRAIN/TEST, WITH A RANDOM COLUMN ADDED FOR DOING CV (SHOWN LATER)
trainData,testData = encodedFinal.randomSplit([trainingFraction, testingFraction], seed=seed);




# CACHE DATA FRAMES IN MEMORY
trainData.persist(); 
testData.persist();
from pyspark.ml.feature import RFormula
from pyspark.ml.regression import LinearRegression
from pyspark.mllib.evaluation import RegressionMetrics

## DEFINE REGRESSION FURMULA
regFormula = RFormula(formula="fare_amount ~ pickup_day + pickup_hour + pickup_weekday + pickup_month + TrafficTimeBinsIndex + total_amount + tolls_amount + pickup_longitude + pickup_latitude + dropoff_longitude + dropoff_latitude + tip_amount")

## DEFINE INDEXER FOR CATEGORIAL VARIABLES
featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=32)

## DEFINE ELASTIC NET REGRESSOR
eNet = LinearRegression(featuresCol="indexedFeatures", maxIter=50,regParam=0.01, elasticNetParam=0.7)

## Fit model, with formula and other transformations
model = Pipeline(stages=[regFormula, featureIndexer, eNet]).fit(trainData)

print("Coefficients: " + str(model.coefficients))
print("Intercept: " + str(model.intercept))
# model_fit = Pipeline.fit(trainData)
# fitandlabel = model_fit.select("label","model_fit").rdd
# testMetrics = RegressionMetrics(fitandlabel)
# print("R-sqr = %s" % testMetrics.r2)

## PREDICT ON TEST DATA AND EVALUATE
predictions = model.transform(testData)
predictionAndLabels = predictions.select("label","prediction").rdd
testMetrics = RegressionMetrics(predictionAndLabels)
print("RMSE = %s" % testMetrics.rootMeanSquaredError)
print("R-sqr = %s" % testMetrics.r2)

## PLOC ACTUALS VS. PREDICTIONS
predictions.select("label","prediction").createOrReplaceTempView("tmp_results");

from pyspark.ml.feature import RFormula
from sklearn.metrics import roc_curve,auc
from pyspark.ml.regression import RandomForestRegressor
from pyspark.mllib.evaluation import RegressionMetrics

modelDir = '../output/kaggle/working3/'; # The last backslash is needed;

## DEFINE REGRESSION FURMULA
regFormula = RFormula(formula="fare_amount ~ pickup_day + pickup_hour + pickup_weekday + pickup_month + TrafficTimeBinsIndex + total_amount + tolls_amount + pickup_longitude + pickup_latitude + dropoff_longitude + dropoff_latitude + tip_amount")

## DEFINE INDEXER FOR CATEGORIAL VARIABLES
#featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=32)

## DEFINE RANDOM FOREST ESTIMATOR
randForest = RandomForestRegressor(featuresCol = 'features', labelCol = 'label', numTrees=10,
                                   featureSubsetStrategy="auto",impurity='variance', maxDepth=4, maxBins=100)

## Fit model, with formula and other transformations
model = Pipeline(stages=[regFormula,randForest]).fit(trainData)

## SAVE MODEL
datestamp = datetime.datetime.now().strftime('%m-%d-%Y-%s');
fileName = "RandomForestRegressionModel_" + datestamp;
randForestDirfilename = modelDir + fileName;
model.save(randForestDirfilename)

## PREDICT ON TEST DATA AND EVALUATE
predictions = model.transform(testData)
predictionAndLabels = predictions.select("label","prediction").rdd
testMetrics = RegressionMetrics(predictionAndLabels)
print("RMSE = %s" % testMetrics.rootMeanSquaredError)
print("R-sqr = %s" % testMetrics.r2)

## PLOC ACTUALS VS. PREDICTIONS
predictions.select("label","prediction").createOrReplaceTempView("tmp_results");

#%%sql -q -o predictionsPD

dataDir3 = '../output/kaggle/working3/';

query = "SELECT * from tmp_results"

query_df = spark.sql(query)

# REGISTER JOINED TRIP-FARE DF IN SQL-CONTEXT
query_df.createOrReplaceTempView("tmp_results2")

query_dfwr = query_df

trainfilename3 = dataDir3 + "TrainDatafinal";

query_dfwr.write.mode("overwrite").parquet(trainfilename3)

predictionsPD = pd.read_parquet(trainfilename3)

#%%local


ax = predictionsPD.plot(kind='scatter', figsize = (5,5), x='label', y='prediction', color='blue', alpha = 0.25, label='Actual vs. predicted');
fit = np.polyfit(predictionsPD['label'], predictionsPD['prediction'], deg=1)
ax.set_title('Actual vs. Predicted Tip Amounts ($)')
ax.set_xlabel("Actual"); ax.set_ylabel("Predicted");
ax.plot(predictionsPD['label'], fit[0] * predictionsPD['label'] + fit[1], color='magenta')
plt.axis([-1, 15, -1, 15])
plt.show(ax)
## DEFINE RANDOM FOREST MODELS
randForest = RandomForestRegressor(featuresCol = 'features', labelCol = 'label',
                                   featureSubsetStrategy="auto",impurity='variance', maxBins=100)

## DEFINE MODELING PIPELINE, INCLUDING FORMULA, FEATURE TRANSFORMATIONS, AND ESTIMATOR
pipeline = Pipeline(stages=[regFormula, randForest])

## DEFINE PARAMETER GRID FOR RANDOM FOREST
paramGrid = ParamGridBuilder() \
    .addGrid(randForest.numTrees, [10, 25, 50]) \
    .addGrid(randForest.maxDepth, [3, 5, 7]) \
    .build()

## DEFINE CROSS VALIDATION
crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=RegressionEvaluator(metricName="rmse"),
                          numFolds=3)

## TRAIN MODEL USING CV
cvModel = crossval.fit(trainData)

## PREDICT AND EVALUATE TEST DATA SET
predictions = cvModel.transform(testData)
evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")
r2 = evaluator.evaluate(predictions)
print("R-squared on test data = %g" % r2)

## SAVE THE BEST MODEL
datestamp = datetime.datetime.now().strftime('%m-%d-%Y-%s');
fileName = "CV_RandomForestRegressionModel_" + datestamp;
CVDirfilename = modelDir + fileName;
cvModel.bestModel.save(CVDirfilename);
savedModel = PipelineModel.load(randForestDirfilename)

predictions = savedModel.transform(testData)
predictionAndLabels = predictions.select("label","prediction").rdd
testMetrics = RegressionMetrics(predictionAndLabels)
print("RMSE = %s" % testMetrics.rootMeanSquaredError)
print("R-sqr = %s" % testMetrics.r2)
## READ IN DATA FRAME FROM CSV
taxi_valid_df = spark.read.csv(path=taxi_valid_file_loc, header=True, inferSchema=True)
taxi_valid_df.printSchema()
## READ IN DATA FRAME FROM CSV
taxi_valid_df = spark.read.csv(path=taxi_valid_file_loc, header=True, inferSchema=True)

## CREATE A CLEANED DATA-FRAME BY DROPPING SOME UN-NECESSARY COLUMNS & FILTERING FOR UNDESIRED VALUES OR OUTLIERS
taxi_df_valid_cleaned = taxi_valid_df.drop('medallion').drop('hack_license').drop('store_and_fwd_flag').drop('pickup_datetime')\
    .drop('dropoff_datetime').drop('pickup_longitude').drop('pickup_latitude').drop('dropoff_latitude')\
    .drop('dropoff_longitude').drop('tip_class').drop('total_amount').drop('tolls_amount').drop('mta_tax')\
    .drop('direct_distance').drop('surcharge')\
    .filter("passenger_count > 0 and passenger_count < 8 AND payment_type in ('CSH', 'CRD') \
    AND tip_amount >= 0 AND tip_amount < 30 AND fare_amount >= 1 AND fare_amount < 150 AND trip_distance > 0 \
    AND trip_distance < 100 AND trip_time_in_secs > 30 AND trip_time_in_secs < 7200" )

## REGISTER DATA-FRAME AS A TEMP-TABLE IN SQL-CONTEXT
taxi_df_valid_cleaned.createOrReplaceTempView("taxi_valid")

### CREATE FOUR BUCKETS FOR TRAFFIC TIMES
sqlStatement = """ SELECT *, CASE
     WHEN (pickup_hour <= 6 OR pickup_hour >= 20) THEN "Night" 
     WHEN (pickup_hour >= 7 AND pickup_hour <= 10) THEN "AMRush" 
     WHEN (pickup_hour >= 11 AND pickup_hour <= 15) THEN "Afternoon"
     WHEN (pickup_hour >= 16 AND pickup_hour <= 19) THEN "PMRush"
    END as TrafficTimeBins
    FROM taxi_valid
"""
taxi_df_valid_with_newFeatures = spark.sql(sqlStatement)

## APPLY THE SAME TRANSFORATION ON THIS DATA AS ORIGINAL TRAINING DATA
encodedFinalValid = Pipeline(stages=[sI1, sI2, sI3, sI4]).fit(taxi_df_train_with_newFeatures).transform(taxi_df_valid_with_newFeatures)
## LOAD SAVED MODEL, SCORE VALIDATION DATA, AND EVALUATE
savedModel = PipelineModel.load(CVDirfilename)
predictions = savedModel.transform(encodedFinalValid)
r2 = evaluator.evaluate(predictions)
print("R-squared on validation data = %g" % r2)
datestamp = datetime.datetime.now().strftime('%m-%d-%Y-%s');
fileName = "Predictions_CV_" + datestamp;
predictionfile = dataDir + fileName;
predictions.select("label","prediction").write.mode("overwrite").csv(predictionfile)
spark.stop()