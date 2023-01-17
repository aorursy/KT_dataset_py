!pip -q install pyspark
import os

import pandas as pd

import numpy as np



from pyspark import SparkConf, SparkContext

from pyspark.sql import SparkSession, Window

from pyspark.sql.types import *

import pyspark.sql.functions as f



from pyspark.ml import Transformer, Pipeline, regression

from pyspark.ml.feature import VectorAssembler, OneHotEncoder

from pyspark.ml.evaluation import RegressionEvaluator



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
spark = SparkSession.builder.master("local[*]").appName("retail_demand_forecasting").getOrCreate()
spark
schema = StructType([StructField("date", DateType()),StructField("store", IntegerType()),

                     StructField("item", IntegerType()),StructField("sales", FloatType())])

train_df = spark.read.csv(path = '/kaggle/input/demand-forecasting-kernels-only/train.csv', schema=schema, header = True).cache()

train_df.printSchema()



schema = StructType([StructField("id", IntegerType()),

                     StructField("date", DateType()),StructField("store", IntegerType()),

                     StructField("item", IntegerType())])

test_df = spark.read.csv(path = '/kaggle/input/demand-forecasting-kernels-only/test.csv', schema=schema, header = True).cache()

test_df.printSchema()
train_df = train_df.withColumn('type',f.lit("train"))

train_df = train_df.withColumn('id',f.lit(None))



test_df = test_df.withColumn('type',f.lit("test"))

test_df = test_df.withColumn('sales',f.lit(None))



df = train_df.unionByName(test_df)
class DomExtractor(Transformer):

    def __init__(self, inputCol, outputCol='dayofmonth'):

        self.inputCol = inputCol

        self.outputCol = outputCol



    def _transform(self, df):

        return df.withColumn(self.outputCol, f.dayofmonth(df[self.inputCol]))

    

class DoyExtractor(Transformer):

    def __init__(self, inputCol, outputCol='dayofyear'):

        self.inputCol = inputCol

        self.outputCol = outputCol



    def _transform(self, df):

        return df.withColumn(self.outputCol, f.dayofyear(df[self.inputCol]))

    

class DowDayExtractor(Transformer):

    def __init__(self, inputCol, outputCol='dayofweek'):

        self.inputCol = inputCol

        self.outputCol = outputCol



    def _transform(self, df):

        return df.withColumn(self.outputCol, f.dayofweek(df[self.inputCol]))

    

    

class MonthExtractor(Transformer):

    def __init__(self, inputCol, outputCol='month'):

        self.inputCol = inputCol

        self.outputCol = outputCol



    def _transform(self, df):

        return df.withColumn(self.outputCol, f.month(df[self.inputCol]))

    

class YearExtractor(Transformer):

    def __init__(self, inputCol, outputCol='year'):

        self.inputCol = inputCol

        self.outputCol = outputCol



    def _transform(self, df):

        return df.withColumn(self.outputCol, f.year(df[self.inputCol]))

    

class YearQuarterExtractor(Transformer):

    def __init__(self, inputCol='month', outputCol='yearquarter'):

        self.inputCol = inputCol

        self.outputCol = outputCol



    def _transform(self, df):

        return df.withColumn(self.outputCol, f.when((df[self.inputCol] <= 3), 0).otherwise(f.when((df[self.inputCol] <= 6), 1).otherwise(f.when((df[self.inputCol] <= 9), 2).otherwise(3))))

    

    

class WeekendExtractor(Transformer):

    def __init__(self, inputCol='dayofweek', outputCol='is_weekend'):

        self.inputCol = inputCol

        self.outputCol = outputCol



    def _transform(self, df):

        return df.withColumn(self.outputCol, f.when(((df[self.inputCol] == 1) | (df[self.inputCol] == 7)), 1).otherwise(0))



class MonthBeginExtractor(Transformer):

    def __init__(self, inputCol='dayofmonth', outputCol='monthbegin'):

        self.inputCol = inputCol

        self.outputCol = outputCol



    def _transform(self, df):

        return df.withColumn(self.outputCol,f.when((df[self.inputCol] <= 7), 1).otherwise(0))

    

    

class MonthEndExtractor(Transformer):

    def __init__(self, inputCol='dayofmonth', outputCol='monthend'):

        self.inputCol = inputCol

        self.outputCol = outputCol



    def _transform(self, df):

        return df.withColumn(self.outputCol, f.when((df[self.inputCol] >= 24), 1).otherwise(0))

    

class logTransform(Transformer):

    def __init__(self, inputCol, outputCol):

        self.inputCol = inputCol

        self.outputCol = outputCol



    def _transform(self, df):

        return df.withColumn(self.outputCol, f.log1p(f.col(self.inputCol)))

    



class LagExtractor(Transformer):

    """Creating sales lag features"""

    def __init__(self, inputCol, outputCol='lag', dateCol='date', idCol=['store', 'item'], lags = [91]):

        self.inputCol = inputCol

        self.outputCol = outputCol

        self.dateCol = dateCol

        self.idCol = idCol

        self.lags = lags

        

    def _transform(self, df):

        for lag in self.lags:

            col_name = (self.outputCol + '%s' % lag)

            

            w = Window.partitionBy(self.idCol).orderBy(self.dateCol).rowsBetween(-lag, -lag)

            

            df = df.withColumn(col_name, f.collect_list(self.inputCol).over(w)[0])

            

        return df

    

class RmeanExtractor(Transformer):

    """Creating sales rolling mean features"""

    def __init__(self, inputCol, outputCol='rmean', dateCol='date', idCol=['store', 'item'], avgRange=[30], shift = 90):

        self.inputCol = inputCol

        self.outputCol = outputCol

        self.dateCol = dateCol

        self.idCol = idCol

        self.avgRange = avgRange

        self.shift = shift



    def _transform(self, df):

        for ar in self.avgRange:

            col_name = (self.outputCol + '%s_%s' % (ar,self.shift))

            

            w = Window.partitionBy(self.idCol).orderBy(self.dateCol).rowsBetween(-ar-self.shift, -1-self.shift) # exclude itself

            

            df = df.withColumn(col_name, f.avg(self.inputCol).over(w))

            

        return df
# Feature extraction

dom = DomExtractor(inputCol='date')

doy = DoyExtractor(inputCol='date')

dow = DowDayExtractor(inputCol='date')

mon = MonthExtractor(inputCol='date')

#year = YearExtractor(inputCol='date')

yq = YearQuarterExtractor()



wked = WeekendExtractor()

mbe = MonthBeginExtractor()

med = MonthEndExtractor()



logt = logTransform(inputCol ='sales', outputCol='logSales')

lagex = LagExtractor(inputCol = 'logSales', lags = [91,98,105,112,119,126,182,364,546,728])

meanex = RmeanExtractor(inputCol = 'logSales', avgRange=[364,546])



encoder = OneHotEncoder(inputCols=["store","item","dayofmonth", "dayofweek","month","yearquarter"],

                        outputCols=["storeVec","itemVec","dayofmonthVec", "dayofweekVec","monthVec","yearquarterVec"])



pipeline = Pipeline(stages=[ dom, doy, dow, mon, yq, wked, mbe, med, logt, lagex, meanex, encoder])



processing = pipeline.fit(df)

transformed = processing.transform(df)

transformed.printSchema()



assembler = VectorAssembler(inputCols=["dayofweekVec","monthVec","storeVec","itemVec","dayofmonthVec","yearquarterVec",

                                       "is_weekend","monthbegin","monthend",

                                       "lag91","lag98","lag105","lag112","lag119","lag126",

                                       "lag182","lag364","lag546","lag728",

                                       "rmean364_90","rmean546_90"], 

                            outputCol="features")





transformed_train = assembler.transform(transformed.filter(  (f.col('lag728').isNotNull()) & (f.col('type') == 'train')    ))

transformed_test = assembler.transform(transformed.filter(f.col('type') == 'test'))
# model fitting

#rf = regression.RandomForestRegressor(numTrees = 100, maxDepth=6, featuresCol='features', labelCol='logSales')

gb = regression.GBTRegressor(maxDepth=5, seed=42, featuresCol='features', labelCol='logSales')

model = gb.fit(transformed_train)
pred_train = model.transform(transformed_train)

pred_train = pred_train.select('sales',f.expm1('prediction').alias('pred'))



evaluator = RegressionEvaluator(predictionCol='pred', labelCol='sales', metricName='rmse')

print(evaluator.evaluate(pred_train))



evaluator = RegressionEvaluator(predictionCol='pred', labelCol='sales', metricName='mae')

print(evaluator.evaluate(pred_train))
# make predictions

pred_test = model.transform(transformed_test)

pred_test = pred_test.select('*',f.expm1('prediction').alias('pred'))

sub_df = pred_test.select('id',f.col('pred').alias('sales')).orderBy('id').toPandas()

sub_df.to_csv('submission.csv',index=False)