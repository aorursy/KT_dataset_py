# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
#!pip install pyspark
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import sklearn
import nltk
import pandas as pd
from pyspark import SQLContext
from pyspark import SparkContext
import matplotlib.pyplot as plt
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import (DecisionTreeClassifier, GBTClassifier, RandomForestClassifier, LogisticRegression)
from pyspark.ml.evaluation import BinaryClassificationEvaluator
sales_df=pd.read_csv("../input/competitive-data-science-predict-future-sales/sales_train.csv")
test_df=pd.read_csv("../input/competitive-data-science-predict-future-sales/test.csv")
sales_df.head()
sales = sales_df.drop(["date","item_id","item_price"], axis=1)
pl=sales.groupby(['date_block_num','shop_id'])['item_cnt_day'].sum()
ax = pl.plot(x='date_block_num',y='item_cnt_day',color=['black', 'red', 'green', 'blue', 'cyan'],rot=0 ) 
#kp5=sales_df[['shop_id','item_cnt_day','date_block_num','shop_id','item_price']]
#kp5.head()
sc = SparkContext.getOrCreate()
sqlContext = SQLContext(sc)
df=sqlContext.createDataFrame(sales_df)
#df=df[['shop_id','item_cnt_day','date_block_num','item_price']]
#df.show(5)

data = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load("../input/competitive-data-science-predict-future-sales/sales_train.csv")
drop=['date']
data=data.select([column for column in sales_df.columns if column not in drop])
vectorAssembler = VectorAssembler(inputCols = ['shop_id','item_cnt_day','item_price'], outputCol = 'features')
df = vectorAssembler.transform(data)
df = df.select(['features', 'date_block_num'])
splits = df.randomSplit([0.6, 0.4])
train_df = splits[0]
test_df = splits[1]
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import DecisionTreeRegressor
dt = DecisionTreeRegressor(featuresCol ='features', labelCol = 'date_block_num')
dt_model = dt.fit(train_df)
dt_predictions = dt_model.transform(test_df)
dt_evaluator = RegressionEvaluator(
    labelCol="date_block_num", predictionCol="prediction", metricName="rmse")
rmse = dt_evaluator.evaluate(dt_predictions)
print("Karesel hata (RMSE) on test data = %g" % rmse)
dt_model.featureImportances
df.take(1)

from pyspark.ml.regression import GBTRegressor
gbt = GBTRegressor(featuresCol = 'features', labelCol = 'date_block_num', maxIter=10)
gbt_model = gbt.fit(train_df)
gbt_predictions = gbt_model.transform(test_df)
gbt_predictions.select('prediction', 'date_block_num', 'features').show(5)
gbt_evaluator = RegressionEvaluator(
    labelCol="date_block_num", predictionCol="prediction", metricName="rmse")
rmse = gbt_evaluator.evaluate(gbt_predictions)
print("Karesel hata (RMSE) on test data = %g" % rmse)