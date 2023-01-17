!pip install pyspark
!wget https://raw.githubusercontent.com/prasadpatil99/Recommendation-System/master/ratings.csv
from pyspark.sql import SparkSession

from pyspark.ml.evaluation import RegressionEvaluator

from pyspark.ml.recommendation import ALS
spark = SparkSession.builder.appName('rec').getOrCreate()
data1 = spark.read.csv('ratings.csv',inferSchema=True,header=True)

features_drop = ['timestamp']

data = data1.drop(*features_drop)
data.printSchema()
data.show()
data.describe().show()
(train, test) = data.randomSplit([0.7, 0.2])
als = ALS(maxIter=4, regParam=0.15, userCol="userId", 

          itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")

model = als.fit(train)
predictions = model.transform(test)
predictions.show()
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",predictionCol="prediction")

rmse = evaluator.evaluate(predictions)

print("Root-mean-square error = " + str(rmse))
predict = test.filter(test['userId']>500).select(['movieId','userId'])
reccomendations = model.transform(predict)
reccomendations.orderBy('prediction',ascending=False).show()