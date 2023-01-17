# load datasets from kaggle
def get_data():
    !mkdir /kaggle
    !mkdir /kaggle/input
    !pip install -q kaggle
    from google.colab import files
    files.upload()
    !mkdir -p ~/.kaggle
    !mv kaggle.json ~/.kaggle/
    !chmod 600 /root/.kaggle/kaggle.json

    !kaggle datasets download -d meraxes10/bd-models
    !unzip bd-models.zip -d /kaggle/input/bd-models > /dev/null
    !rm bd-models.zip    

    !!kaggle datasets download -d azathoth42/myanimelist
    !unzip myanimelist.zip -d /kaggle/input/myanimelist > /dev/null
    !rm myanimelist.zip
!wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
!unzip ngrok-stable-linux-amd64.zip
#get_data()
!pip install pyspark > /dev/null
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import Window
import pyspark.ml.feature
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, CountVectorizer
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.stat import Summarizer
import numpy as np
import re
import gc
SEED = 42
ROOT = '/kaggle/input/myanimelist/'
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
conf = SparkConf().set("spark.ui.port", "4050") \
                  .set('spark.executor.memory', '9G') \
                  .set('spark.driver.memory', '7G') \
                  .set('spark.sql.autoBroadcastJoinThreshold', '-1')

sc = pyspark.SparkContext(conf=conf)
spark = SparkSession.builder.getOrCreate()
get_ipython().system_raw('./ngrok http 4050 &')
!curl -s http://localhost:4040/api/tunnels
!curl -s http://localhost:4040/api/tunnels | python3 -c "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"
anime_df = spark.read.format("csv").option("header", "true") \
                                   .option("headers", "true") \
                                   .option('escape','"') \
                                   .option("inferSchema", "true") \
                                   .load(ROOT + "AnimeList.csv", sep=',')
cols = ['title', 'title_english', 'title_japanese', 'title_synonyms', 'image_url', 'aired_string', 'background',
       'broadcast', 'related', 'opening_theme', 'ending_theme', 'studio']
anime_df = anime_df.drop(*cols)
cols = ['premiered', 'producer', 'licensor', 'rank']
anime_df = anime_df.drop(*cols)
cols = ['aired', 'duration', 'airing']
anime_df = anime_df.drop(*cols)
anime_df = anime_df.fillna('', subset=['genre'])
anime_df = anime_df.withColumn(
    'genre',
    split(regexp_replace('genre', ' ', ''), ',').cast("array<string>").alias("genre")
)
anime_df = anime_df.withColumn("episodes", anime_df.episodes.cast('float'))
anime_df = anime_df.withColumn("score", anime_df.score.cast('float'))
anime_df = anime_df.withColumn("scored_by", anime_df.scored_by.cast('float'))
anime_df = anime_df.withColumn("popularity", anime_df.popularity.cast('float'))
anime_df = anime_df.withColumn("members", anime_df.members.cast('float'))
anime_df = anime_df.withColumn("favorites", anime_df.favorites.cast('float'))
anime_df.show(5)
anime_df = CountVectorizer(inputCol="genre", outputCol="genre_fv").fit(anime_df).transform(anime_df)
anime_df = anime_df.drop('genre')
categoricalColumns = ['type', 'source', 'status', 'rating']
stages = []
for categoricalCol in categoricalColumns:
    stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')
    encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
    stages += [stringIndexer, encoder]
numericCols = ['episodes', 'score', 'scored_by', 'popularity', 'members', 'favorites', 'genre_fv'] 
assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="item_feats_profile")
stages += [assembler]
@udf("array<integer>")
def indices(v):
    return v.indices.tolist()
pipeline = Pipeline(stages = stages)
pipelineModel = pipeline.fit(anime_df)
anime_df = pipelineModel.transform(anime_df)
selectedCols = ['anime_id', 'item_feats_profile']
anime_df = anime_df.select(selectedCols)
anime_df.show(5)
user_df = spark.read.format("csv").option("header", "true") \
                                   .option("headers", "true") \
                                   .option('escape','"') \
                                   .option("inferSchema", "true") \
                                   .load(ROOT + "UserList.csv", sep=',')
user_df = user_df.filter(col('username').isNotNull())
user_df = user_df.filter(col('stats_episodes').isNotNull())
cols = ['location', 'access_rank', 'stats_mean_score', 'birth_date', 'gender']
user_df = user_df.drop(*cols)
cols = ['join_date', 'last_online']
user_df = user_df.drop(*cols)
user_df = user_df.withColumn("user_id", user_df.user_id.cast('float'))
user_df = user_df.withColumn("user_watching", user_df.user_watching.cast('float'))
user_df = user_df.withColumn("user_completed", user_df.user_completed.cast('float'))
user_df = user_df.withColumn("user_onhold", user_df.user_onhold.cast('float'))
user_df = user_df.withColumn("user_dropped", user_df.user_dropped.cast('float'))
user_df = user_df.withColumn("user_plantowatch", user_df.user_plantowatch.cast('float'))
user_df = user_df.withColumn("user_days_spent_watching", user_df.user_days_spent_watching.cast('float'))
user_df = user_df.withColumn("stats_rewatched", user_df.stats_rewatched.cast('float'))
user_df = user_df.withColumn("stats_episodes", user_df.stats_episodes.cast('float'))
user_df.show(5)
cols = ['user_watching', 'user_completed', 'user_onhold', 'user_dropped', 'user_plantowatch', 'user_days_spent_watching', 
        'stats_rewatched', 'stats_episodes']
assembler = VectorAssembler(inputCols=cols, outputCol="user_feats_profile")
stages = [assembler]
pipeline = Pipeline(stages = stages)
pipelineModel = pipeline.fit(user_df)
user_df = pipelineModel.transform(user_df)
selectedCols = ['username', 'user_id', 'user_feats_profile']
user_df = user_df.select(selectedCols)
user_df.show(5)
user_anime_df = spark.read.format("csv").option("header", "true").load(ROOT + "UserAnimeList.csv")
cols = ['my_watched_episodes', 'my_start_date', 'my_finish_date', 'my_status', 'my_rewatching', 
        'my_rewatching_ep', 'my_last_updated', 'my_tags']
user_anime_df = user_anime_df.drop(*cols)
user_anime_df = user_anime_df.withColumn("anime_id", user_anime_df.anime_id.cast('float'))
user_anime_df = user_anime_df.withColumn("my_score", user_anime_df.my_score.cast('float'))
user_anime_df = user_anime_df.na.drop()
user_anime_df = user_anime_df.filter(user_anime_df.my_score <= 10)
user_anime_df = user_anime_df.filter(user_anime_df.my_score != 0)
user_anime_df.show(5)
user_anime_df = user_anime_df.join(user_df, 'username', how='left')
user_anime_df = user_anime_df.drop('username')
user_anime_df = user_anime_df.join(anime_df, 'anime_id', how='left')
user_anime_df = user_anime_df.na.drop()
assembler = VectorAssembler(inputCols=["user_feats_profile", "item_feats_profile"], outputCol="features")
user_anime_df = assembler.transform(user_anime_df)
user_anime_df.show(5)
(training, test) = user_anime_df.randomSplit([0.8, 0.2], seed=SEED)
(training, valid) = training.randomSplit([0.9, 0.1], seed=SEED)
avg_score_by_anime = training.groupBy('anime_id').agg(avg('my_score').alias('preds_0'))
avg_score = training.agg(avg('my_score').alias('overall_average'))
c = avg_score.collect()
valid = valid.join(avg_score_by_anime, 'anime_id', how='left')
valid = valid.fillna(c[0].overall_average, subset=['preds_0'])
evaluator = RegressionEvaluator(metricName="rmse", labelCol="my_score", predictionCol="preds_0")
rmse = evaluator.evaluate(valid)
print("RMSE:" + str(rmse))
test = test.join(avg_score_by_anime, 'anime_id', how='left')
test = test.fillna(c[0].overall_average, subset=['preds_0'])
avg_score_by_user = training.groupBy('user_id').agg(avg('my_score').alias('preds_1'))
valid = valid.join(avg_score_by_user, 'user_id', how='left')
valid = valid.fillna(c[0].overall_average, subset=['preds_1'])
evaluator = RegressionEvaluator(metricName="rmse", labelCol="my_score", predictionCol="preds_1")
rmse = evaluator.evaluate(valid)
print("RMSE:" + str(rmse))
test = test.join(avg_score_by_user, 'user_id', how='left')
test = test.fillna(c[0].overall_average, subset=['preds_1'])
from pyspark.ml.regression import RandomForestRegressor, RandomForestRegressionModel
rf = RandomForestRegressor(featuresCol='features', labelCol='my_score', 
                           numTrees=5, maxMemoryInMB=1024,
                           subsamplingRate=0.1)
#rf_model = rf.fit(training)
rf_model = RandomForestRegressionModel.load('/kaggle/input/bd-models/rf_model')
valid = rf_model.transform(valid)
valid = valid.withColumnRenamed("prediction", "preds_2")
test = rf_model.transform(test)
test = test.withColumnRenamed("prediction", "preds_2")
evaluator = RegressionEvaluator(metricName="rmse", labelCol="my_score", predictionCol="preds_2")
rmse = evaluator.evaluate(valid)
print("RMSE:" + str(rmse))
from pyspark.ml.regression import LinearRegression, LinearRegressionModel
from pyspark.ml.feature import StandardScaler, StandardScalerModel
scaler = StandardScaler(inputCol="features", 
                        outputCol="std_features",
                        withStd=True, withMean=True)
#model_scaler = scaler.fit(training)
model_scaler = StandardScalerModel.load('/kaggle/input/bd-models/scaler_model')
training = model_scaler.transform(training)
valid = model_scaler.transform(valid)
test = model_scaler.transform(test)
lm = LinearRegression(featuresCol='std_features', labelCol='my_score', maxIter=10)
#lr_model = lm.fit(training)
lr_model = LinearRegressionModel.load('/kaggle/input/bd-models/lr_model')
valid = lr_model.transform(valid)
test = lr_model.transform(test)
valid = valid.withColumnRenamed("prediction", "preds_3")
test = test.withColumnRenamed("prediction", "preds_3")
evaluator = RegressionEvaluator(metricName="rmse", labelCol="my_score", predictionCol="preds_3")
rmse = evaluator.evaluate(valid)
print("RMSE:" + str(rmse))
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
als = ALS(userCol="user_id", itemCol="anime_id", ratingCol="my_score", coldStartStrategy="drop")

param_grid = ParamGridBuilder().addGrid(als.rank, [25]) \
                               .addGrid(als.regParam, [0.1, 0.15]) \
                               .addGrid(als.maxIter, [10]) \
                               .build()
tvs = TrainValidationSplit(estimator=als,
                           estimatorParamMaps=param_grid,
                           evaluator=RegressionEvaluator(metricName="rmse", labelCol="my_score", predictionCol="prediction"),
                           trainRatio=0.8)
maxIter = 10
regParam = 0.1
rank=25
als = ALS(maxIter=maxIter, regParam=regParam, rank=rank, 
          userCol="user_id", itemCol="anime_id", ratingCol="my_score",
          coldStartStrategy="drop")
#als_model = als.fit(training)
als_model = ALSModel.load('/kaggle/input/bd-models/als_model')
valid = als_model.transform(valid)
test = als_model.transform(test)
valid = valid.withColumnRenamed("prediction", "preds_4")
test = test.withColumnRenamed("prediction", "preds_4")
evaluator = RegressionEvaluator(metricName="rmse", labelCol="my_score", predictionCol="preds_4")
rmse = evaluator.evaluate(valid)
print("RMSE:" + str(rmse))
userRecs = als_model.recommendForAllUsers(10)
userRecs.show(5)
alpha_0 = 0.1
alpha_1 = 0.1
alpha_2 = 0.1
alpha_3 = 0.1
alpha_4 = 0.6
valid = valid.fillna(c[0].overall_average)
valid = valid.withColumn('prediction', alpha_0*valid['preds_0'] + alpha_1*valid['preds_1'] + alpha_2*valid['preds_2'] + alpha_3*valid['preds_3'] + alpha_4*valid['preds_4'] )
valid = valid.select(['anime_id', 'user_id', 'my_score', 'prediction'])
evaluator = RegressionEvaluator(metricName="rmse", labelCol="my_score", predictionCol="prediction")
rmse = evaluator.evaluate(valid)
print("RMSE:" + str(rmse))
test = test.fillna(c[0].overall_average)
test = test.withColumn('prediction', alpha_0*test['preds_0'] + alpha_1*test['preds_1'] + alpha_2*test['preds_2'] + alpha_3*test['preds_3'] + alpha_4*test['preds_4'] )
test = test.select(['anime_id', 'user_id', 'my_score', 'prediction'])
evaluator = RegressionEvaluator(metricName="rmse", labelCol="my_score", predictionCol="prediction")
rmse = evaluator.evaluate(test)
print("RMSE:" + str(rmse))
