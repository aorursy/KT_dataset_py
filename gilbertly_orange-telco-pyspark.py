# imports
from time import time
from subprocess import check_output
from pyspark.sql import (SparkSession, Row)
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import (StringIndexer, StandardScaler)
from pyspark.ml.tuning import (CrossValidator, ParamGridBuilder)
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.linalg import Vectors
# load spark session
sc = SparkSession\
    .builder\
    .master("local[*]")\
    .appName('telco_churn')\
    .getOrCreate()

# check for csv file(s)
print(check_output(["ls", "../input"]).decode("utf8"))
df = sc.read.load('../input/bigml_59c28831336c6604c800002a.csv', 
                  format='com.databricks.spark.csv', 
                  header='true', 
                  inferSchema='true').cache()
df.printSchema()
# check sample data
df.toPandas().head(10).transpose()
# drop some columns
df = df.drop(
    "State", "Area code", "Total day charge",
    "Total night charge", "Total intl charge",
    "phone number")

# cast churn label from boolean to string
df = df.withColumn("Churn", df["Churn"].cast("string"))
# turn categoric data to numerical
def toNumeric(df):
    cols = ["Churn", "international plan", "voice mail plan"]
    for col in cols:
        df = StringIndexer(
            inputCol=col,
            outputCol=col+"_indx")\
            .fit(df)\
            .transform(df)\
            .drop(col)\
            .withColumnRenamed(col+"_indx", col)
    return df

df = toNumeric(df).cache()
# check label proportions
df.groupBy("Churn").count().toPandas()
# perform some down-sampling
df = df.sampleBy(
    "Churn", 
    fractions={
        0: 483./2850,
        1: 1.0
    }).cache()
df.groupBy("Churn").count().toPandas()
feature_cols = df.columns
feature_cols.remove("Churn")

# make label as last column
df = df[feature_cols + ["Churn"]]

# vectorize labels and features
row = Row("label", "features")
df_vec = df.rdd.map(
    lambda r: (row(r[-1], Vectors.dense(r[:-1])))
).toDF()
df_vec.show(5)
# normalize the features
df_vec = StandardScaler(
    inputCol="features",
    outputCol="features_norm",
    withStd=True,
    withMean=True)\
    .fit(df_vec)\
    .transform(df_vec)\
    .drop("features")\
    .withColumnRenamed("features_norm", "features")

# split data into train/test
train, test = df_vec.randomSplit([0.8, 0.2])
print("Train values: '{}'".format(train.count()))
print("Test values: '{}'".format(test.count()))
start_time = time()
r_forest = RandomForestClassifier(
    numTrees = 100,
    labelCol = "label"
)
rf_model = r_forest.fit(train)

print("Training time taken: {0:.4f}(min)".format((time() - start_time)/60))
predictions = rf_model.transform(test)
acc = BinaryClassificationEvaluator(
    rawPredictionCol="rawPrediction",
    labelCol="label",
    metricName="areaUnderROC")\
    .evaluate(predictions)
print("Accuracy (binary): '{0:.4f}%'".format(acc*100))
# # tune best performing model: random forest
# paramGrid = ParamGridBuilder()\
#     .addGrid(r_forest.maxDepth, [5,10,15,20,25,30])\
#     .addGrid(r_forest.numTrees, [30, 60, 90, 120, 150, 180, 200])\
#     .build()

# # define evaluation metric
# evaluator = BinaryClassificationEvaluator(
#     rawPredictionCol="rawPrediction", 
#     metricName="areaUnderROC"
# )

# # start tuning
# cv = CrossValidator(
#     estimator=r_forest, 
#     estimatorParamMaps=paramGrid, 
#     evaluator=evaluator, 
#     numFolds=5
# )

# # start timer
# cv_start_time = time()

# # fit tuned model
# cvModel = cv.fit(train)

# # calculate time taken to tune prameters
# print("Hyper-param tuning time taken: '{0:.2}' (min)".format((time() - cv_start_time)/60))
# # accuracy after tuning
# train_pred = cvModel.transform(train)
# test_pred = cvModel.transform(test)
# print("Random forest accuracy (train): {0:.4f}%".format((evaluator.evaluate(train_pred))*100))
# print("Random forest accuracy (test): {0:.4f}%".format((evaluator.evaluate(test_pred))*100))


