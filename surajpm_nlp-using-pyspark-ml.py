!pip install pyspark
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

from pyspark.sql import SparkSession

from pyspark.ml import Pipeline

from pyspark.ml.feature import CountVectorizer,StringIndexer, RegexTokenizer,StopWordsRemover

from pyspark.sql.functions import col, udf,regexp_replace,isnull

from pyspark.sql.types import StringType,IntegerType

from pyspark.ml.classification import NaiveBayes, RandomForestClassifier, LogisticRegression, DecisionTreeClassifier, GBTClassifier

from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
spark = SparkSession.builder.appName('nlp').getOrCreate()
filepath = '/kaggle/input/nlp-getting-started'

sdf_train = spark.read.csv(f'{filepath}/train.csv', header = True, inferSchema = True)

sdf_test = spark.read.csv(f'{filepath}/test.csv', inferSchema=True, header=True)



sdf_sample_submission = spark.read.csv(f'{filepath}/sample_submission.csv', 

                                       inferSchema=True, header=True)

sdf_train.printSchema()
import pandas as pd

pd.DataFrame(sdf_train.take(5), columns=sdf_train.columns)
print("Training Data Record Count:",sdf_train.count())

print("Test Data Record Count:",sdf_test.count())
sdf_train.toPandas().groupby(['target']).size()
ml_df = sdf_train.select("id","text","target")

ml_df.show(5)
ml_df = ml_df.dropna()

ml_df.count()
ml_df = ml_df.withColumn("only_str",regexp_replace(col('text'), '\d+', ''))

ml_df.show(5)
regex_tokenizer = RegexTokenizer(inputCol="only_str", outputCol="words", pattern="\\W")

raw_words = regex_tokenizer.transform(ml_df)

raw_words.show(5)
remover = StopWordsRemover(inputCol="words", outputCol="filtered")

words_df = remover.transform(raw_words)

words_df.select("id","words","target","filtered").show(5, truncate=False)
cv = CountVectorizer(inputCol="filtered", outputCol="features")

model = cv.fit(words_df)

countVectorizer_train = model.transform(words_df)

countVectorizer_train = countVectorizer_train.withColumn("label",col('target'))

countVectorizer_train.show(5)
countVectorizer_train.select('text','words','filtered','features','target').show()
(train, validate) = countVectorizer_train.randomSplit([0.8, 0.2],seed = 97435)
trainData = countVectorizer_train



#cleaning and preparing the test data

testData = sdf_test.select("id","text")#.dropna()

testData = testData.withColumn("only_str",regexp_replace(col('text'), '\d+', ''))

regex_tokenizer = RegexTokenizer(inputCol="only_str", outputCol="words", pattern="\\W")  #Extracting raw words

testData = regex_tokenizer.transform(testData)

remover = StopWordsRemover(inputCol="words", outputCol="filtered") #Removing stop words

testData = remover.transform(testData)

cv = CountVectorizer(inputCol="filtered", outputCol="features")

model = cv.fit(testData)

countVectorizer_test = model.transform(testData)

testData = countVectorizer_test

testData.show(5)
nb = NaiveBayes(modelType="multinomial",labelCol="label", featuresCol="features")

nbModel = nb.fit(train)

nb_predictions = nbModel.transform(validate)
nbEval = BinaryClassificationEvaluator()

print('Test Area Under ROC', nbEval.evaluate(nb_predictions))
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

nb_accuracy = evaluator.evaluate(nb_predictions)

print("Accuracy of NaiveBayes is = %g"% (nb_accuracy))
from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(featuresCol = 'features', labelCol = 'target', maxIter=10)

lrModel = lr.fit(train)
import matplotlib.pyplot as plt

import numpy as np



beta = np.sort(lrModel.coefficients)

plt.plot(beta)

plt.ylabel('Beta Coefficients')

plt.show()
trainingSummary = lrModel.summary

lrROC = trainingSummary.roc.toPandas()



plt.plot(lrROC['FPR'],lrROC['TPR'])

plt.ylabel('False Positive Rate')

plt.xlabel('True Positive Rate')

plt.title('ROC Curve')

plt.show()



print('Training set areaUnderROC: ' + str(trainingSummary.areaUnderROC))
pr = trainingSummary.pr.toPandas()

plt.plot(pr['recall'],pr['precision'])

plt.ylabel('Precision')

plt.xlabel('Recall')

plt.show()
lrPreds = lrModel.transform(validate)

lrPreds.select('id','prediction').show(5)
lrEval = BinaryClassificationEvaluator()

print('Test Area Under ROC', lrEval.evaluate(lrPreds))
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

lr_accuracy = evaluator.evaluate(lrPreds)

print("Accuracy of Logistic Regression is = %g"% (lr_accuracy))
from pyspark.ml.classification import DecisionTreeClassifier



dt = DecisionTreeClassifier(featuresCol = 'features', labelCol = 'target', maxDepth = 3)

dtModel = dt.fit(train)

dtPreds = dtModel.transform(validate)

dtPreds.show(5)

#dtPreds.select('age', 'job', 'label', 'rawPrediction', 'prediction', 'probability').show(10)
dtEval = BinaryClassificationEvaluator()

dtROC = dtEval.evaluate(dtPreds, {dtEval.metricName: "areaUnderROC"})

print("Test Area Under ROC: " + str(dtROC))
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

dt_accuracy = evaluator.evaluate(dtPreds)

print("Accuracy of Decision Trees is = %g"% (dt_accuracy))
dt = DecisionTreeClassifier(featuresCol = 'features', labelCol = 'target', maxDepth = 3)

dtModel = dt.fit(trainData)

dtPreds = dtModel.transform(testData)

dtPreds.show(5)
dtPreds.select('id','prediction').withColumnRenamed('prediction','target').toPandas().to_csv('dt_Pred.csv',index=False,header=True)
#rfPreds.select('id', 'prediction').withColumnRenamed('prediction','target').toPandas()#.to_csv('rf_Preds.csv',index=False)
#gbtPreds.select('id', 'prediction').withColumnRenamed('prediction','target').toPandas().to_csv('gbt_Preds.csv',index=False)