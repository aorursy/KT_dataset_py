# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
!pip install pyspark
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pyspark.ml.feature import Tokenizer, RegexTokenizer, StopWordsRemover,HashingTF, IDF, StringIndexer
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType
from pyspark.sql import SparkSession
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("TwitterClassificationNaive")\
        .getOrCreate()
df = pd.read_excel("/kaggle/input/text-tweet-classification/text_classification_dataset.xlsx", sheet_name="Sheet1")
sdf = spark.createDataFrame(df)
indexer = StringIndexer(inputCol="type", outputCol="label")
indexedDF = indexer.fit(sdf).transform(sdf)
tokenizer = Tokenizer(inputCol="text", outputCol="words")

tokenized = tokenizer.transform(indexedDF)
tokenized.show(truncate=False)
remover = StopWordsRemover(inputCol="words", outputCol="filtered")
filteredDF = remover.transform(tokenized)
filteredDF.show(truncate=False)
hasher = HashingTF(inputCol="filtered", outputCol="rawFeatures")
featurizedData = hasher.transform(filteredDF)
featurizedData.show(truncate=False)

idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)
rescaledData.show(truncate=False)
splits = rescaledData.randomSplit([0.75, 0.25])
train = splits[0]
test = splits[1]
# create the trainer and set its parameters
nb = NaiveBayes(smoothing=1.0, modelType="multinomial")

# train the model
model = nb.fit(train)
# select example rows to display.
predictions = model.transform(test)
predictions.show()

# compute accuracy on the test set
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                              metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test set accuracy = " + str(accuracy))