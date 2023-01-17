! pip install pyspark
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("."))



# Any results you write to the current directory are saved as output.
from pyspark.sql import SparkSession

spark = SparkSession.builder.master("local[*]").getOrCreate()
from pyspark.ml import Pipeline

from pyspark.ml.classification import LogisticRegression

from pyspark.ml.feature import Tokenizer, HashingTF
training = spark.createDataFrame([

    (0, "a b c d e spark", 1.0),

    (1, "b d", 0.0),

    (2, "spark f g h", 1.0),

    (3, "hadoop mapreduce", 0.0)

], ["id", "text", "label"])

print(training.printSchema())

training.toPandas()
tokenizer = Tokenizer(inputCol = "text", outputCol = "text_token")

type(tokenizer)
hashingTf = HashingTF(inputCol = "text_token", outputCol = "features" )
lr = LogisticRegression(maxIter = 10, regParam = 0.001)


pipeline = Pipeline(stages = [tokenizer,hashingTf,lr])
model = pipeline.fit(training)

type(model)
test = spark.createDataFrame([

    (4, "spark i j k"),

    (5, "l m n"),

    (6, "spark hadoop spark"),

    (7, "apache hadoop")

], ["id", "text"])

test.toPandas()
prediction = model.transform(test)

print(prediction.printSchema())

prediction.toPandas()
prediction.select("features").show(20,False)