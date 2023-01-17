#install Apache Spark

!pip install pyspark --quiet
#Generic Libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# -------- PySpark Libraries --------



#Apache Spark Libraries

import pyspark

from pyspark.sql import SparkSession



#Apache Spark ML CLassifier Libraries

from pyspark.ml.classification import DecisionTreeClassifier,RandomForestClassifier,NaiveBayes



#Apache Spark Evaluation Library

from pyspark.ml.evaluation import MulticlassClassificationEvaluator



#Apache Spark Features libraries

from pyspark.ml.feature import StandardScaler,StringIndexer



#Apache Spark Pipelin Library

from pyspark.ml import Pipeline



# Apache Spark `DenseVector`

from pyspark.ml.linalg import DenseVector





# -------- SciKit Libraries --------



#Data Split Libraries

import sklearn

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.metrics import accuracy_score



#ML Classifier Algorithm Libraries

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB, MultinomialNB

from sklearn.ensemble import RandomForestClassifier



#Tabulating Data

from tabulate import tabulate



#Garbage

import gc
#Building Spark Session

spark = (SparkSession.builder

                  .appName('Apache Spark Beginner Tutorial')

                  .config("spark.executor.memory", "1G")

                  .config("spark.executor.cores","4")

                  .getOrCreate())
spark.sparkContext.setLogLevel('INFO')
spark.version
url = '../input/iris-dataset/iris.csv'



# PySpark DataFrame

data_sprk = spark.read.format("csv").option("header", "true").option("inferSchema","true").load(url) 

data_sprk.cache() #for faster re-use



# SciKit DataFrame

data_sk = pd.read_csv(url, header="infer")
#Total records 

print("PySpark - " , data_sprk.count())

print("SciKit - ", data_sk.shape)
#Data Type



#PySpark

print(data_sprk.printSchema())

#SciKit

print(data_sk.info())

#Display records



#PySpark

print(data_sprk.show(5))

#SciKit

print(data_sk.head())
#Records per Species



#PySpark

print(data_sprk.groupBy('species').count().show())



#SciKit

print(data_sk.groupby('species').size())
#Dataset Summary Stats



#PySpark

print(data_sprk.describe().show())



#SciKit

print(data_sk.describe().transpose())
# -- PySpark --

SIndexer = StringIndexer(inputCol='species', outputCol='species_indx')

data_sprk = SIndexer.fit(data_sprk).transform(data_sprk)



# -- SciKit --

label_encode = LabelEncoder()

data_sk['species'] = label_encode.fit_transform(data_sk['species'])



#Inspect the dataset

print(data_sprk.show(5))

print(data_sk.head())

#creating a seperate dataframe with re-ordered columns

df_sprk = data_sprk.select("species_indx","sepal_length", "sepal_width", "petal_length", "petal_width")



#Inspect the dataframe

df_sprk.show(5)
# Define the `input_data` as Dense Vector

input_data = df_sprk.rdd.map(lambda x: (x[0], DenseVector(x[1:])))
# Creating a new Indexed Dataframe

df_sprk_indx = spark.createDataFrame(input_data, ["label", "features"])
#view the indexed dataframe

df_sprk_indx.show(5)
#Feature & Target Selection - SciKit

features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

target = ['species']



X = data_sk[features]

y = data_sk[target]
# --- PySpark ---

stdScaler = pyspark.ml.feature.StandardScaler(inputCol="features", outputCol="features_scaled")

scaler = stdScaler.fit(df_sprk_indx)

df_sprk_scaled =scaler.transform(df_sprk_indx)
# --- SciKit ---

sc = StandardScaler()

df_sk_scaled = sc.fit_transform(X)

df_sk_scaled = pd.DataFrame(df_sk_scaled, columns= ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
#Inspect the Scaled Data

print(df_sprk_scaled.show(5))

print(df_sk_scaled.head())
#Dropping the Features column

df_sprk_scaled = df_sprk_scaled.drop("features")
#PySpark

train_data_sprk, test_data_sprk = df_sprk_scaled.randomSplit([0.9, 0.1], seed = 12345)



#SciKit

X_train, X_test, y_train, y_test = train_test_split(df_sk_scaled, y, test_size=0.1, random_state= 1234)

#Inspect Training Data

print(train_data_sprk.show(5))

print(X_train.head())
model = ['Decision Tree','Random Forest','Naive Bayes']

model_results = []
# -- PySpark --



dtc_sprk = pyspark.ml.classification.DecisionTreeClassifier(labelCol="label", featuresCol="features_scaled")          

dtc_sprk_model = dtc_sprk.fit(train_data_sprk)                                                        

dtc_sprk_pred = dtc_sprk_model.transform(test_data_sprk)                                              



#Evaluate Model

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

dtc_sprk_acc = evaluator.evaluate(dtc_sprk_pred)                             
# -- SciKit --

dtc_sk = DecisionTreeClassifier()

dtc_sk.fit(X_train,y_train)

dtc_sk_pred = dtc_sk.predict(X_test)



#Evaluate Model

dtc_sk_acc = accuracy_score(y_test, dtc_sk_pred)

# -- PySpark --

rfc_sprk = pyspark.ml.classification.RandomForestClassifier(labelCol="label", featuresCol="features_scaled", numTrees=10)          

rfc_sprk_model = rfc_sprk.fit(train_data_sprk)                                                                     

rfc_sprk_pred = rfc_sprk_model.transform(test_data_sprk)                                                       



#Evaluate the Model

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

rfc_sprk_acc = evaluator.evaluate(rfc_sprk_pred)

                                      
# -- SciKit --

rfc_sk = RandomForestClassifier(n_estimators=1000, criterion='entropy', random_state=None, bootstrap=True)

rfc_sk.fit(X_train,y_train)

rfc_sk_pred = rfc_sk.predict(X_test)



#Evaluate Model

rfc_sk_acc = accuracy_score(y_test, rfc_sk_pred)
# -- PySpark --

nbc_sprk = pyspark.ml.classification.NaiveBayes(smoothing=1.0,modelType="gaussian", labelCol="label",featuresCol="features_scaled")   

nbc_sprk_model = nbc_sprk.fit(train_data_sprk)                                                                          

nbc_sprk_pred = nbc_sprk_model.transform(test_data_sprk)                                                                



#Evaluate the Model

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

nbc_sprk_acc = evaluator.evaluate(nbc_sprk_pred)

                                        
# -- SciKit --

nbc_sk = GaussianNB()

nbc_sk.fit(X_train,y_train)

nbc_sk_pred = nbc_sk.predict(X_test)



#Evaluate Model

nbc_sk_acc = accuracy_score(y_test, nbc_sk_pred)
#freeing memory

gc.collect()
model_data = [['Decision Tree Classifier', '{:.2%}'.format(dtc_sprk_acc),'{:.2%}'.format(dtc_sk_acc)], \

              ['Random Forest Classifier', '{:.2%}'.format(rfc_sprk_acc),'{:.2%}'.format(rfc_sk_acc)], \

              ['Naive Bayes (Gaussian)',   '{:.2%}'.format(nbc_sprk_acc),'{:.2%}'.format(nbc_sk_acc)]]
print (tabulate(model_data, headers=["Classifier Models", "PySpark Accuracy", "SciKit Accuracy"]))