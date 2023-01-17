!pip install pyspark --q
#Generic Libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#Apache Spark Libraries

import pyspark

from pyspark.sql import SparkSession

from pyspark.sql.types import IntegerType



#Apache Spark ML CLassifier Libraries

from pyspark.ml.classification import NaiveBayes



#Apache Spark Evaluation Library

from pyspark.ml.evaluation import BinaryClassificationEvaluator



#Apache Spark Features libraries

from pyspark.ml.feature import HashingTF, IDF, Tokenizer

from pyspark.ml.feature import Word2Vec



#Apache Spark Pipelin Library

from pyspark.ml import Pipeline



#Apache Spark Fine Tuning Libraries

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder





#Gensim Library for Text Processing

import gensim.parsing.preprocessing as gsp

from gensim import utils







#Tabulating Data

from tabulate import tabulate



#Garbage

import gc



#Warnings

import warnings

warnings.filterwarnings("ignore")
#Building Spark Session

spark = (SparkSession.builder

                  .appName('Restaurant Reviews Text Classification using Pyspark')

                  .config("spark.executor.memory", "1G")

                  .config("spark.executor.cores","4")

                  .getOrCreate())



# Logging Level

spark.sparkContext.setLogLevel('INFO')
url = '../input/restaurant-reviews/Restaurant_Reviews.csv'



data = spark.read.csv(url, header=True, inferSchema=True)
#total records

data.count()
#Data Types

data.printSchema()
#Converting Liked column data to integer

data = data.withColumn('Liked', data['Liked'].cast(IntegerType()))
#Records per Liked column

data.groupby('Liked').count().show()
# Filling the null value with 0

data = data.fillna(0)
#inspect data

data.show(5)
# Create list of pre-processing func

processes = [

           gsp.strip_tags, 

           gsp.strip_punctuation,

           gsp.strip_multiple_whitespaces,

           gsp.strip_numeric,

           gsp.remove_stopwords, 

           gsp.strip_short, 

           gsp.stem_text

          ]



# Create func to pre-process text

def proc_txt(txt):

    text = txt[0]

    text = text.lower()  #lowering the case

    text = utils.to_unicode(text)

    for p in processes:

        text = p(text)

    return (text,txt[1])    

        
#Creating a temp dataset with processed data

temp_ds = data.rdd.map(lambda x : proc_txt(x))



# Create a new Dataset

data_proc = temp_ds.toDF(['Proc_Review','Liked'])



#Inspect New Dataset

data_proc.show(5)
# Split the processed data into 90-10 ratio [90% - Training & 10% - Validation]

train_data, test_data = data_proc.randomSplit([0.9, 0.1])
#TF-IDF Vectorizing

tok = Tokenizer(inputCol='Proc_Review', outputCol='Tok_Review')

hashT = HashingTF(inputCol=tok.getOutputCol(), outputCol='raw_features_tf', numFeatures=30)

idf = IDF(inputCol=hashT.getOutputCol(), outputCol='Features_tf', minDocFreq=5)



#Create a TF-IDF pipeline

tf_pipe = Pipeline(stages=[tok, hashT, idf])
#Fit TF-IDF Pipeline to Training & Test Data

tf_mod = tf_pipe.fit(train_data)



#Transforming the data

train_data = tf_mod.transform(train_data)

test_data = tf_mod.transform(test_data)
#Function to Create, Traing & Evaluate Multinomial NB Model



def mnb_mod(train,test):

    

    # Build Seperate Models for TF-IDF & Word2Vec

    mnb_tf = NaiveBayes(smoothing=1.0, labelCol='Liked',featuresCol='Features_tf', modelType="multinomial")

    

    # Fit the Models to Train Data

    mnb_mod_tf = mnb_tf.fit(train)

    

    # Make Predictions

    pred_tf = mnb_mod_tf.transform(test)

    

    # Evaluation

    mnb_eval = BinaryClassificationEvaluator(labelCol='Liked')

    

    acc_tf = mnb_eval.evaluate(pred_tf)

    

    print("Multinomial Naive Bayes Model Accuracy =", '{:.2%}'.format(acc_tf))

        



    

#Applying the Function to vectorized data

mnb_mod(train_data,test_data)
# Fine Tuning the model using Cross Validator & ParamBuilder



def MNB_CV(train,test):

    

    mnb = NaiveBayes(smoothing=1.0, labelCol='Liked',featuresCol='Features_tf', modelType="multinomial")

    

    pipe = Pipeline(stages= [mnb])

    

    paramGrid = ParamGridBuilder().addGrid(mnb.smoothing, [1.0, 2.0, 3.0]).build()

    

    evaluate = BinaryClassificationEvaluator(labelCol="Liked")

    

    crossValidator = CrossValidator(estimator=pipe,

                                        evaluator=evaluate,

                                        estimatorParamMaps=paramGrid,

                                        numFolds=10)

    

    # use the Multinomial Model to train (fit) the model

    # and Get the best Multinomial Naive Bayes model



    cv = crossValidator.fit(train)

    tuned_mod = cv.bestModel.stages[0]



    predict = tuned_mod.transform(train)



    acc_new = evaluate.evaluate(predict)

    

    print("Multinomial Naive Bayes Model Accuracy (fine tuned) =", '{:.2%}'.format(acc_new)) 





# Applying the function to train & test data

MNB_CV(train_data,test_data)
