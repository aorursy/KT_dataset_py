import findspark
findspark.init()
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, udf
from pyspark.ml.feature import RegexTokenizer, CountVectorizer, \
    IDF, StopWordsRemover, StringIndexer
from pyspark.ml.classification import NaiveBayes
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.types import IntegerType

import re
spark = SparkSession.builder \
    .master("local") \
    .appName("Sentiment Analysis") \
    .getOrCreate()
path = 'data/amazon-alexa-reviews.tsv'
df = spark.read.option("sep", "\t") \
    .option("header", "true") \
    .csv(path)
df.show(3)
df.toPandas().info()
df.groupBy('rating').agg(count('rating')).orderBy('rating').show()
label_col = udf(lambda x: int((x =='5')|(x=='4')), IntegerType())  
df = df.withColumn('classe', label_col(df.rating))
df.show(1)
dfn = df.drop(*['date', 'variation', 'feedback'])
dfn.show(3)
def preprocessor(text):
    text = re.sub('<[^>]*>', '', str(text))
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) +\
        ' '.join(emoticons).replace('-', '')
    return text
preprocessor_udf = udf(preprocessor)
dfn = dfn.withColumn('prepared_reviews', preprocessor_udf(col('verified_reviews')))
dfn.show(3)
# diviser le texte en mots séparés
regexTokenizer = RegexTokenizer(inputCol="prepared_reviews", outputCol="mots", pattern="\\W")
dfn = regexTokenizer.transform(dfn)
dfn.show(1)
# supprimer les mots vides
remover = StopWordsRemover(inputCol='mots', outputCol='mots_clean')
dfn = remover.transform(dfn).select('rating', 'mots_clean')
dfn.show(2)
# trouver le terme fréquences des mots
cv = CountVectorizer(inputCol="mots_clean", outputCol="TF")
cvmodel = cv.fit(dfn)
dfn = cvmodel.transform(dfn)
dfn.take(1)
# trouver le Inter-document Frequency
idf = IDF(inputCol="TF", outputCol="features")
idfModel = idf.fit(dfn)
dfn = idfModel.transform(dfn)
dfn.head()
# créer la colonne d'étiquette
indexer = StringIndexer(inputCol="classe", outputCol="label")
data = df.drop(*['date', 'variation', 'feedback'])
data = data.withColumn('prepared_reviews', preprocessor_udf(col('verified_reviews')))
# Divisez le jeu de données au hasard en ensembles de formation et de test
(trainingData, testData) = data.randomSplit([0.7, 0.3], seed = 100)
# créer le pipeline
nb = NaiveBayes(smoothing=1.0)
pipeline = Pipeline(stages=[regexTokenizer, remover, cv, idf, indexer, nb])
# éxecuter les étapes du pipeline et former le modele
model = pipeline.fit(trainingData)
# Faire des prédictions sur testData 
#afin que nous puissions mesurer la précision de notre modèle sur de nouvelles données
predictions = model.transform(testData)
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",\
            metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Model Accuracy: ", accuracy)
spark.stop()