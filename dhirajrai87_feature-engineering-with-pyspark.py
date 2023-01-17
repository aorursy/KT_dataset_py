# Initializing a Spark session
from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local").appName("Word Count").config("spark.some.config.option","some-value").getOrCreate()
# # Lets us begin by reading the "retail-data/by-day" which is in .csv format
# sales = spark.read.format("csv") \ # here we space the format of the file we intend to read
#         .option("header","true") \ # setting "header" as true will consider the first row as the header of the Dataframe
#         .option("inferSchema", "true") \ # Spark has its own mechanism to infer the schema which I will leverage at this poit of time
#         .load("/data/retail-data/by-day/*.csv") \ # here we specify the path to our csv file(s)
#         .coalesce(5)\
#         .where("Description IS NOT NULL") # We intend to take only those rows in the which the value in the description column is not null
# Lets us begin by reading the "retail-data/by-day" which is in .csv format and save it into a Spark dataframe named 'sales'
sales = spark.read.format("csv").option("header","true").option("inferSchema", "true").load(r"data/retail-data/by-day/*.csv").coalesce(5).where("Description IS NOT NULL")
# Lets us read the parquet files in "simple-ml-integers" and make a Spark dataframe named 'fakeIntDF'
fakeIntDF=spark.read.parquet("/home/spark/DhirajR/Spark/feature_engineering/data/simple-ml-integers")
# Lets us read the parquet files in "simple-ml" and make a Spark dataframe named 'simpleDF'
simpleDF=spark.read.json(r"/home/spark/DhirajR/Spark/feature_engineering/data/simple-ml")
# Lets us read the parquet files in "simple-ml-scaling" and make a Spark dataframe named 'scaleDF'
scaleDF=spark.read.parquet(r"/home/spark/DhirajR/Spark/feature_engineering/data/simple-ml-scaling")
sales.cache()
sales.show()
type(sales)
# Let us see what kind of data do we have in 'fakeIntDF'
fakeIntDF.cache()
fakeIntDF.show()
# Let us import the vector assembler
from pyspark.ml.feature import VectorAssembler
# Once the Vector assembler is imported we are required to create the object of the same. Here I will create an object anmed va
# The above result shows that we have three features in 'FakeIntDF' i.e. int1, int2, int3. Let us create the object va so as to combine the three features into a single column named features
assembler = VectorAssembler(inputCols=["int1", "int2", "int3"],outputCol="features")
# Now let us use the transform method to transform our dataset
assembler.transform(fakeIntDF).show()
# Let us create a sample dataframe for demo purpose

data = [(-999.9,), (-0.5,), (-0.3,), (0.0,), (0.2,), (999.9,)]
dataFrame = spark.createDataFrame(data, ["features"])
from pyspark.ml.feature import Bucketizer
bucketBorders=[-float("inf"), -0.5, 0.0, 0.5, float("inf")]

bucketer=Bucketizer().setSplits(bucketBorders).setInputCol("features").setOutputCol("Buckets")
bucketer.transform(dataFrame).show()
scaleDF.show()
from pyspark.ml.feature import StandardScaler
# Let us create an object of StandardScaler class
Scalerizer=StandardScaler().setInputCol("features").setOutputCol("Scaled_features")
Scalerizer.fit(scaleDF).transform(scaleDF).show(truncate=False)
from pyspark.ml.feature import MinMaxScaler
# Let us create an object of MinMaxScaler class
MinMaxScalerizer=MinMaxScaler().setMin(5).setMax(10).setInputCol("features").setOutputCol("MinMax_Scaled_features")
MinMaxScalerizer.fit(scaleDF).transform(scaleDF).show()
from pyspark.ml.feature import MaxAbsScaler
# Let us create an object of MinAbsScaler class
MinAbsScalerizer=MaxAbsScaler().setInputCol("features").setOutputCol("MinAbs_Scaled_features")
MinAbsScalerizer.fit(scaleDF).transform(scaleDF).show(truncate =False)
from pyspark.ml.feature import ElementwiseProduct
from pyspark.ml.linalg import Vectors

# Let us define a scaling vector 

ScalebyVector=Vectors.dense([10,0.1,-1])

# Let us create an object of the class Elementwise product
ScalingUp=ElementwiseProduct().setScalingVec(ScalebyVector).setInputCol("features").setOutputCol("ElementWiseProduct")
# Let us transform
ScalingUp.transform(scaleDF).show(truncate=False)
from pyspark.ml.feature import Normalizer
# Let us create an object of the class Normalizer product
l1_norm=Normalizer().setP(1).setInputCol("features").setOutputCol("l1_norm")
l2_norm=Normalizer().setP(2).setInputCol("features").setOutputCol("l2_norm")
linf_norm=Normalizer().setP(float("inf")).setInputCol("features").setOutputCol("linf_norm")
# Let us transform
l1_norm.transform(scaleDF).show(truncate=False)
l2_norm.transform(scaleDF).show(truncate=False)
linf_norm.transform(scaleDF).show(truncate=False)
simpleDF.show(5)
from pyspark.ml.feature import StringIndexer
# Let us create an object of the class StringIndexer
lblindexer=StringIndexer().setInputCol("lab").setOutputCol("LabelIndexed")
# Let us transform
idxRes=lblindexer.fit(simpleDF).transform(simpleDF)
idxRes=idxRes.drop("value1","value2")
idxRes.show(5)
from pyspark.ml.feature import IndexToString
LabelReverse=IndexToString().setInputCol("LabelIndexed").setOutputCol("ReverseIndex")
LabelReverse.transform(idxRes).show()
from pyspark.ml.linalg import Vectors
dataln=spark.createDataFrame([(Vectors.dense(1,2,3),1),(Vectors.dense(2,5,6),2),(Vectors.dense(1,8,9),3)]).toDF("features","labels")
dataln.show()
from pyspark.ml.feature import VectorIndexer
VecInd=VectorIndexer().setInputCol("features").setMaxCategories(2).setOutputCol("indexed")
VecInd.fit(dataln).transform(dataln).show()
simpleDF.show()
# Let us encode the "color" feature in the "simpleDF"
from pyspark.ml.feature import StringIndexer,OneHotEncoder
SI=StringIndexer().setInputCol('color').setOutputCol('StrIndexed')
ColorIdx=SI.fit(simpleDF).transform(simpleDF)
ohe=OneHotEncoder().setInputCol('StrIndexed').setOutputCol("oheIndexed")
ohe.transform(ColorIdx).show()
# Let us import the tokenizer
from pyspark.ml.feature import Tokenizer
# Create an object of the Tokenizer class
Tok=Tokenizer().setInputCol("Description").setOutputCol("Tokenized")
sales_tok=Tok.transform(sales).select("Description",'Tokenized')
sales_tok.show()

from pyspark.sql.types import StringType
mydata=['Too Fast For You','For|Your|Eyes|Only','As|a|Matter|of|Fact','As|far|as|I|know','Away|from|Keyboard']
data_txt=spark.createDataFrame(mydata,StringType()).toDF("Text")
data_txt.show()
# Let us import RegexTokenizer class
from pyspark.ml.feature import RegexTokenizer
# Create an object of this class
RegTok=RegexTokenizer().setInputCol('Text').setOutputCol("Tokenized").setPattern("|").setGaps(False)
RegTok.transform(data_txt).show()
from pyspark.sql.types import StringType
mydata=['Too Fast For You You','For Your Eyes Only','As a Matter of Fact','As far as I know','Away from Keyboard']
dataln=spark.createDataFrame(mydata,StringType())
dataln.show()
# Let us import the StopWordsRemover
from pyspark.ml.feature import StopWordsRemover
# Let us import a predefined corpus of stopwords
englishStopWords = StopWordsRemover.loadDefaultStopWords("english")
# Create an object of StopWordsRemover
stops=StopWordsRemover().setStopWords(englishStopWords).setInputCol('Tokenized').setOutputCol("Stops_removed")
stops.transform(sales_tok).show()
# Let us import the NGram class

from pyspark.ml.feature import NGram
uni_gram=NGram().setInputCol("Tokenized").setOutputCol("uni_gram").setN(1)
bi_gram=NGram().setInputCol("Tokenized").setOutputCol("bi_gram").setN(2)
tri_gram=NGram().setInputCol("Tokenized").setOutputCol("tri_gram").setN(3)

# uni_gram.transform(sales_tok.select("Tokenized"))
bi_gram.transform(sales_tok.select("Tokenized")).show(truncate=True)
from pyspark.ml.feature import CountVectorizer
cv = CountVectorizer().setInputCol("Tokenized").setOutputCol("CountVec").setVocabSize(500).setMinTF(1).setMinDF(2)
cv.fit(sales_tok).transform(sales_tok).select("Tokenized","CountVec").show(truncate=False)
from pyspark.ml.feature import Tokenizer
Tok=Tokenizer().setInputCol("value").setOutputCol("Tokenized")
dataln=Tok.transform(dataln)
dataln.show()
from pyspark.ml.feature import CountVectorizer
cv = CountVectorizer().setInputCol("Tokenized").setOutputCol("CountVec").setVocabSize(500).setMinTF(1).setMinDF(1)
cv.fit(sales_tok).transform(dataln).select("Tokenized","CountVec").show(truncate=False)
tfIDFln=sales_tok.where("array_contains(Tokenized,'red')").select("Tokenized").limit(10)
tfIDFln.show(truncate=False)
from pyspark.ml.feature import HashingTF,IDF
tf=HashingTF().setInputCol("Tokenized").setOutputCol("TFOut").setNumFeatures(10000)
idf=IDF().setInputCol("TFOut").setOutputCol("IDFOut").setMinDocFreq(2)
idf.fit((tf.transform(tfIDFln))).transform((tf.transform(tfIDFln))).select("IDFOut").show(1,False)
documentDF=spark.createDataFrame([("Hi I heard about Spark".split(" "), ),("I wish Java could use case classes".split(" "), ),("Logistic regression models are neat".split(" "), )],["text"])
documentDF.show(truncate=False)

from pyspark.ml.feature import Word2Vec
w2v=Word2Vec(vectorSize=5,minCount=0,inputCol="text",outputCol="result")
model=w2v.fit(documentDF)
results=model.transform(documentDF)
# minCount is the minimum number of times a word should appear in the complete document to be included in the vocabulary
# fro more information about various parameters in Word2Vec please refer https://spark.apache.org/docs/2.1.1/api/java/org/apache/spark/mllib/feature/Word2Vec.html
for row in results.collect():
    text, vector = row
    print("Text:[%s] => \n Vector: %s\n" % (",".join(text),str(vector)))

scaleDF.show()
from pyspark.ml.feature import PCA
pca=PCA().setInputCol("features").setOutputCol("PCA_features").setK(2)
scaleDF=pca.fit(scaleDF).transform(scaleDF)

scaleDF.show(truncate=False)
sales=sales.where("Description is NOT NULL").where("CustomerID is NOT NULL").select("Description","CustomerID")
sales.show(10)
# Let us tokenize the "Description"
from pyspark.ml.feature import Tokenizer
Tok=Tokenizer().setInputCol("Description").setOutputCol("Tokenized")
sales=Tok.transform(sales)
sales.show(10)
# Let us countvectorize the Tokenized column
from pyspark.ml.feature import CountVectorizer
cv = CountVectorizer().setInputCol("Tokenized").setOutputCol("CountVec").setVocabSize(500).setMinTF(1).setMinDF(2)
sales=cv.fit(sales).transform(sales)
sales.show(10)
from pyspark.ml.feature import ChiSqSelector
chisq=ChiSqSelector().setFeaturesCol("CountVec").setLabelCol("CustomerID").setNumTopFeatures(5).setOutputCol("Aspects")
sales=chisq.fit(sales).transform(sales)
sales.drop("Description","CustomerID","Tokenized").show(truncate=False)
scaleDF.show()
from pyspark.ml.feature import PolynomialExpansion
PE=PolynomialExpansion().setInputCol("PCA_features").setOutputCol("Poly_features").setDegree(2)
scaleDF=PE.transform(scaleDF)

scaleDF.select('PCA_features','Poly_features').show(10,truncate=False)
