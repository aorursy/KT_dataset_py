!pip install pyspark
import pyspark
from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[4]") \
                    .appName('SparkByExamples.com') \
                    .getOrCreate()
flightData2015 = spark\
                 .read\
                 .option("inferSchema","true")\
                 .option("header","true")\
                 .csv("../input/dataset/data/flight-data/csv/2015-summary.csv")
                
flightData2015.take(3)
flightData2015.sort("count").take(3) #sort is wide and read is narrow transformation actions
flightData2015.sort("count").explain()
spark.conf.set("spark.sql.shuffle.partitions","5")
flightData2015.sort("count").take(2)
flightData2015.createOrReplaceTempView("flight_data_2015")
sql_code = spark.sql("SELECT DEST_COUNTRY_NAME,count(1) from flight_data_2015 GROUP BY DEST_COUNTRY_NAME")
sql_code.show(5)
#same thing in python
python_code = flightData2015.groupBy("DEST_COUNTRY_NAME").count()
python_code.show(5)
sql_code.explain() == python_code.explain()
spark.sql("select max(count) from flight_data_2015").take(1) #sql
from pyspark.sql.functions import *
flightData2015.select(max("count")).take(1) #pyhton
#any sql query
maxsql = spark.sql("""
                   SELECT DEST_COUNTRY_NAME, sum(count) destinamtion_total from flight_data_2015
                   group by DEST_COUNTRY_NAME order by sum(count) desc limit 5 """)
maxsql.show()
maxpython = flightData2015.groupBy("DEST_COUNTRY_NAME").sum("count").withColumnRenamed("sum(count)","destinamtion_total")\
            .sort(desc("destinamtion_total")).limit(5).show()
flightData2015.groupBy("DEST_COUNTRY_NAME").sum("count").withColumnRenamed("sum(count)","destinamtion_total")\
            .sort(desc("destinamtion_total")).limit(5).explain()
#./bin/spark-submit -- master local ./examples/src/main/python/pi.py 10
staticDataframe = spark.read.format("csv").option("header","true").option("inferschema","true")\
                  .load("../input/dataset/data/retail-data/by-day/*.csv")
staticDataframe.createOrReplaceTempView("retail_data")
staticschema = staticDataframe.schema
staticschema
sql_select = spark.sql("select * from retail_data limit 3").show()
#python
total_cust_per_session = staticDataframe.selectExpr("CustomerID","(Quantity*UnitPrice) as total_cost","InvoiceDate")\
                         .groupby(col("customerID"),window(col("InvoiceDate"),"1 day"))\
                         .sum("total_cost")\
                         .show(10)
streamingDataFrame = spark.readStream.schema(staticschema).option("maxFilesPerTrigger",1)\
                     .format("csv")\
                     .option("header","true").load("../input/dataset/data/retail-data/by-day/*.csv")
streamingDataFrame.isStreaming
#writting same code again
total_cust_per_day = streamingDataFrame.selectExpr("CustomerID","(Quantity*UnitPrice) as total_cost","InvoiceDate")\
                         .groupby(col("customerID"),window(col("InvoiceDate"),"1 day"))\
                         .sum("total_cost")
                    
#it is in  lazy execution mode 
total_cust_per_day.writeStream.format("memory").queryName("customer_purchase").outputMode("complete").start()
#Now query the table
spark.sql("""select * from customer_purchase order by `sum(total_cost)` desc""").show(5)
#Now query the table
spark.sql("""select * from customer_purchase order by `sum(total_cost)` desc""").take(2)
total_cust_per_day.writeStream.format("console").queryName("customer_purchase_2").outputMode("complete").start()
#create df
spark.range(100)
df = spark.range(100).toDF("Number")
df.select(df["Number"] + 3).show(2)
df = spark.read.format("json").load("../input/dataset/data/flight-data/json/2015-summary.json") #dataframe 
df.printSchema()
df.schema
#creating schema manually
from pyspark.sql.types import *
myschema = StructType([
                      StructField("DEST_COUNTRY_NAME",StringType(),True),
                      StructField("ORIGIN_COUNTRY_NAME",StringType(),True),
                      StructField("count",LongType(),False)
])

df = spark.read.format("json").schema(myschema).load("../input/dataset/data/flight-data/json/2015-summary.json")
#accessing columns
spark.read.format("json").load("../input/dataset/data/flight-data/json/2015-summary.json").columns
#see the first row
df.first()
#filtering col
df.filter(col("count")<2).take(3)
df.filter(col("count")<2).where(col("DEST_COUNTRY_NAME")!="United States").take(2)
#unique value
df.select("ORIGIN_COUNTRY_NAME","DEST_COUNTRY_NAME").distinct().count()
#doing random sampleing
seed = 2
fraction = 0.7
withReplacement = False
df.sample(withReplacement,fraction,seed).count()

dataframe = df.randomSplit([0.5,0.5],seed)
dataframe[0] == dataframe[1]
dataframe[0].count() == dataframe[1].count()
#union of rows
from pyspark.sql import Row 
schema = df.schema
newRow = [
    Row("New Country","Some Country", 3),
    Row("New Country2","Some Country2",13),
    Row("New Country3","Some Country3",30)
    
]
parallelizedRows = spark.sparkContext.parallelize(newRow)
newdf = spark.createDataFrame(parallelizedRows,schema)

#union
df.union(newdf).where("count=1").where(col("ORIGIN_COUNTRY_NAME") != "United States").show()
#sorting
df.sort("count").show()
df.orderBy(col("count"),col("ORIGIN_COUNTRY_NAME")).show()
#partitions
df.rdd.getNumPartitions()
#repartition
df.repartition(5)
df = spark.read.format("csv").option("header","true").option("inferSchema","true").load("../input/dataset/data/retail-data/by-day/2010-12-01.csv")
df.printSchema()
df.createOrReplaceTempView("dfTable")
df.show(5)
#some filtering boolean operator
df.where(col("InvoiceNo") != 536365).select(col("Description")).show(3)
df.where(col("InvoiceNo") != 536365).show(5,False)
#filtering with variable
pricefilter = col("UnitPrice") > 2.5
df.where(df.StockCode.isin("DOT")).where(pricefilter).show()
#addition of column
pricefilter = col("UnitPrice") > 600
DOTfilter = col("StockCode") == "DOT"
Descfilter = instr(col("Description"),"POSTAGE") >= 1
df.withColumn("isExpensive",DOTfilter & (pricefilter | Descfilter)).where("isExpensive").select("InvoiceNo","isExpensive","UnitPrice").show(5)
artificalquantity = pow(col("Quantity")*col("UnitPrice"),2) + 7
df.select(expr("CustomerId"),artificalquantity.alias("Quantity")).show(3)
df.selectExpr("CustomerId","(POWER((Quantity*UnitPrice),2)+5) as realquantity").show(2)
df.select(corr("Quantity","UnitPrice")).show()
df.describe().show()
df.stat.crosstab("StockCode","Quantity").show()
df.stat.freqItems(["StockCode","Quantity"]).show()
df.select(monotonically_increasing_id()).show(3)
#strings
df.select(initcap(col("Description"))).show()
spark.sql("SELECT initcap(Description) from dfTable ").show(5)
df.select(col("Description"),upper(col("Description")),lower(upper(col("Description")))).show(3)
#sql
spark.sql("SELECT Description,lower(Description),upper(Description) from dfTable").show(3)
df.select(ltrim(lit("   Hello   ")).alias("ltrim")).show(5)
df.select(lpad(lit("Hello   "),3," ").alias("ltrim")).show(5)
regexp_string = "BLACK|GREEN|RED|BLUE|WHITE"
df.select(regexp_replace(col("Description"),regexp_string,"COLOR")).alias("color_clean").show(3)
#translate
df.select(translate(col("Description"),"LEET","1337"),col("Description")).show(2)

regexp_string = "(BLACK|GREEN|RED|BLUE|WHITE)"
df.select(regexp_extract(col("Description"),regexp_string,1).alias("color_clean"),col("Description")).show(3)
df.printSchema()
dateDF = spark.range(10).withColumn("today",current_date()).withColumn("now",current_timestamp())
dateDF.createOrReplaceTempView("dateTable")
dateDF.printSchema()
dateDF.select(date_sub(col("today"),5),date_add(col("today"),5)).show(1)
#coalesce
df.select(coalesce(col("Description"),col("CustomerId"))).show(1)
df.select(count("StockCode")).show()
df.select(approx_count_distinct("StockCode",0.15)).show()
!pip install tensorframes h5py pyspark
# Import libraries
from pyspark.sql import SparkSession

# Creating SparkSession
spark = (SparkSession
            .builder
            .config('spark.jars.packages', 'databricks:spark-deep-learning:1.5.0-spark2.4-s_2.11')
            .getOrCreate()
)

# Import Spar-Deep-Learning-Pipelines
import sparkdl
# Import Spar-Deep-Learning-Pipelines
import sparkdl


img_dir = "../input/dataset/data/deep-learning-images"
image_df = sparkdl.image.imageIO.Image.open("../input/dataset/data/deep-learning-images/daisy/100080576_f52e8ee070_n.jpg")
image_df
image_df = spark.read.format("image").load(img_dir)
image_df.printSchema()
from pyspark.sql.functions import lit
tulip_dfs = spark.read.format("image").load(img_dir + "/tulips").withColumn("label",lit(1))
daisy_dfs = spark.read.format("image").load(img_dir + "/daisy").withColumn("label",lit(0))
tulip_train,tulip_test = tulip_dfs.randomSplit([0.6,0.4])
daisy_train,daisy_test = daisy_dfs.randomSplit([0.6,0.4])
train_df = tulip_train.unionAll(daisy_train)
test_df = tulip_test.unionAll(daisy_test)
# train logistic regression on features generated by InceptionV3:
featurizer = DeepImageFeaturizer(inputCol="image", outputCol="features", modelName="InceptionV3")
# Build logistic regression transform
lr = LogisticRegression(maxIter=20, regParam=0.05, elasticNetParam=0.3, labelCol="label")
# Build ML pipeline
p = Pipeline(stages=[featurizer, lr])
# Build our model
p_model = p.fit(train_df)
# Run our model against test dataset
tested_df = p_model.transform(test_df)
# Evaluate our model
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print("Test set accuracy = " + str(evaluator.evaluate(tested_df.select("prediction", "label"))))
from sparkdl import DeepImagePredictor
predictor = DeepImagePredictor(inputCol="image",outputCol="predicted_labels",modelName="InceptionV3",
                              topK=10)
prediction_df = predictor.transform(image_df)
import pandas as pd
df = pd.read_csv("../input/dataset/data/flight-data/csv/2010-summary.csv")
sparkDf = spark.createDataFrame(df)
sparkDf.show(3)
pandas_df=sparkDf.toPandas()
pandas_df.head()
bikestation = spark.read.option("header","true").csv("../input/dataset/data/bike-data/201508_station_data.csv")
tripData = spark.read.option("header","true").csv("../input/dataset/data/bike-data/201508_trip_data.csv")
#creating vertice and edge
stationVertice = bikestation.withColumnRenamed("name","id").distinct()
tripEdges = tripData.withColumnRenamed("Start Station","src").withColumnRenamed("End Station","dst")

!pip install graphframes
from graphframes import GraphFrame
stationGraph = GraphFrame(stationVertice,tripEdges)
stationGraph.cache()
