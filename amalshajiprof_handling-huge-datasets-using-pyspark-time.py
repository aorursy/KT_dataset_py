#import findspark
#findspark.init('/home/davinci/spark-2.4.5-bin-hadoop2.7')
!pip install pyspark  #for installing spark in kaggle kernel
from pyspark.sql import SparkSession
# May take a little while on a local computer
spark = SparkSession.builder.appName("Basics").getOrCreate()
#This is a dataset available online 
# Might be a little slow locally
df = spark.read.json('../input/peoplejson1/people.json')
# Note how data is missing!
df.show()
df.printSchema()
from pyspark.sql.types import StructField,StringType,IntegerType,StructType
data_schema = [StructField("age", IntegerType(), True),StructField("name", StringType(), True)]
final_struc = StructType(fields=data_schema)
df = spark.read.json('../input/peoplejson1/people.json', schema=final_struc)
df.printSchema()
df.columns
df.head(2) #by default shows 1 row
print(df['age'])
print(df.select('age'))
df.select('age').show()
df.select(['age','name']).show()
# Adding a new column which is copied from an old column.
df.withColumn('newage',df['age']).show()
# Renaming a column
df.withColumnRenamed('age','supernewage').show()
df.withColumn('doubleage',df['age']*4).show()
# Let Spark know about the header and infer the Schema types!
df = spark.read.csv('../input/applstock/appl_stock.csv',inferSchema=True,header=True)
df.show()
# Using SQL
df.filter("Close<500").show()
# Using SQL with .select()
df.filter("Close<500").select(['Open','Close']).show()
# Using normal df methods.
df.filter(df["Close"] < 200).show()
# Make sure to add in the parenthesis separating the statements!
df.filter( (df["Close"] < 200) & (df['Open'] > 200) ).show()
# Make sure to add in the parenthesis separating the statements!
df.filter( (df["Close"] < 200) | (df['Open'] > 200) ).show()
# Make sure to add in the parenthesis separating the statements!
df.filter( (df["Close"] < 200) & ~(df['Open'] < 200) ).show()
df.filter(df["Low"] == 197.16).show()
df = spark.read.csv("../input/containsnull1/ContainsNull.csv",header=True,inferSchema=True)
df.show()
# Drop any row that contains missing data
df.na.drop().show()
# Has to have at least 2 NON-null values
df.na.drop(thresh=2).show()
df.na.drop(subset=["Sales"]).show()
df.na.drop(how='any').show()
df.na.drop(how='all').show()
df.na.fill('NEW VALUE').show()
df.na.fill(0).show()
df.na.fill('No Name',subset=['Name']).show()
from pyspark.sql.functions import mean
mean_val = df.select(mean(df['Sales'])).show()


