# installing pyspark in container

!pip install pyspark
# Import SparkSession from pyspark.sql

from pyspark.sql import SparkSession



# Create my_spark

spark_ex = SparkSession.builder.getOrCreate()



# Print my_spark

print(spark_ex)
# Print the tables in the catalog

print(spark_ex.catalog.listTables())
#registering a table in catalog

import pandas as pd

df1 = pd.read_csv("../input/titanic/train.csv")

df2=df1.iloc[:,0:4]

spark_df = spark_ex.createDataFrame(df2)

spark_df.registerTempTable("sample_table")

#spark_df.show()
# Don't change this query

query = "FROM sample_table SELECT * LIMIT 10"



# Get the first 10 rows of flights

titanic10 = spark_ex.sql(query)



# Show the results

titanic10.show()
# Don't change this query

query = "SELECT Pclass,COUNT(*) as Survived_Count FROM sample_table GROUP BY Pclass"



# Run the query

titanic_counts = spark_ex.sql(query)



# Convert the results to a pandas DataFrame

pd_counts = titanic_counts.toPandas()



# Print the head of pd_counts

print(pd_counts.head())
import numpy as np

pd_temp = pd.DataFrame(np.random.random(10))

spark_temp = spark_ex.createDataFrame(pd_temp)

print(spark_ex.catalog.listTables())

spark_temp.createOrReplaceTempView("temp")



# Examine the tables in the catalog again

print(spark_ex.catalog.listTables())
# Don't change this file path

file_path = "../input/titanic/test.csv"



# Read in the titanic data

titanic = spark_ex.read.csv(file_path,header=True)



# Show the data

titanic.show()
# Add Fare x100 Column

titanic = titanic.withColumn("Fare x 100",titanic.Fare*100)

titanic.show()