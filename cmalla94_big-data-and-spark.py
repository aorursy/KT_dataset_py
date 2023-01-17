import os

input_dir = '../input'

os.listdir(input_dir)

file = 'traffic-collision-data-from-2010-to-present.csv'

path = os.path.join(input_dir,file)

print(path)
!pip install pyspark
import sys

from pyspark.sql import SparkSession, functions, types

 

spark = SparkSession.builder.appName('example 1').getOrCreate()

spark.sparkContext.setLogLevel('WARN')



assert sys.version_info >= (3, 5) # make sure we have Python 3.5+

assert spark.version >= '2.3' # make sure we have Spark 2.3+



data = spark.read.csv(path, header=True,

                      inferSchema=True)

data.show()
# let's see the schema

data.printSchema()
# select some columns

data.select(data['Crime Code'], data['Victim Age']).show()
# filter the data

data.filter(data['Victim Age'] < 40).select('Victim Age', 'Victim Sex').show()
# write to a json file

json_file = data.filter(data['Victim Age'] < 40).select('Victim Age', 'Victim Sex')

json_file.write.json('json_output', mode='overwrite')
!ls json_output
# a few more things



# perform a calculation on a column and rename it

data.select((data['Council Districts']/2).alias('CD_dividedBy2')).show()



# rename columns 

data.withColumnRenamed('Victim Sex', 'Gender').select('Gender').show()



# drop columns and a cleaner vertical format for the top 10 

d = data.drop('Neighborhood Councils')

d.show(n=10, truncate=False, vertical=True)