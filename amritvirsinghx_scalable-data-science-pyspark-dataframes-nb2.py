!pip install pyspark
from pyspark.sql import SparkSession

spark_ex = SparkSession.builder.getOrCreate()

print(spark_ex)
# Don't change this file path

file_path = "../input/titanic/test.csv"



# Read in the titanic data

titanic = spark_ex.read.csv(file_path,header=True)



# Show the data

titanic.show()
titanic = titanic.filter(titanic.Age.isNotNull()).filter(titanic.SibSp.isNotNull()).filter(titanic.Fare.isNotNull()).filter(titanic.Parch.isNotNull())

titanic.show()
titanic.count()
titanic.filter("Fare > 200").show()
titanic.filter(titanic.Fare > 200).show()
# selecting some columns

selected1 = titanic.select("Pclass", "Sex", "Age")

selected1.show()
# trying in a different way

temp = titanic.select(titanic.Pclass, titanic.Sex, titanic.Age)

temp.show()
# first filter

filterA = titanic.Pclass == "2"



# second filter

filterB = titanic.Sex == "female"



# Filter the data, first by filterA then by filterB

selected2 = temp.filter(filterA).filter(filterB)

selected2.show()
titanic=titanic.withColumn("Parch", titanic.Parch.cast("Int")).withColumn("SibSp", titanic.SibSp.cast("Int")).withColumn("Fare", titanic.Fare.cast("Int"))
#lets check schema 

titanic.printSchema()
# Lets calculate avg fare per member (total memebers= sibilings+ parents + 1(that person))

avg_fare = (titanic.Fare/(titanic.SibSp+titanic.Parch+1)).alias("avg_fare")



#lets check the dataframe

avg = titanic.select("PassengerId","Fare",avg_fare)

avg.show()
# We can do the same using expression

fare_avg = titanic.selectExpr("PassengerId", "Fare", "Fare/(SibSp+Parch+1) as avg_fare")

fare_avg.show()
# Find the min fare from females

titanic.filter(titanic.Sex == "female").groupBy().min("Fare").show()



# Find the max fare from males

titanic.filter(titanic.Sex == "male").groupBy().max("Fare").show()
# avg fare of male travelling in Pclass

titanic.filter(titanic.Pclass=="3").filter(titanic.Sex=="male").groupBy().avg("Fare").show()
# Group by Sex

by_sex = titanic.groupBy("Sex")



# Number of flights each plane made

by_sex.count().show()


# Group by origin

by_class = titanic.groupBy("Pclass")



# Average duration of flights from PDX and SEA

by_class.avg("Fare").show()
import pyspark.sql.functions as F



# Group by Sex and Pclass

by_month_dest = titanic.groupBy("Sex","Pclass")



# Average departure delay by month and destination

by_month_dest.avg("Fare").show()
# Standard deviation of departure delay

by_month_dest.agg(F.stddev("Fare")).show()
#creating table1

t1 = titanic.select("PassengerId", "Name", "Sex")

t1.show()
#creating table2

t2 = titanic.select("PassengerId", "Age", "Fare")

t2.show()
t2 = t2.withColumnRenamed("Fare", "New_Fare")

t2.show()
joined=t1.join(t2,on="PassengerId",how="leftouter")

joined.show()