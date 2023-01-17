import types
import pandas as pd

pandasRAW = pd.read_csv('../input/master.csv')
pandasRAW.head()

pandasRAW.drop("HDI for year", inplace=True, axis=1)
pandasRAW.rename(columns={'gdp_for_year($)': 'gdp_year', 'gdp_per_capita ($)': 'gdp_capita'}, inplace=True)
pandasRAW.dtypes
pandasRAW.isnull().sum()
pandasRAW[pandasRAW.duplicated(keep=False)]

import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
sparkRAW=sqlContext.createDataFrame(pandasRAW)
sparkRAW.registerTempTable("suicides")
import matplotlib.pyplot as plt
plt.close('all')
query1=spark.sql("select year,round((sum(suicides_no)/sum(population))*100000,1) as suicide_percentage from suicides group by year order by year")
query1.show(n=10)
pandas1=query1.toPandas()
pandas1.cumsum()
plt.figure()
pandas1.plot.line(x='year',y='suicide_percentage')

query2=spark.sql("select country,year,round((sum(suicides_no)/sum(population))*100000,2) sums from suicides where year=2004 group by year,country order by year,sums DESC LIMIT 5")
query2.show(n=10)
pandas2=query2.toPandas()
pandas2.plot.bar(x='country',y='sums')
query3=spark.sql("select country,sum(suicides_no) total_suicides from suicides group by country order by total_suicides DESC LIMIT 5")
query3.show(n=10)
pandas3=query3.toPandas()
pandas3.plot.bar(x='country',y='total_suicides')
from pyspark.sql.functions import udf
from pyspark.sql.types import *
query3=spark.sql("select country,sum(suicides_no) total_suicides from suicides where year=2015 group by country order by total_suicides DESC")
def valueToCategory(value):
   if   value >0 and value<=500: 
    return 'Low'
   elif value >500 and value<=1000:
    return 'Moderate'
   elif value >1000:
    return 'High'
   else: 
    return 'n/a'

udfValueToCategory = udf(valueToCategory, StringType())
query4= query3.withColumn("category", udfValueToCategory("total_suicides"))
query4.registerTempTable("temp")
finalQuery=spark.sql("select category,count(*) counts from temp group by category order by counts desc ")
finalQuery.show(n=10)
pandas4=finalQuery.toPandas()
pandas4.head
pandas4.plot(kind='pie',y='counts',labels=pandas4.category,autopct='%.2f')
query5=spark.sql("select age,sum(suicides_no) sums from suicides where year=2016 group by age order by sums desc")
query5.show(n=50)
pandas5=query5.toPandas()
pandas5.plot.bar(x='age',y='sums')

query6=spark.sql("select sex,sum(suicides_no) sums from suicides group by sex order by sums desc")
query6.show(n=50)
pandas6=query6.toPandas()
pandas6.plot(kind='pie',labels=pandas6.sex,y='sums',autopct='%.2f')
query7_1=spark.sql("select sex x,age,sum(suicides_no) sums from suicides where sex='male' group by age,sex order by sums desc limit 1")
query7_2=spark.sql("select sex x,age,sum(suicides_no) sums from suicides where sex='female' group by age,sex order by sums desc limit 1")
final=query7_1.union(query7_2)
final.show(n=10)
pandas7=final.toPandas()
pandas7.plot.pie(y='sums',labels=['male_35-54','female_35-54'],autopct='%.2f')
query8=spark.sql("select year,country,round(sum(suicides_no)/sum(population)*100000,2) sums,gdp_capita from suicides where country='Russian Federation' group by year,country,gdp_capita order by gdp_capita desc ")
def valueToCategory(value):
   if   value >0 and value<=10000: 
    return 'Low GDP'
   elif value >10000:
    return 'High GDP'
   else: 
    return 'n/a'

udfValueToCategory = udf(valueToCategory, StringType())
query8= query8.withColumn("category", udfValueToCategory("gdp_capita"))
query8.registerTempTable("temp")
final=spark.sql("select category,round(avg(sums),2) avg_suicide_rate from temp group by category")
final.show(n=10)
pandas8=final.toPandas()
pandas8.plot(kind='pie',labels=pandas8.category,y='avg_suicide_rate',autopct='%.2f')
final.registerTempTable("temp1")
decrease=spark.sql("select round(max(avg_suicide_rate)-min(avg_suicide_rate),2) avg_decrease_rate from temp1")
pandas8_result=decrease.toPandas()
pandas8_result.head()

query9=spark.sql("select generation,sum(suicides_no) sums from suicides where year=2000 group by generation order by sums desc")
pandas9=query9.toPandas()
pandas9.plot(kind='pie',labels=pandas9.generation,y='sums',autopct='%.2f')
query10=spark.sql("select sex,max(suicides_no) from suicides where sex='male' AND year=1995 group by sex")
pandas10=query10.toPandas()
pandas10.head()