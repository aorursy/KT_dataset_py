from pyspark.sql.functions import *
from pyspark.sql import SQLContext, Row
import pandas
sqlContext = SQLContext(sc)
bitstampUSD=sqlContext.read.format("com.databricks.spark.csv").options(header='true', inferschema='true').load("file:///home/cloudera/anaconda2/BDM/GrpAss/bitstampUSD.csv")


bitstampUSD.registerTempTable("bitstampUSD")

bitstampUSD = sqlContext.sql("select *,from_unixtime(Timestamp) as `dateTime` from bitstampUSD")

bitstampUSD=bitstampUSD.dropna('any')
bitstampUSD=bitstampUSD.withColumnRenamed("Volume_(BTC)", "VolBTC").withColumnRenamed("Volume_(Currency)", "VolCurrency")


bitstampUSD.registerTempTable("bitstampUSD")
bitstampUSD.printSchema()
bitstampUSD.cache()
yearlyChange_bitstampDF=sqlContext.sql("""(Select openVal._c1 year,Open,Close,Close-Open Change, ((Close-Open)/Open)*100 PercentaceChange from(Select Open,year(dateTime) From bitstampUSD b 
join (select min(dateTime) FstDay from bitstampUSD group by year(dateTime)) dt
on b.dateTime=dt.FstDay) openVal join (Select Close,year(dateTime) From bitstampUSD b 
join (select max(dateTime) LastDay from bitstampUSD group by year(dateTime)) dt
on b.dateTime=dt.LastDay) clsVal on (openVal._c1=clsVal._c1))""")
monthlyChange_bitstampDF=sqlContext.sql("""(Select openVal._c1 year,openVal._c2 month,Open,Close,Close-Open Change,((Close-Open)/Open)*100 PercentaceChange
from(Select Open,year(dateTime),month(dateTime) From bitstampUSD b 
join (select min(dateTime) mnthFstDay from bitstampUSD group by year(dateTime),month(dateTime)) dt
on b.dateTime=dt.mnthFstDay) openVal join (Select Close,year(dateTime),month(dateTime) From bitstampUSD b 
join (select max(dateTime) mnthLastDay from bitstampUSD group by year(dateTime),month(dateTime)) dt
on b.dateTime=dt.mnthLastDay) clsVal on (openVal._c1=clsVal._c1 and openVal._c2=clsVal._c2 ))""")

yearlyChange_bitstampDF.registerTempTable("yearlyChange_bitstampDF")
yearlyValue_bitstampDF=sqlContext.sql("""Select year,Open,Close,MaxValue,MinValue from yearlyChange_bitstampDF OCVal join (select max(High) MaxValue,min(Low) MinValue,year(dateTime) from bitstampUSD group by year(dateTime)) minMax on OCVal.year=minMax._c2""")
import matplotlib.pyplot as plt
import matplotlib as mpl

monthlyWP_bitstampDF=sqlContext.sql("""Select dateTime,Weighted_Price from bitstampUSD main join 
(select max(dateTime) mnthLastDay from bitstampUSD group by year(dateTime),month(dateTime)) dt
on main.dateTime=dt.mnthLastDay order by dateTime""")

monthlyWP_bitstampDF.show()
dailyWP_bitstampDF=sqlContext.sql("""Select dateTime,Weighted_Price from bitstampUSD main join 
(select max(dateTime) EOD from bitstampUSD group by year(dateTime),month(dateTime),dayofmonth(dateTime)) dt
on main.dateTime=dt.EOD order by dateTime""")
dailyWP_bitstampDF.show()
QuarterlyWP_bitstampDF=sqlContext.sql("""select dateTime,Weighted_Price from bitstampUSD main join (Select max(dateTime) EOQ 
from (select *,
case when month(dateTime) between 1 and 3 then 1 
     when month(dateTime) between 4 and 6 then 2 
     when month(dateTime) between 7 and 9 then 3 
     when month(dateTime) between 10 and 12 then 4 
     else 5 end as
 quater from bitstampUSD ) temp group by year(temp.dateTime),temp.quater) dt on main.dateTime=dt.EOQ order by dateTime""")
QuarterlyWP_bitstampDF.show()
yearlyWP_bitstampDF=sqlContext.sql("""Select dateTime,Weighted_Price from bitstampUSD main join 
(select max(dateTime) EOY from bitstampUSD group by year(dateTime)) dt
on main.dateTime=dt.EOY order by dateTime""")

yearlyWP_bitstampDF.show()
dailyWP_bitstampDF=dailyWP_bitstampDF.toPandas()
monthlyWP_bitstampDF=monthlyWP_bitstampDF.toPandas()
QuarterlyWP_bitstampDF=QuarterlyWP_bitstampDF.toPandas()
yearlyWP_bitstampDF=yearlyWP_bitstampDF.toPandas()


plt.subplot(221)
plt.plot(dailyWP_bitstampDF.Weighted_Price, '-', label='By Day')
plt.legend()

plt.subplot(222)
plt.plot(monthlyWP_bitstampDF.Weighted_Price, '-', label='By Months')
plt.legend()

plt.subplot(223)
plt.plot(QuarterlyWP_bitstampDF.Weighted_Price, '-', label='By Quarter')
plt.legend()

plt.subplot(224)
plt.plot(yearlyWP_bitstampDF.Weighted_Price, '-', label='By Year')
plt.legend()

plt.show()
