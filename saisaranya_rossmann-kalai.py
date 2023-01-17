!pip install pyspark 
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from pyspark.sql import SparkSession
spark=SparkSession.builder.appName("DEMO").getOrCreate()
df=spark.read.csv("../input/train.csv",inferSchema=True,header=True)
df.printSchema()
from pyspark.sql.functions import to_date



df=df.withColumn('Date',to_date("Date"))
df.printSchema()
df.dtypes
df.show()
from pyspark.sql.functions import *
#df.groupby(df['DayOfWeek']).count().orderBy('count', ascending=False).show()

df.select([count(when(isnull(c),'c')).alias(c)  for c in df.columns]).show()
df.describe("Store","DayOfWeek","Date","Sales").show()
df.describe("Customers","Open","Promo","StateHoliday","SchoolHoliday").show()
df1=df.filter((df["Store"]==1)&(year(df["Date"])=="2013"))
df1.describe("Store","DayOfWeek","Date","Sales").show()
df1.describe("Customers","Open","Promo","StateHoliday","SchoolHoliday").show()
df1.select("StateHoliday").distinct().show()
df2=df1.filter(df["Open"]!=0).orderBy("Date")
df2.describe("Store","DayOfWeek","Date","Sales").show()
df2.describe("Customers","Open","Promo","StateHoliday","SchoolHoliday").show()
from datetime import *
base = date(2013,1,1)

new_date_list = []

for x in range(0, 365):

    date_list = [base + timedelta(days=x)]

    new_date_list.append(date_list)
new_date_list
test = spark.createDataFrame(new_date_list,['Date'])
test.show()
df3=test.join(df2,["Date"],"leftouter").orderBy("Date")
df3.count()
df3.printSchema()
df3.show()
df3 = df3.withColumn("Store", when(df3.Store.isNull(), lit(1)).otherwise(df3.Store))
df3=df3.withColumn("DayOfWeek", dayofweek("Date"))
df3 = df3.withColumn("Sales", when(df3.Sales.isNull(), lit(0)).otherwise(df3.Sales))
df3=df3.withColumn("Customers", when(df3.Customers.isNull(), lit(0)).otherwise(df3.Customers))
df3=df3.withColumn("Open", when(df3.Open.isNull(), lit(0)).otherwise(df3.Open))
df3=df3 .withColumn("Promo", when(df3.Promo.isNull(), lit(0)).otherwise(df3.Promo))
df3=df3 .withColumn("SchoolHoliday", when(df3.SchoolHoliday.isNull(), lit(0)).otherwise(df3.SchoolHoliday))
df3.show()
df3=df3.withColumn("WeekOfYear", weekofyear("Date"))
df3.show()
df4=df3.groupBy("WeekOfYear").sum("Sales").orderBy("WeekOfYear")
df4.show()
from pyspark.sql import Window

from pyspark.sql import functions as F
df3_cum = df3.withColumn('week_sales', F.sum('Sales').over(Window.partitionBy('WeekOfYear').orderBy('WeekOfYear')

             .rangeBetween(Window.unboundedPreceding, 0)))
df3_cum=df3_cum.orderBy('WeekOfYear')

df3_cum.show()
df3_cum = df3_cum.withColumn('week2_sales', F.sum('Sales').over(Window.partitionBy(df3_cum['WeekOfYear']==0).orderBy('WeekOfYear')

             .rangeBetween(Window.unboundedPreceding, 0)))
df3_cum=df3_cum.orderBy('WeekOfYear')

df3_cum.select("week_sales","week2_sales").show()
x=df

x.count()

x=x.filter(x['Date']<="2013-31-12")

x.filter("open=1").groupBy("Store").agg(count("Date")).orderBy("count(Date)",ascending=False).show()
x=x.withColumn("WeekOfYear", weekofyear("Date"))
x3=x.groupBy("Store","WeekOfYear").sum("Sales").orderBy("Store","WeekOfYear")
x_cum = x.withColumn('week_sales', F.sum('Sales').over(Window.partitionBy('Store','WeekOfYear').orderBy('WeekOfYear')

             .rangeBetween(Window.unboundedPreceding, 0)))
x_cum=x_cum.orderBy('Store','WeekOfYear')
x_cum.select("Store","WeekOfYear","week_sales").show()

x3_cum = x_cum.withColumn('week2_sales', F.sum('Sales').over(Window.partitionBy("Store",x_cum['WeekOfYear']==0).orderBy('WeekOfYear')

             .rangeBetween(Window.unboundedPreceding, 0)))
x3_cum=x3_cum.orderBy('Store','WeekOfYear')
x3_cum.select("Store","WeekOfYear","week_sales","week2_sales").show()

