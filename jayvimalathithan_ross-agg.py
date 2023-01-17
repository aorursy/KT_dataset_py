# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are avai_lable in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
! pip install pyspark
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("task1").getOrCreate()
train_path = "../input/train.csv"
from pyspark.sql.functions import lag,date_format,udf

from pyspark.sql.types import IntegerType,ArrayType,TimestampType
df = spark.read.csv(train_path,inferSchema=True,header=True)

#df = df.filter("Store = 1")
from pyspark.sql import Window

from pyspark.sql.functions import lag,date_format,col,sum,weekofyear,year,mean,round,when,collect_list,dayofweek,expr
df = df.withColumn("year",year(df["date"]))
df = df.withColumn("week_id",weekofyear(df["date"]))
def date_range(min_date,max_date):

    base = min_date

    numdays =  ((max_date-min_date).days) +1

    date_list = []

    for x in range(0, numdays):

        date_list.append([base + timedelta(days=x)])

    return date_list
col_name="Date"

from pyspark.sql.functions import min,max,col

from datetime import datetime,timedelta

min_date = df.select(min(col(col_name))).collect()[0][0]

max_date = df.select(max(col(col_name))).collect()[0][0]

date_list = date_range(min_date,max_date)

full_date = spark.createDataFrame(date_list,[col_name])
store = df.select("store").distinct()
store_date = store.crossJoin(full_date)
missing_data = store_date.select("store","date").exceptAll(df.select("store","date"))
missing_data = missing_data.select("store",dayofweek("date").alias("DayOfWeek"),"date",expr("0").alias("sales"),expr("0").alias("customers"),expr("0").alias("open"),expr("-1").alias("promo"),expr("1").alias("StateHoliday"),expr("1").alias("SchoolHoliday"),year("date").alias("year"),weekofyear("date").alias("week_id"))
#missing_data.count()
comp = df.union(missing_data)
close = df.filter("open=1").groupBy("year","week_id","store").agg(mean("sales").alias("avg_sales"),round(mean("Customers"),0).alias("avg_customers")).orderBy("store","year","week_id")

#close.show()
new_df = comp.join(close,on=((comp.year==close.year) & (comp.week_id==close.week_id) & (comp.Store==close.store) ))

new_df = new_df.select(comp.Store,comp.DayOfWeek,comp.Date,when(comp.Sales==0,close.avg_sales).otherwise(comp.Sales).alias("sales"), when(comp.Customers==0,close.avg_customers).otherwise(comp.Customers).alias("customers"),comp.Promo,comp.StateHoliday,comp.SchoolHoliday,comp.year,comp.week_id ).orderBy("Store","year","week_id")
#new_df.show()
gdf = new_df.groupBy("year","week_id","store").agg(sum("sales").alias("weekly_sales")).orderBy("year","week_id","store")
#gdf.show(2000)
win = Window.partitionBy("store").orderBy("year","week_id","store")

cuml_sales = gdf.withColumn("cuml_sales",sum("weekly_sales").over(win))

#cuml_sales.show()
cuml_sales.orderBy("store","year","week_id").show(1000)
#new_df.filter("store=1 and week_id=2 and year=2013").select(sum("sales")).show()
#new_df.filter("store=1 and week_id=1 and year=2013").select("store","week_id","year","sales").show()