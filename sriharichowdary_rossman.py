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

spark=SparkSession.builder.getOrCreate()

df1=spark.read.csv('../input/store.csv', header= True)

df2=spark.read.csv('../input/train.csv', header=True)



from pyspark.sql.functions import *

df2=df2.withColumn('Date',to_date('Date'))

#df2.printSchema()

#df2.describe('Sales').show()

#df2.filter(df2['Sales'].between(0,200)).count()

#df2.filter(df2['Sales']<1).count()

#df2.select(df2['Sales'].isNull()).count()

#df2.filter(df2['customers']==0).count()

#df2.count()



#dfopen=df2.filter(df2['Open']==1)



#dfopen.describe('Sales').show()

#dfopen.show()

#dfopen.filter(dfopen['Sales']==0).count()             #54 rows

#dfopen.filter(dfopen['Customers']==0).count()         #52 rows



#dfopen.count()

#dfopen.show(2)

#dfopen.select(dfopen['year']==2015)     #2015=196032, 2014=310417, 2013=337943

#dfopen.select('Date').distinct().count()   #942,   365*3=1095

#dfopen.describe('Date').show()

#df2.printSchema()

#df2.filter(df2['store']==2).filter(year(df2['Date'])==2014).show()

#df2.filter(year(df2['Date'])==2013).show()

#&(year(df2['Date'])=='2015').

#df1=df.filter((df["Store"]==1)&(year(df["Date"])=="2013"))
from pyspark.sql.functions import year

df2=df2.withColumn('year',year('Date') )



df001=df2.filter(df2['Store']==1)

df001=df001.filter(df001['year']==2014).orderBy('Date')

df002=df001.select(df001['Date'])

df002=df002.withColumnRenamed('Date','Dates')

#df002.printSchema()



df001.orderBy('Date').show()

df002.orderBy('Date').show()

#df1=df.filter((df["Store"]==1)&(year(df["Date"])=="2013"))

#df001.select('Date').distinct().count()           # we have 212 dates in this dataframe

#df001.select(df001['Store']).distinct().show()    #verifing we selected only one store

#df001.select(df001['year']).distinct().show()     #verifing we selected only one year
from pyspark.sql.functions import *

#df001.select([count(when(isnull(c),'c')).alias(c)  for c in df001.columns]).show()

df001.select([count(when(isnull(mshc),'mshc')).alias(mshc) for mshc in df001.columns]).show()
df001=df001.filter(df001['Open']==1)

df001.show()



#df001.filter(df001['Open']==1).orderBy('Date').show()

#df001.select('Date').count()    

#212-175=37 days store was closed
df001=df001.join(df002, df001.Date == df002.Dates, 'outer')

df001=df001.drop('Date')

df001.orderBy('Dates').show()
from pyspark.sql.functions import dayofweek

df001=df001.na.fill({'Store':1,'year':2014,'Open':0,'Promo':0})



#df001.na.fill({'Sales':mean(Sales)}).show()



df001.show()

from pyspark.sql.types import IntegerType

df001 =df001.withColumn("Sales", df001["Sales"].cast(IntegerType()))

#data_df = data_df.withColumn("drafts", data_df["drafts"].cast(IntegerType()))

#df001.printSchema()





#df001.na.fill({'Sales':mean('Sales')}).show()

from pyspark.sql.window import Window

import pyspark.sql.functions as func

import sys



df001=df001.withColumn("Sales", func.last('Sales', True).over(Window.partitionBy('Store').orderBy('Dates').rowsBetween(-sys.maxsize, 0)))

df001=df001.withColumn("Customers", func.last('Customers', True).over(Window.partitionBy('Store').orderBy('Dates').rowsBetween(-sys.maxsize,0)))

df001=df001.withColumn('DayOfWeek',dayofweek('Dates'))

df001.orderBy('Dates').show()
df001=df001.drop('StateHoliday','SchoolHoliday')

#df001.show()



df001=df001.withColumn('week',weekofyear('Dates'))

df003=df001.groupBy("week").sum("Sales").orderBy('week')



df003=df003.withColumnRenamed('week', 'Weekly')

#df003.printSchema()

df001.show()

df003.show()



#df001.withColumn('Weekly_Sales',(df001.groupBy("week").sum("Sales"))).orderBy('week').show()

df001=df001.join(df003, df001.week == df003.Weekly, 'outer')

df001=df001.drop('Weekly')

df001=df001.withColumnRenamed('sum(Sales)', 'Weekly_Sales').orderBy('Dates')

df001.show()
df001=df001.drop('Weekly')

df001=df001.withColumnRenamed('sum(Sales)', 'Weekly_Sales').orderBy('Dates')

df001.show()
from pyspark.sql import Window

from pyspark.sql import functions as F

windowval = (Window.partitionBy('Store').orderBy('Dates').rangeBetween(Window.unboundedPreceding, 0))

windowval1 = (Window.partitionBy('week').orderBy('Dates').rangeBetween(Window.unboundedPreceding, 0))

df001 = df001.withColumn('WeekWise_Cummulative', F.sum('Sales').over(windowval1))

df001 = df001.withColumn('DayWise_Cumulative', F.sum('Sales').over(windowval))



#df001.orderBy('Dates').show()

df001.select('Store','year','Dates','Sales','week','Weekly_Sales','DayOfWeek','week','WeekWise_Cummulative','Dates','DayWise_Cumulative').show()
from pyspark.sql.types import *

cSchema = StructType([StructField("Date", DateType())])

import datetime

Start = '2015-01-01'

stop = '2015-01-06'

start = datetime.datetime.strptime(start, '%Y-%m-%d')

end = datetime.datetime.strptime(stop, '%Y-%m-%d')

step = datetime.timedelta(days=1)

while start <= end:

     print('Date:',start.date())

     start += step
base = date(2013,1,1)

new_date_list = []

for x in range(0, 365):

    date_list = [base + timedelta(days=x)]

    new_date_list.append(date_list)
