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

spark=SparkSession.builder.appName('eda').getOrCreate()
features=spark.read.csv('../input/Features data set.csv', header='TRUE')

sales=spark.read.csv('../input/sales data-set.csv',header='TRUE')

stores=spark.read.csv('../input/stores data-set.csv',header='TRUE')
#sales.filter(sales.Dept)

#sales.filter(sales.Dept=='77').show()

#df00.filter(df00.Store=='1').show(100)
df=sales.join(features, ['Store','Date','IsHoliday'], 'left_outer')

df.show(2)

df.count()
from pyspark.sql.functions import *

#df.withColumn('year', year('Date')).show(5)                    #we got null values in year if we take year directly from date, so we need to convert date into std format first



d=to_date(df['Date'], 'dd/MM/yy').cast('date')

df=df.withColumn('Date', d)
#df=df.filter(df['Weekly_Sales']>'0')                                               #420212

#df.count()


df.select(df['Date']).distinct().orderBy('Date', ascending=True).show(2)  

df.select(df['Date']).distinct().orderBy('Date', ascending=False).show(2)  
#df00=df.filter(df.Date=='2010-02-05')

#df00.filter(df00.Store=='1').show(100)
#df01=df.filter(df.Date=='2010-02-19')

#df01.filter(df01.Store=='25').show(100)
df1=df.select(['Store','Dept']).distinct()                             #3323, we cant remove now itself the neg weekly sales, we need to remove before FF only\

                                                                                           #after data gets countinuous



df1.count()
from datetime import *

base = date(2010,2,5)

new_date_list = []

for x in range(0, 1000, 7):

    date_list = [base + timedelta(days=x)]

    new_date_list.append(date_list)



from pyspark.sql.types import *

df12=spark.createDataFrame( new_date_list)

#dfd=spark.createDataFrame('new_date_list', ['Date'])





df12=df12.withColumnRenamed('_1', 'Date')

df12=df12.select(df12['Date']).distinct().orderBy('Date', ascending=True)              #max=2012-10-26   #min=2010-02-05
df12.select(df12['Date']).distinct().orderBy('Date', ascending=True).show(2)  

df12.select(df12['Date']).distinct().orderBy('Date', ascending=False).show(2)  

df12.count()

#df12.show(3)
df2=df12.crossJoin(df1.select('Store','Dept'))

df2.count()    

#df2.orderBy('Store', 'Dept','Date').show(2)
#df2.join(df, ['Store', 'Dept', 'Date'], 'left_outer')

dfm=df2.join(df, ['Date', 'Store', 'Dept'], 'left_outer')

dfm.count()
#dfm.select([count(when(isnull(mshc),'mshc')).alias(mshc) for mshc in dfm.columns]).show()
'''from pyspark.sql.types import *

dfm=dfm.withColumn('Weekly_Sales', dfm['Weekly_Sales'].cast('float'))

dfm=dfm.withColumn('Store', dfm['Store'].cast(IntegerType()))

dfm=dfm.withColumn('Dept', dfm['Dept'].cast(IntegerType()))

dfm=dfm.withColumn('Temperature', dfm['Temperature'].cast('float'))

dfm=dfm.withColumn('Fuel_Price', dfm['Fuel_Price'].cast('float'))

dfm=dfm.withColumn('MarkDown1', dfm['MarkDown1'].cast('float'))

dfm=dfm.withColumn('MarkDown2', dfm['MarkDown2'].cast('float'))

dfm=dfm.withColumn('MarkDown3', dfm['MarkDown3'].cast('float'))

dfm=dfm.withColumn('MarkDown4', dfm['MarkDown4'].cast('float'))

dfm=dfm.withColumn('MarkDown5', dfm['MarkDown5'].cast('float'))

dfm=dfm.withColumn('CPI', dfm['CPI'].cast('float'))

dfm=dfm.withColumn('Unemployment', dfm['Unemployment'].cast('float'))'''
#dfm.filter(dfm.Weekly_Sales.isNull()).orderBy('Date','Store','Dept').show(5)

#df.filter(df.height.isNull()).collect()
from pyspark.sql import *

from pyspark.sql.functions import *

import sys



# define the window

window = Window.partitionBy('Store').orderBy('Date').rowsBetween(0, sys.maxsize)



# define the forward-filled column

filled_column = last(dfm['Weekly_Sales'], ignorenulls=True).over(window)



# do the fill

dfm= dfm.withColumn('Weekly_Sales', filled_column)



window = Window.partitionBy('Store').orderBy('Date').rowsBetween(-sys.maxsize,0)

filled_column = last(dfm['Temperature'], ignorenulls=True).over(window)

dfm= dfm.withColumn('Temperature', filled_column)



window = Window.partitionBy('Store').orderBy('Date').rowsBetween(-sys.maxsize,0)

filled_column = last(dfm['Fuel_Price'], ignorenulls=True).over(window)

dfm= dfm.withColumn('Fuel_Price', filled_column)





window = Window.partitionBy('Store').orderBy('Date').rowsBetween(-sys.maxsize,0)

filled_column = last(dfm['IsHoliday'], ignorenulls=True).over(window)

dfm= dfm.withColumn('IsHoliday', filled_column)



window = Window.partitionBy('Store').orderBy('Date').rowsBetween(-sys.maxsize,0)

filled_column = last(dfm['CPI'], ignorenulls=True).over(window)

dfm= dfm.withColumn('CPI', filled_column)



window = Window.partitionBy('Store').orderBy('Date').rowsBetween(-sys.maxsize,0)

filled_column = last(dfm['Unemployment'], ignorenulls=True).over(window)

dfm= dfm.withColumn('Unemployment', filled_column)

dfm.select([count(when(isnull(mshc),'mshc')).alias(mshc) for mshc in dfm.columns]).show()
dfm.columns
dfm.show()
import pandas as pd
#dfm.to_pandas()

#pandas_df = some_df.toPandas()

pdf = dfm.toPandas()
from IPython.display import HTML

import pandas as pd

import numpy as np



pdf.to_csv('submission.csv')



def create_download_link(title = "Download CSV file", filename = "data.csv"):  

    html = '<a href={filename}>{title}</a>'

    html = html.format(title=title,filename=filename)

    return HTML(html)



# create a link to download the dataframe which was saved with .to_csv method

create_download_link(filename='submission.csv')
