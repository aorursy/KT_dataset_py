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

spark = SparkSession.builder.master("local").appName("Retail Data Analytics").config("spark.some.config.option", "some-value").getOrCreate()
features=spark.read.csv('../input/Features data set.csv',header=True,inferSchema=True)

sales=spark.read.csv('../input/sales data-set.csv',header=True,inferSchema=True)

stores=spark.read.csv('../input/stores data-set.csv',header=True,inferSchema=True)
features .show(5)

sales.show(5)

stores.show(2)
from pyspark.sql.functions import *

format='dd/MM/yy'

col=to_date(features['Date'], format).cast('date')

features=features.withColumn('Date',col)

#features=features.withColumn('Date',to_date('Date'))

features=features.withColumn('year',year('Date'))

#features.select('Date').distinct().orderBy('Date').show(200)                              #8190       #182

#features.select('Store','Date').distinct().count()                                        #8190       #45



coll=to_date(sales['Date'], 'dd/MM/yyyy').cast('date')

sales=sales.withColumn('Date', coll)

#sales.select('Date').distinct().orderBy('Date').show(200)                                #421570       #143

#sales.select('Store','Date').distinct().count()                                           #421570            #45





sales=sales.withColumn('year',year('Date'))
#features.show(3)

#features.dtypes

#sales.dtypes

#features.select(features['year']).distinct().show()

#sales.select(sales['year']).distinct().show()

#sales.show(3)
df0=df=sales.join(features, ['Date','Store','IsHoliday'], 'left_outer').drop(features['year'])

df=df.filter(df['year']=='2012')

df36=df.filter(df['Store']=='36')

#df36.show(5)

#df36.printSchema()

df.show(2)
from pyspark.sql.types import *

df36=df36.withColumn('MarkDown1', df36['MarkDown1'].cast(IntegerType()))

df36=df36.withColumn('MarkDown2', df36['MarkDown2'].cast(IntegerType()))

df36=df36.withColumn('MarkDown3', df36['MarkDown3'].cast(IntegerType()))

df36=df36.withColumn('MarkDown4', df36['MarkDown4'].cast(IntegerType()))

df36=df36.withColumn('MarkDown5', df36['MarkDown5'].cast(IntegerType()))



from pyspark.sql.functions import *

#df36.filter(df36['MarkDown1'].rlike('[0-9]')).show()



df36.select([count(when(isnull(mshc),'mshc')).alias(mshc) for mshc in df36.columns]).show()

df36.describe('Weekly_Sales').show()
df36.filter(df['Weekly_Sales'] <= '0').count()                                  # 1 record found

df36=df36.filter(df['Weekly_Sales']>='0')                                        # filtering the remaining records

df36.filter(df['Dept']=='30').show()
df36.filter(df['Weekly_Sales']>='20000').show(5)   

df36.describe('Weekly_Sales').show()

df36.agg(corr('Dept','Weekly_Sales')).show()
pdf36=df36.groupby('Dept','Date').sum('Weekly_Sales').orderBy('Dept')



pdf36.show()



pdf=df36.toPandas()

#15,19,22,27,28,30

import numpy as np

import matplotlib.pyplot as plt
from pyspark.sql.functions import *

pdf.shape

pdf.dtypes
import seaborn as sns

fig,ax = plt.subplots(figsize=(36,50))

sns.boxplot(x='Dept',y='Weekly_Sales',data=pdf,ax=ax)
pdf.groupby(['Dept'])['Weekly_Sales'].sum().plot(kind='bar',figsize=(30,10),fontsize=12,grid='True')
#2010

dfm.count()

dfm.show(2)
dfm.filter(df['MarkDown1'and'MarkDown2'and'MarkDown3'and'MarkDown4'and'MarkDown5']!='NA').count()  #4032 #2921 #3613 #3464 #4050
df11.show()

