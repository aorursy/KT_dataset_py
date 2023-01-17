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
df=spark.read.csv('../input/submission (1).csv', header=True)
df.count()
df=df.drop('_c0')
df.show(2)
from pyspark.sql.functions import *

import datetime

d=to_date(df['Date'], 'dd-MM-yy').cast('date')

df=df.withColumn('Date', d)

from pyspark.sql.types import *

df=df.withColumn('Weekly_Sales', df['Weekly_Sales'].cast('float'))

df=df.withColumn('Store', df['Store'].cast(IntegerType()))

df=df.withColumn('Dept', df['Dept'].cast(IntegerType()))

df=df.withColumn('Temperature', df['Temperature'].cast('float'))

df=df.withColumn('Fuel_Price', df['Fuel_Price'].cast('float'))

df=df.withColumn('MarkDown1', df['MarkDown1'].cast('float'))

df=df.withColumn('MarkDown2', df['MarkDown2'].cast('float'))

df=df.withColumn('MarkDown3', df['MarkDown3'].cast('float'))

df=df.withColumn('MarkDown4', df['MarkDown4'].cast('float'))

df=df.withColumn('MarkDown5', df['MarkDown5'].cast('float'))

df=df.withColumn('CPI', df['CPI'].cast('float'))

df=df.withColumn('Unemployment', df['Unemployment'].cast('float'))
df.select(df['Date']).distinct().orderBy('Date', ascending=True).show(1)

df.select(df['Date']).distinct().orderBy('Date', ascending=False).show(1)
df.filter(df['Weekly_Sales']<='0').count()
df.filter(df['Weekly_Sales'].isNull()).show()
from pyspark.sql import *

from pyspark.sql.functions import *

import sys



# define the window

window = Window.partitionBy('Store').orderBy('Date').rowsBetween(-sys.maxsize,0)



# define the forward-filled column

filled_column = last(df['Weekly_Sales'], ignorenulls=True).over(window)



# do the fill

df= df.withColumn('Weekly_Sales', filled_column)

df.select([count(when(isnull(mshc),'mshc')).alias(mshc) for mshc in df.columns]).show()    #

#df.filter()

df.select(df['Weekly_Sales']).distinct().orderBy('Weekly_Sales', ascending=True).show(5)

df.select(df['Weekly_Sales']).distinct().orderBy('Weekly_Sales', ascending=False).show(5)

df.describe('Weekly_Sales').show()

import pandas

pdf=df.toPandas()

#pdf = dfm.toPandas()



pdf.describe()
Q1=3652.150

Q3=21461.949

IQR=Q3-Q1

IQR



Low_outlier=Q1-(1.5*IQR)

high_outlier=Q3+(1.5*IQR)





print(Low_outlier)

print(high_outlier)
import matplotlib.pyplot as plt



import seaborn as sns

fig,ax = plt.subplots(figsize=(18,9))

sns.boxplot(x='Dept',y='Weekly_Sales',data=pdf,ax=ax)
#is is not correct aasumtion

import matplotlib.pyplot as plt



import seaborn as sns

fig,ax = plt.subplots(figsize=(18,9))

sns.boxplot(x='Dept',y='Weekly_Sales',data=pdf,ax=ax)
pdf.groupby(['Store'])['Weekly_Sales'].mean().plot(kind='bar',figsize=(30,10),fontsize=12,grid='True')


pdf.groupby(['Dept'])['Weekly_Sales'].mean().plot(kind='bar',figsize=(30,10),fontsize=12,grid='True')
pdf.groupby(['Store'])['Temperature'].mean().plot(kind='bar',figsize=(30,10),fontsize=12,grid='True')
#pdf.groupby(['Store','Dept'])['Weekly_Sales'].mean().plot(kind='bar',figsize=(100,10),fontsize=12,grid='True')
'''import seaborn as sns; sns.set()

import matplotlib.pyplot as plt

#pdf = sns.load_dataset("pdf")

ax = sns.lineplot(x="Date", y="Weekly_Sales", sizes = (18,9),data=pdf)



#ax = sns.lineplot(x="timepoint", y="signal", hue="event",err_style="bars", ci=68, data=pdf)'''
"""import pandas as pd

#Weekly_Sales = pdf.Series(Weekly_Sales, name="Weekly_Sales")

#ax = sns.distplot(Weekly_Sales)



fig, ax = plt.subplots(2,figsize=(14,14))



sns.distplot(pdf['Weekly_Sales'],bins=10, ax=ax[0])"""
'''fig, ax = plt.subplots(2,figsize=(14,14))



sns.distplot(pdf['Temperature'],bins=10, ax=ax[0])

#sns.distplot(pdf['Temperature'], bins=10, ax=ax[1])'''
