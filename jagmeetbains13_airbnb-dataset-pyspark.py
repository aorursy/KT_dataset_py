#  This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas_profiling



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#installing pyspark version 2.3.4

!pip install pyspark==2.3.4
import pyspark
from pyspark import SparkContext
#Starting SparkContext

sc = SparkContext("local","jagmeet")
sqlContext = pyspark.SQLContext(sc)
sc.version
from pyspark.sql.types import *
#Defining own schema

#This is done so that there would not be any conflict between datatypes

data_schema = [StructField('id', IntegerType(), True), 

               StructField('name', StringType(), True),

               StructField('host_id', IntegerType(), True),

               StructField('host_name', StringType(), True),

               StructField('neighbourhood_group', StringType(), True),

               StructField('neighbourhood', StringType(), True),

               StructField('latitude', FloatType(), True),

               StructField('longitude', FloatType(), True),

               StructField('room_type', StringType(), True),

               StructField('price', IntegerType(), True),

               StructField('minimum_nights', IntegerType(), True),

               StructField('number_of_reviews', IntegerType(), True),

               StructField('last_review', DateType(), True),

               StructField('reviews_per_month', FloatType(), True),

               StructField('calculated_host_listings_count', IntegerType(), True),

               StructField('availability_365', IntegerType(), True),

              ]
final_struc = StructType(data_schema)
#Reading csv file into spark dataframe using our schema 

df = sqlContext.read.csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv', header = True,schema = final_struc,mode="DROPMALFORMED")
df.printSchema()
import seaborn as sns

sns.set()
import matplotlib.pyplot as plt

%matplotlib inline
import pandas as pd
import pyspark.sql.functions as func
#groupby, aggregating, sorting and then converting the dataframe to pandas

#conversion to pandas is just for plotting purposes

#Calculating mean of price for all the distinct neighbourhood_group

df_neigh_gr_pd = df.groupBy("neighbourhood_group").agg(func.mean("price").alias("mean_price")).orderBy('mean_price', ascending=False).toPandas().head(10)
df_neigh_gr_pd
plt.figure(figsize=(25, 10))

sns.barplot(x="neighbourhood_group", y="mean_price", data=df_neigh_gr_pd)
#This will calculate mean price of each distinct neighbourhood

df_neigh_pd = df.groupBy('neighbourhood').agg(func.mean("price").alias("mean_price")).orderBy('mean_price', ascending=True).toPandas().head(20)
df_neigh_pd
plt.figure(figsize=(25, 10))

sns.barplot(x="neighbourhood", y="mean_price", data=df_neigh_pd)
#calculaitng sum of reviews for each host_id

df_host_popularity = df.groupby('host_id').agg(func.sum('number_of_reviews').alias('sum_review'))
#calculating total reviews for complete data

total_reviews = df.select(func.sum('number_of_reviews')).head()[0]
total_reviews
#import udf

from pyspark.sql import udf
import pyspark.sql.functions as F
#Defining function for withcolumn operation and then udf

def func_divide(a1,a2):

  return a1*100.0/float(a2)
divide_func_sp = F.udf(func_divide,FloatType())
divide_func_sp
#withColumn adds new column and applied the funcition to make new column

df_host_popularity = df_host_popularity.withColumn('total_sum_reviews', F.lit(total_reviews))
df_host_popularity.show()
#creating popularity_index column by diving each host_id reviews by total no. of reviews

df_host_popularity = df_host_popularity.withColumn('popularity_index',divide_func_sp('sum_review','total_sum_reviews'))
df_host_popularity.orderBy('popularity_index', ascending=False).show(10)
df_host_popularity = df_host_popularity.select('host_id','popularity_index')
df_host_popularity.show()
df_host_popularity.orderBy('popularity_index', ascending=False).show(10)
#merging the popularity_index to the main dataframe

df = df.join(df_host_popularity, "host_id", "left")
df.show(2)
#spark filter and sorting

df.filter(df['minimum_nights'] > 10).orderBy('popularity_index', ascending=False).select(df['neighbourhood']).show(20)
df_popular_regions = df.groupby('neighbourhood').agg(func.sum('popularity_index').alias('pop_reg'))
df_popular_regions.show()
df_popular_regions = df_popular_regions.orderBy('pop_reg', ascending=False).toPandas().head(10)
df_popular_regions.head()
plt.figure(figsize=(16, 6))

sns.barplot(x='neighbourhood', y='pop_reg', data=df_popular_regions)
#Summing no. of reviews for each neighbourhood

df_neighbourhood_sum = df.groupby('neighbourhood').agg(func.sum('number_of_reviews').alias('sum_reviews_ne'))
df_neighbourhood_sum.show(5)
df_neighbourhood_sum.head()
df_neighbourhood_host_sum = df.groupby('host_id','neighbourhood').agg(func.sum('number_of_reviews').alias('sum_reviews_id_ne'))
df_neighbourhood_host_sum.show()
#merging column

df_neighbourhood_host_merged = df_neighbourhood_host_sum.join(df_neighbourhood_sum,'neighbourhood','left')
df_neighbourhood_host_merged.filter(df['neighbourhood'] == 'Williamsburg').orderBy('host_id', ascending=True).show(10)
def func_divide(a1,a2):

  if(a2!=0):

    ans = a1*100.0/float(a2)

  else:

    ans = 0

  return ans
divide_func_sp = F.udf(func_divide,FloatType())
df_neighbourhood_host_merged = df_neighbourhood_host_merged.withColumn('host_neighbourhood_popularity',divide_func_sp('sum_reviews_id_ne','sum_reviews_ne'))
df_neighbourhood_host_merged.show()
df_neighbourhood_host_merged.orderBy('host_neighbourhood_popularity', ascending=False).show(20)
df_neighbourhood_host_merged = df_neighbourhood_host_merged.select('host_id','neighbourhood','host_neighbourhood_popularity')
df_neighbourhood_host_merged.show(5)
#To validate results, for baychester the sum of all the host_neighbourhood_popularity = 100.0

df_neighbourhood_host_merged.filter(df_neighbourhood_host_merged['neighbourhood']=='Baychester').show()
df = df.join(df_neighbourhood_host_merged,['host_id','neighbourhood'],'left')
df.filter(df['host_neighbourhood_popularity']>90).show(2)
#Filtering only private room and entire room

df_neighbourhood_room_type = df.filter(df['room_type'] != 'Shared room')
df_neighbourhood_room_type = df_neighbourhood_room_type.groupby('neighbourhood','room_type').agg(func.sum('price').alias('price_sum'))
df_neighbourhood_room_type.show()
#pivot operation 

df_neighbourhood_room_type = df_neighbourhood_room_type.groupby('neighbourhood').pivot('room_type').sum('price_sum')
df_neighbourhood_room_type.show()
#Maximum revenue in Private Rooms

df_neighbourhood_room_type.orderBy('Private room', ascending=False).show(1)
df_min = df_neighbourhood_room_type.orderBy('Private room', ascending=True)
#Minimum revenue in Private Rooms after filetering out null values

df_min[df_min['Private room'].isNotNull()].show(1)
##minimum revenue in Entire home/apt Rooms

df_neighbourhood_room_type.orderBy('Entire home/apt', ascending=True).filter(df_neighbourhood_room_type['Entire home/apt'].isNotNull()).show(1)
###maximum revenue in Entire home/apt Rooms



df_neighbourhood_room_type.orderBy('Entire home/apt', ascending=False).show(1)
df_neighbourhood_room_type_mean = df.filter(df['room_type'] != 'Shared room')
df_neighbourhood_room_type_mean = df_neighbourhood_room_type_mean.groupby('neighbourhood','room_type').agg(func.mean('price').alias('average_region_price'))
df_neighbourhood_room_type_mean_pivot = df_neighbourhood_room_type_mean.groupby('neighbourhood').pivot('room_type').sum('average_region_price')
df_neighbourhood_room_type_mean_pivot.show()
df_neighbourhood_room_type_mean_pivot = df_neighbourhood_room_type_mean_pivot.toPandas()
#Plotting 

plt.figure(figsize=(30, 10))

labels = []

import matplotlib.pyplot as plt

ax=df_neighbourhood_room_type_mean_pivot.sort_values('Private room', ascending=False).head().plot(kind='bar', width = 0.5)

ax.set_xlabel('neighbourhood', fontsize = 20)

ax.set_ylabel('average_price', fontsize = 20)

fig_size = plt.rcParams["figure.figsize"]

fig_size[0] = 20

fig_size[1] = 10

a = [item.get_text() for item in ax.get_xticklabels()]  #to get labels as they are coming as index

for i in a:

  labels.append(df_neighbourhood_room_type_mean_pivot['neighbourhood'][int(i)])

ax.set_xticklabels(labels)  #setting xticks as neighbourhood name

ax = plt.gca()

for tick in ax.xaxis.get_major_ticks():

    tick.label.set_fontsize(20) 

    tick.label.set_rotation('horizontal')

for tick in ax.yaxis.get_major_ticks():

    tick.label.set_fontsize(16)     

plt.rcParams["figure.figsize"] = fig_size

plt.legend(prop={'size':'15'})

plt.show(ax)
df = df.join(df_neighbourhood_room_type_mean,['neighbourhood','room_type'],'left')
df.filter(df['neighbourhood'] == 'Kensington').show()
#Filtering out null values for price

temp1 = df.filter(df['price'].isNotNull())
#Filtering out null values for average_region_price

temp1 = temp1.filter(temp1['average_region_price'].isNotNull())
#Defining function and converting it to udf function for with column operation

def func_divide1(a1,a2):

  if(type(a1)=='NoneType' or type(a2)=='NoneType'):

    ans = 0

  elif(a1!=0):

    ans = ((a1-a2)*100.0)/float(a1)

  else:

    ans = 0

  return ans



divide_func_sp1 = F.udf(func_divide1,FloatType())
#withcolumn operation

df = temp1.withColumn('region_price_margin',divide_func_sp1('price','average_region_price'))
df.show(1)