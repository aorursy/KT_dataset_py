
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, FloatType
import pandas as pd
import numpy as np

#Location and the file type
file_location = "/FileStore/tables/OnlineRetail.csv"
file_type = "csv"

#We can make the read operation(when DAG turns) much faster by defining the schema
schema = StructType([
  StructField("InvoiceNo",IntegerType(),True),
  StructField("StockCode",StringType(),True),
  StructField("Description",StringType(),True),
  StructField("Quantity",IntegerType(),True),
  StructField("InvoiceDate",StringType(),True),
  StructField("UnitPrice",FloatType(),True),
  StructField("CustomerID",IntegerType(),True),
  StructField("Country",StringType(),True)
])
df = spark.read.option("header","true") \
                         .format("csv") \
                         .schema(schema) \
                         .load("/FileStore/tables/OnlineRetail.csv")

display(df.head(5))
#The schema structures of the dataframe
df.printSchema()
#Columns in the data
df.columns
#Data Cleaning and Data Manipulation
from pyspark.sql.functions import count

#Returning the counts from all the columns to check that how many column contain how much null values(NaN values)
def my_count(df_in):
    df_in.agg( *[ count(c).alias(c) for c in df_in.columns ] ).show()
    
    
my_count(df)
#Since as we can see that our count values are not same and in CustomerID column we have some null values so we need to handle these null values
#First going with the basic method as our dataset is sufficiently large and if the model fails to generalize good on the dataset we will use other missing vaue handling techniques

df = df.dropna(how="any")
my_count(df)

#So as now we can see that the null values data points have been removed from our data
#Converting InvoiceDate coulmn to UTC time stamp format(New column NewInvoiceDate Created)

from pyspark.sql.functions import to_utc_timestamp, unix_timestamp, lit, datediff, col
timeFormat = "MM/dd/yy HH:mm"
df = df.withColumn("NewInvoiceDate",to_utc_timestamp(unix_timestamp(col("InvoiceDate"),timeFormat).cast("timestamp"),"UTC"))
df.show(5)
#Calculating the total price
#For Calculating the monetary value we will be requiring the total amount that the customer has spent(so we need to get the price spent  by customer and which is equal to Quantity*unit_price_of_quantity)
from pyspark.sql.functions import round
df = df.withColumn("TotalPrice",round(df.Quantity*df.UnitPrice,2))
df.show(5)
#Calculating the time difference
import pandas as pd
from pyspark.sql.functions import mean,min,max,sum,datediff,to_date
date_max = df.select(max("NewInvoiceDate")).toPandas()
current = to_utc_timestamp(unix_timestamp(lit(str(date_max[0][0])),"yy-MM-dd HH:mm").cast("timestamp"),"UTC")

#Calculating the Duration(Duration is another important attribute for RFM analysis which tell how often did customer purchase)(From how much time he hasn't purchased)

df = df.withColumn("Duration",datediff(lit(current),"NewInvoiceDate"))
df.show(5)
#Building Recency,Frequency and Monetary attribute corresponding to the customers ID(Customers)

recency = df.groupBy("CustomerID").agg(min("Duration").alias("Recency"))
frequency = df.groupBy("CustomerID","InvoiceNo").count().groupBy("CustomerID").agg(count("*").alias("Frequency"))
monetary = df.groupBy("CustomerID").agg(round(sum("TotalPrice"), 2).alias("Monetary"))
rfm = recency.join(frequency,"CustomerID",how ="inner").join(monetary,"CustomerID",how ="inner")
def describe_pd(df_in, columns, deciles=False):
    if deciles:
        percentiles = np.array(range(0, 110, 10))
    else:
        percentiles = [25, 50, 75]
    percs = np.transpose([np.percentile(df_in.select(x).collect(),percentiles) for x in columns])
    percs = pd.DataFrame(percs,columns=columns)
    percs["summary"] = [str(p) + "%"for p in percentiles]
    spark_describe = df_in.describe().toPandas()
    new_df = pd.concat([spark_describe, percs],ignore_index=True)
    new_df = new_df.round(2)
    return new_df[["summary"] + columns]
cols = ["Recency","Frequency","Monetary"]
describe_pd(rfm,cols,1)
#Use obove function describe_pd or either use this below piece of code for short statistical inference
cols = ["Recency","Frequency","Monetary"]
rfm.select(cols).describe().show()
#Using the quantile for defining the R,F,M values between 1 and 4
#According to the magnitudes we have assigned values between 1 to 4 to the attributes

def RScore(x):
  #Smaller value of x(Recency) tells us that the particular customer has done some activity(like buying something or using some product) recently and contrary larger the value of x will give some inference that customer wasn't involved in activity from a long time
  if x <= 16:
    return 1
  elif x<= 50:
    return 2
  elif x<= 143:
    return 3
  else:
    return 4

def FScore(x):
  #Smaller the value of x(Frequency) tell that the customer is not involved in activities frequently and for customer with high value of x denotes that customer is involved in Frequent activities
  if x <= 1:
    return 4
  elif x <= 3:
    return 3
  elif x <= 5:
    return 2
  else:
    return 1

def MScore(x):
  #Smaller the value of x(Monetary value) tells us that customer activities cost is not much(has not spent much money on buying some product etc) and contrary higher value of x denotes that customer has spent a lot of money on activities
  if x <= 293:
    return 4
  elif x <= 648:
    return 3
  elif x <= 1611:
    return 2
  else:
    return 1

#A customer can have any of the permutation of these values corresponding to their activities
  

#For each and every value of R,F,M we will pass them through the lambda function in corresponding R_udf,F_udf,M_udf
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, DoubleType

R_udf = udf(lambda x: RScore(x), StringType())
F_udf = udf(lambda x: FScore(x), StringType())
M_udf = udf(lambda x: MScore(x), StringType())

#RFM segamentation
from pyspark.sql.functions import concat

rfm_seg=rfm.withColumn("r_seg", R_udf("Recency"))
rfm_seg=rfm_seg.withColumn("f_seg", F_udf("Frequency"))
rfm_seg=rfm_seg.withColumn("m_seg", M_udf("Monetary"))
#Display is inbuilt function Databricks environment to show the dataframe
display(rfm_seg.head(5))
col_list=["r_seg","f_seg","m_seg"]

#RFM score is nothing but the concatenated R,F,M values
rfm_seg=rfm_seg.withColumn("RFMScore",concat(*col_list))
display(rfm_seg.sort("RFMScore").head(5))
#Statistical summary for each RFM score(Mapping of RFM score against average R,F,M values)
display(rfm_seg.groupBy("RFMScore").agg({"Recency":"mean","Frequency":"mean","Monetary":"mean"} ).sort(["RFMScore"]).head(5))
