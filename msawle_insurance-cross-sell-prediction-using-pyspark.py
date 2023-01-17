# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
! pip install pyspark
from pyspark.sql import SparkSession

spark = SparkSession\
        .builder\
        .appName("Python Spark regression example")\
        .config("spark.some.config.option", "random")\
        .getOrCreate()
df = spark.read.format("csv").\
    options(inferSchema = True, header = True).load("/kaggle/input/health-insurance-cross-sell-prediction/train.csv")
df.show(5, False)

df.printSchema()
df.select(['Gender', 'Age', 'Policy_Sales_Channel']).show(5, False) # Method 1

df.select(df["Age"], df["Gender"], df["Policy_Sales_Channel"]).show(10, False) # Method 2

from pyspark.sql.functions import col # Method 3 - By import SQL function col
df.select(col("Age"), col("Gender"), col("Policy_Sales_Channel")).show(5, False)
df.filter(df["Age"] > 30 ).filter(df["Age"] < 45).show(5) # Selecting subset using chain of Filter of option

df.filter((df["Age"] > 30) & (df["Age"] < 45 )).show(5, False) # Selcting subset by using AND between the criteria
pandasDF = df.toPandas()

pandasDF.head(10)
ageCount = df.groupBy(["Age", "Response"]).count().toPandas()
avgPremiumByAge = df.groupBy(["Age", "Response"]).avg("Annual_Premium").withColumnRenamed("avg(Annual_Premium)","Avg_Annual_Premium").toPandas()
maxPremiumByAge = df.groupBy(["Age", "Response"]).max("Annual_Premium").withColumnRenamed("max(Annual_Premium)","Max_Annual_Premium").toPandas()
import matplotlib.pyplot as plt
import seaborn as sns


fig, (ax1, ax2, ax3) = plt.subplots(nrows = 3, ncols = 1, sharex = False, figsize = (20, 24))
sns.barplot(ageCount["Age"], ageCount["count"], hue = ageCount["Response"], ax = ax1)
ax1.set_title("Count of People Surveyed by their Age and Response", fontsize = 20)
ax1.set_xlabel("Age",fontsize = 16); ax1.set_ylabel("Count",fontsize = 16)
sns.barplot(avgPremiumByAge["Age"], avgPremiumByAge["Avg_Annual_Premium"], hue = avgPremiumByAge["Response"],ax = ax2)
ax2.set_xlabel("Age",fontsize = 16); ax2.set_ylabel("Average Annual Premium",fontsize = 16)
ax2.set_title("Average insurance premium by Age and Response", fontsize = 20)
sns.barplot(maxPremiumByAge["Age"], maxPremiumByAge["Max_Annual_Premium"],hue = maxPremiumByAge["Response"],ax = ax3)
ax3.set_xlabel("Age",fontsize = 16); ax3.set_ylabel("Maximum Annual Premium",fontsize = 16)
ax3.set_title("Maximum insurance premium by Age and Response", fontsize = 20)


plt.show()
vehicleStats = df.groupBy(["Vehicle_Age", "Response"]).count().toPandas()
premiumStats = df.groupBy(["Vehicle_Damage", "Vehicle_Age","Response"]).avg("Annual_Premium").withColumnRenamed("avg(Annual_Premium)","Avg_Annual_Premium").toPandas()
fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, sharex = False, figsize = (24, 8))
sns.barplot(vehicleStats["Vehicle_Age"], vehicleStats["count"], hue = vehicleStats["Response"], ax = ax1, ci = None)
ax1.set_title("Count of Respondents by Response and the Age of Vehicle owned", fontsize = 20)
ax1.set_xlabel("Vehicle Age",fontsize = 16); ax1.set_ylabel("Count",fontsize = 16)
sns.barplot(premiumStats["Vehicle_Age"], premiumStats["Avg_Annual_Premium"], hue = premiumStats["Vehicle_Damage"],ax = ax2, ci = None)
ax2.set_xlabel("Vehicle Age",fontsize = 16); ax2.set_ylabel("Average Annual Premium",fontsize = 16)
ax2.set_title("Average insurance premiumby Age and Vehicle Condition", fontsize = 20)


plt.show()
splits= [0,10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

from pyspark.ml.feature import Bucketizer

bucketizer = Bucketizer(splits=splits, inputCol="features", outputCol="bucketedFeatures")

bucketizer = Bucketizer(splits = splits, inputCol = "Age", outputCol = "ageBuckets")
bucketedData = bucketizer.transform(df)

bucketedData.show(10, False)


ageBucketSummary = bucketedData.groupBy(["ageBuckets", "Response"]).count().toPandas()
channelSummary = df.groupBy(["Policy_Sales_Channel", "Response"]).count().toPandas()
channelSummary = channelSummary[channelSummary["count"] >= 1000 ].sort_values("count", ascending = False)
fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, sharex = False, figsize = (24, 8))
sns.barplot(ageBucketSummary["ageBuckets"], ageBucketSummary["count"], hue = ageBucketSummary["Response"], ax = ax1, ci = None)
ax1.set_title("Count of People Surveyed by Response and the Age of Vehicle owned", fontsize = 20)
ax1.set_xlabel("Age of Respondents",fontsize = 16); ax1.set_ylabel("Count",fontsize = 16)
sns.barplot(channelSummary["Policy_Sales_Channel"], channelSummary["count"], hue = channelSummary["Response"],ax = ax2, ci = None)
ax2.set_xlabel("Policy Sales Channel",fontsize = 16); ax2.set_ylabel("Average Annual Premium",fontsize = 16)
ax2.set_title("Average insurance premium paid by Age and Response", fontsize = 20)

plt.show()