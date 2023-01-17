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
# install pyspark
!pip install pyspark
from pyspark.sql import SparkSession
from pyspark.sql import Row
import pandas as pd
import collections
import datetime
spark = SparkSession.builder.appName("CovidIndia").getOrCreate()
lines = spark.read.csv("../input/covid19-in-india/IndividualDetails.csv", header=True, mode="DROPMALFORMED")
covidData = lines.select("detected_state","status_change_date","current_status")
import pyspark.sql.functions as F
covidData = covidData.withColumn('status_change_date', 
                   F.to_date(F.unix_timestamp(F.col('status_change_date'), 'dd/MM/yyyy').cast("timestamp")))
covidData.dtypes
schemaState  = covidData.cache()
schemaState.createOrReplaceTempView("covidData")
covidData.printSchema()
pd1 = covidData.fillna("unknown", subset=["detected_state", "current_status"])
pd1.show(10)
test = pd1.groupBy("detected_state").pivot("current_status").agg(F.count("current_status")).sort(["Hospitalized"]).na.fill(0)
test.show(40,True)
pd2 = pd1[(pd1["status_change_date"] > "2020-03-24")]
pd3 = pd2.groupBy("detected_state").pivot("current_status").agg(count("current_status")).sort(["Hospitalized"]).na.fill(0)
pd3.show(40,True)
pd3.dtypes
pd3 = pd3.withColumn('zone',
                    F.when((F.col("Hospitalized") > 15),1).otherwise(0)
                    )
pd3.show()
