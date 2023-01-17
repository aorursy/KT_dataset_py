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
!pip install pyspark
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('spark-dataframe-demo').getOrCreate()
from pyspark.sql import *

 

Employee = Row("firstName", "lastName", "email", "salary")

 

employee1 = Employee('Basher', 'armbrust', 'bash@edureka.co', 100000)

employee2 = Employee('Daniel', 'meng', 'daniel@stanford.edu', 120000 )

employee3 = Employee('Muriel', None, 'muriel@waterloo.edu', 140000 )

employee4 = Employee('Rachel', 'wendell', 'rach_3@edureka.co', 160000 )

employee5 = Employee('Zach', 'galifianakis', 'zach_g@edureka.co', 160000 )

 

print(Employee[0])

 

print(employee3)

 

department1 = Row(id='123456', name='HR')

department2 = Row(id='789012', name='OPS')

department3 = Row(id='345678', name='FN')

department4 = Row(id='901234', name='DEV')
departmentWithEmployees1 = Row(department=department1, employees=[employee1, employee2, employee5])

departmentWithEmployees2 = Row(department=department2, employees=[employee3, employee4])

departmentWithEmployees3 = Row(department=department3, employees=[employee1, employee4, employee3])

departmentWithEmployees4 = Row(department=department4, employees=[employee2, employee3])

departmentsWithEmployees_Seq = [departmentWithEmployees1, departmentWithEmployees2]

dframe = spark.createDataFrame(departmentsWithEmployees_Seq)

display(dframe)

dframe.show()
fifa_df = spark.read.csv("../input/PlayerNames.csv", inferSchema = True, header = True)

 

fifa_df.show()
fifa_df.printSchema()
fifa_df.columns # Column Names

 

fifa_df.count() # Row Count

 

len(fifa_df.columns) # Column Count
fifa_df.describe('Name').show()

fifa_df.describe('url').show()
fifa_df.select('Name','url').show()
fifa_df.select('Name','url').distinct().show()