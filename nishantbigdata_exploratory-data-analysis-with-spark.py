from pyspark import SparkContext

from pyspark.sql import SQLContext

import pandas as pd

from pandas import DataFrame

import numpy as np

import statsmodels.api as sm

import matplotlib.pyplot as plt
#sc = SparkContext('local','example')  # if using locally

sql_sc = SQLContext(sc)



Spark_Full = sc.emptyRDD()

chunk_100k = pd.read_csv('../input/train.csv', chunksize=100000)

# if you have headers in your csv file:

headers = list(pd.read_csv('../input/test.csv', nrows=0).columns)



for chunky in chunk_100k:  #help was taken from stackeaxhange.com

    Spark_Full +=  sc.parallelize(chunky.values.tolist())



data = Spark_Full.toDF(headers)

data.show()
data.printSchema()
data.head(10) #Previewing the data set
# Impute Missing values

data_new = data.fillna(-1)
#Analyzing numerical features

data_new.describe().show()
#Analyze the H1Bs by the status of their visa applications

data_new.select('CASE_STATUS').show()
data_new.select('CASE_STATUS').distinct().count() #to get distinct case status
data_new.registerTempTable("data_new")

data_new.cache()
#Identifying number of visas that are have different case status

data_new.crosstab('EMPLOYER_NAME', 'CASE_STATUS').show()
#Top10 companies getting visa approval (for all the years)

sql_sc.sql("SELECT EMPLOYER_NAME, count(EMPLOYER_NAME) as CERTIFIED_COUNT FROM data_new where CASE_STATUS = 'CERTIFIED' GROUP BY EMPLOYER_NAME order by CERTIFIED_COUNT desc").show(10)
#Top10 companies getting visa approval (for year 2016)

sql_sc.sql("SELECT EMPLOYER_NAME, count(EMPLOYER_NAME) as CERTIFIED_COUNT FROM data_new where CASE_STATUS = 'CERTIFIED' AND YEAR='2016' GROUP BY EMPLOYER_NAME order by CERTIFIED_COUNT desc").show(10)

#Worksites for which most number of visas are approved or certified

sql_sc.sql("SELECT WORKSITE, count(*) as Approved FROM data_new where CASE_STATUS = 'CERTIFIED' GROUP BY WORKSITE order by Approved desc").show(5)
#Worksites for which most number of visas are approved or certified in the year 2016

sql_sc.sql("SELECT WORKSITE, count(*) as Approved FROM data_new where CASE_STATUS = 'CERTIFIED' AND YEAR ='2016' GROUP BY WORKSITE order by Approved desc").show(5)
#TOP 5 JOB TITLE for which visa are approved

sql_sc.sql("SELECT JOB_TITLE, count(*) as Approved FROM data_new where CASE_STATUS = 'CERTIFIED' GROUP BY JOB_TITLE order by Approved desc").show(5)
#TOP 5 JOB TITLE for which visa are approved in the year 2016

sql_sc.sql("SELECT JOB_TITLE, count(*) as Approved FROM data_new where CASE_STATUS = 'CERTIFIED' AND YEAR='2016' GROUP BY JOB_TITLE order by Approved desc").show(5)
# As we can see the job title getting most approvals is programmer_analyst

#Lets check which are the company sending most number of programmed analyst and getting approval on H1B Visa 

sql_sc.sql("SELECT EMPLOYER_NAME,count(*) as Approved FROM data_new where CASE_STATUS = 'CERTIFIED' AND JOB_TITLE ='PROGRAMMER ANALYST' GROUP BY EMPLOYER_NAME order by Approved desc").show(5)
#Lets check which are the company sending most number of programmed analyst and getting approval on H1B Visa in the year 2016

sql_sc.sql("SELECT EMPLOYER_NAME,count(*) as Approved FROM data_new where CASE_STATUS = 'CERTIFIED' AND YEAR='2016' AND JOB_TITLE ='PROGRAMMER ANALYST' GROUP BY EMPLOYER_NAME order by Approved desc").show(5)
#H-1B Salaries Analysis

sql_sc.sql("SELECT EMPLOYER_NAME as businesses, PREVAILING_WAGE as wage, SOC_NAME, JOB_TITLE, YEAR, FULL_TIME_POSITION, CASE_STATUS  FROM data_new where CASE_STATUS ='CERTIFIED' order by PREVAILING_WAGE desc").show(10)

#H-1B Salaries Analysis

sql_sc.sql("SELECT EMPLOYER_NAME as businesses, PREVAILING_WAGE as wage, SOC_NAME, JOB_TITLE, YEAR, FULL_TIME_POSITION, CASE_STATUS  FROM data_new where CASE_STATUS ='CERTIFIED' order by PREVAILING_WAGE desc").show(10)

#Identifying maximim salary by job titles for fulltime position 

sql_sc.sql("SELECT JOB_TITLE ,MAX(PREVAILING_WAGE) as Max_Salary FROM data_new where CASE_STATUS ='CERTIFIED' AND  FULL_TIME_POSITION ='Y' GROUP BY JOB_TITLE ORDER BY Max_Salary DESC").show(10)
#Identifying maximim salary by job titles for fulltime position for 2016

sql_sc.sql("SELECT JOB_TITLE ,MAX(PREVAILING_WAGE) as Max_Salary FROM data_new where CASE_STATUS ='CERTIFIED' AND  FULL_TIME_POSITION ='Y' AND YEAR='2016' GROUP BY JOB_TITLE ORDER BY Max_Salary DESC").show(10)
#Identifying maximum salary by employers for fulltime position 

sql_sc.sql("SELECT EMPLOYER_NAME ,MAX(PREVAILING_WAGE) as Max_Salary FROM data_new where CASE_STATUS ='CERTIFIED' AND  FULL_TIME_POSITION ='Y' GROUP BY EMPLOYER_NAME ORDER BY Max_Salary DESC").show(10)
#Identifying maximum salary by employers for fulltime position for 2016

sql_sc.sql("SELECT EMPLOYER_NAME ,MAX(PREVAILING_WAGE) as Max_Salary FROM data_new where CASE_STATUS ='CERTIFIED' AND  FULL_TIME_POSITION ='Y' AND YEAR='2016' GROUP BY EMPLOYER_NAME ORDER BY Max_Salary DESC").show(10)