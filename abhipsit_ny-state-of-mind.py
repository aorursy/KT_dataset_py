#import libraries 

import numpy as np #computing

import pandas as pd #calculations/Data wrangling

import bq_helper

from bq_helper import BigQueryHelper

import seaborn as sns #visualization package

import matplotlib.pyplot as plt
#Get Data

nyc = bq_helper.BigQueryHelper(active_project="bigquery-public-data",

                                   dataset_name="new_york") #Set up connection

bq_assistant = BigQueryHelper("bigquery-public-data", "new_york") #Set up helper

bq_assistant.list_tables() #Show list of available tables 
#Preview of Dataset 

nyc.head('citibike_trips') #Sneak a peek at data 
#Query Data, eliminate nulls for total_amount

query = """select *

          from `bigquery-public-data.new_york.citibike_trips`  

          limit 10000

          """


data = nyc.query_to_pandas_safe(query, max_gb_scanned=10) 
data.head(5) #See head of data 

data.shape  #(1000 rows,23 columns)

data.describe() #See summary of quant data 

data.columns #See list of columns 

data.info() #Check out structure of each column (if integer, string,etc)



pd.isna(data).sum() #See number of NAs by column. We see that birth_year has a sizeable amount of nulls- will need to investigate and handle 



sns.countplot(data['gender'])
g = sns.countplot(data['start_station_name']) #Create basic plot 

ax = sns.countplot(x=data['start_station_name']) #Store labels 



ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right") #Rotate x-axis

plt.show() #Show visual