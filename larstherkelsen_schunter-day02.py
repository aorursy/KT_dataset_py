import numpy as np
import pandas as pd
from google.cloud import bigquery
from bq_helper import BigQueryHelper
#For plots
import matplotlib.pyplot as plt
hn = BigQueryHelper('bigquery-public-data', 'hacker_news')
hn_tables = hn.list_tables()
hn_tables
for x in range(0,len(hn_tables)):
    print("Table: "+hn_tables[x])
    a=hn.table_schema(hn_tables[x])
    for y in range(0,len(a)):
        print(a[y])
    print("\n\r")    
hn.head("full",3)
sql1 = """SELECT type, COUNT(id)
    FROM `bigquery-public-data.hacker_news.full`
    GROUP BY type
    """
hn.estimate_query_size(sql1)
types = hn.query_to_pandas_safe(sql1)
types.shape
#Lets create a percent column to see the share of each story type
types['Pct']=types['f0_']/sum(types['f0_'])
print("The story 5 types ordered descending, last coloumn is Percentage of total stories")
#And sort the story types in descending order according to share
types_s=types.sort_values(by=['Pct'],ascending=[False])
print(types_s)
#Creating barplot
height = np.log(types_s['f0_'])
bars = types['type']
y_pos = np.arange(len(bars))
#Create bars and axis names
plt.bar(y_pos, height)
plt.xticks(y_pos, bars)
#Show plot
plt.title("Log values of comment type count.")
plt.show()
sql2 = """SELECT count(ID) as Deleted
    FROM  `bigquery-public-data.hacker_news.comments`
    WHERE deleted is TRUE
    """
hn.estimate_query_size(sql2)
del_com = hn.query_to_pandas_safe(sql2)
del_com
sql3 = """SELECT COUNTIF(deleted is TRUE) as Deleted
    FROM  `bigquery-public-data.hacker_news.comments`
    """
hn.estimate_query_size(sql3)
del_com2 = hn.query_to_pandas_safe(sql3)
del_com2
