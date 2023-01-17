import time
start_time=time.time()
import numpy as np
import operator
from collections import Counter
import pandas as pd 
from wordcloud import WordCloud, STOPWORDS
import numpy as np
import operator
import pandas as pd 
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
plt.style.use('fivethirtyeight')
from google.cloud import bigquery
from bq_helper import BigQueryHelper
client = bigquery.Client()
bq_assistant = BigQueryHelper("bigquery-public-data", "bitcoin_blockchain")
data_f=pd.read_csv("../input/bitstampUSD_1-min_data_2012-01-01_to_2018-06-27.csv")
data_f["Timestamp"]=pd.to_datetime(data_f["Timestamp"], unit='s')
data_f.Timestamp=data_f.Timestamp.apply(lambda x: x.replace(hour=0, minute=0, second=0))
data_f=data_f[["Timestamp", "Weighted_Price"]]
data_f=data_f.drop_duplicates(keep="first")
data_f.head()
data_f=data_f.drop_duplicates(subset='Timestamp', keep="last")
q = """
SELECT  TIMESTAMP_MILLIS((timestamp - MOD(timestamp,
          86400000))) as Timestamp, sum(o.output_satoshis) as output_price from 
    `bigquery-public-data.bitcoin_blockchain.transactions`JOIN
    UNNEST(outputs) as o group by timestamp
"""
print (str(round((bq_assistant.estimate_query_size(q)),2))+str(" GB"))
results3=bq_assistant.query_to_pandas(q)
results3["output_price"]=results3["output_price"].apply(lambda x: float(x/100000000))
results3=results3.sort_values(by="Timestamp")
results3.head()
q = """
SELECT  TIMESTAMP_MILLIS((timestamp - MOD(timestamp,
          86400000))) as Timestamp , count(Timestamp) as output_count from 
    `bigquery-public-data.bitcoin_blockchain.transactions` group by timestamp
"""
print (str(round((bq_assistant.estimate_query_size(q)),2))+str(" GB"))
transaction_count=bq_assistant.query_to_pandas(q)
transaction_count=transaction_count.sort_values(by="Timestamp")
transaction_count.head()
import datetime
def to_unix_time(dt):
    epoch =  datetime.datetime.utcfromtimestamp(0)
    return (dt - epoch).total_seconds() * 1000
import plotly
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.tools as tls
import plotly.graph_objs as go
import plotly.tools as tls
data = [go.Scatter(x=transaction_count.Timestamp, y=transaction_count.output_count/6, name="transaction_count/6"), go.Scatter(x=data_f.Timestamp,y=data_f.Weighted_Price*10, name="BITCOIN_PRICE*10"),
       go.Scatter(x=results3.Timestamp, y=results3.output_price/500, name="transactions price/500")
       
       ]
layout = go.Layout(
    xaxis=dict(
        range=[
        to_unix_time(datetime.datetime(2012, 1, 1)),
            to_unix_time(datetime.datetime(2018, 5, 1))]
    )
)

fig=go.Figure(data=data, layout=layout)
py.iplot(fig)
all_months=["","January","February","March","April","May","June","July","August","September","October","November","December"]
def Bitcoin_Price_avg_monthly(year):
    new_data_f=data_f[(data_f['Timestamp']>datetime.date(year,1,1)) & (data_f['Timestamp']<datetime.date(year+1,1,1))]
    new_data_f["Timestamp"]=new_data_f.Timestamp=data_f.Timestamp.apply(lambda x: x.month)
    
    month_dictionary={}
    for i,x in new_data_f.iterrows():
        if x["Timestamp"] not in month_dictionary:
            month_dictionary[int(x["Timestamp"])]=[]
            
        month_dictionary[int(x["Timestamp"])].append(x["Weighted_Price"])
        
    for i in month_dictionary.keys():
        all_sum=month_dictionary[i]
        
        all_sum=float(sum(all_sum))/float(len(all_sum))
        month_dictionary[i]=all_sum
        
    return month_dictionary
    
def Average_transaction_count_monthly(year, average=True,  mode="Price"):
    if mode=="Price":
        new_data_ff=results3[(results3['Timestamp']>datetime.date(year,1,1)) & (transaction_count['Timestamp']<datetime.date(year+1,1,1))]
    else:
        new_data_ff=transaction_count[(transaction_count['Timestamp']>datetime.date(year,1,1)) & (transaction_count['Timestamp']<datetime.date(year+1,1,1))]
    new_data_ff["Timestamp"]=new_data_ff.Timestamp=transaction_count.Timestamp.apply(lambda x: x.month)
    
    month_dictionary={}
    key="output_price"
    if mode!="Price":
        key="output_count"
    for i,x in new_data_ff.iterrows():
        if x["Timestamp"] not in month_dictionary:
            month_dictionary[int(x["Timestamp"])]=[]
        month_dictionary[int(x["Timestamp"])].append(x[key])
    
    for i in month_dictionary.keys():
        all_sum=month_dictionary[i]
        if not average:
            all_sum=int(sum(all_sum))
        else:
            all_sum=float(sum(all_sum))/float(len(all_sum))
        month_dictionary[i]=int(all_sum)  
    return month_dictionary
    
all_months=["","January","February","March","April","May","June","July","August","September","October","November","December"]
from operator import itemgetter
def Compare_Transaction_Price_Yearly(year, average=True, mode="Price",title=""):
    title=title+" "+str(year)
    new_x=Bitcoin_Price_avg_monthly(year)
    new_x2=Average_transaction_count_monthly(year, average=average, mode=mode)
    new_x=Counter(new_x).most_common()
    new_x2=Counter(new_x2).most_common()
    new_x=sorted(new_x, key=itemgetter(0))
    new_x2=sorted(new_x2, key=itemgetter(0))
    for i in range(0,len(new_x)):
        x=list(new_x[i])
        x[0]=all_months[i+1]
        new_x[i]=tuple(x)
        x=list(new_x2[i])
        x[0]=all_months[i+1]
        new_x2[i]=tuple(x)
    
    x0=[x[0] for x in new_x]
    y0=[x[1] for x in new_x]
    x1=[x[0] for x in new_x2]
    y1=[x[1] for x in new_x2]
    plt.figure(figsize=(12,8))
    plt.subplot(1, 2, 1)
    g = sns.barplot( x=x0, y=y0, palette="winter")
    plt.xticks(rotation=90)
    plt.title('Bitcoin average Price monthly '+str(year))
    plt.xlabel("Month")
    plt.ylabel("Price in USD")

    plt.subplot(1, 2, 2)
    g = sns.barplot( x=x1, y=y1, palette="winter")
    plt.xticks(rotation=90)
    plt.title(title)
    plt.ylabel("Transaction_"+mode)
    plt.xlabel("Month")
    plt.tight_layout()
import warnings
warnings.filterwarnings("ignore")

Compare_Transaction_Price_Yearly(2012,average=False, mode="Count", title="Transaction count of ")
Compare_Transaction_Price_Yearly(2013,average=False, mode="Count", title="Transaction count of ")
Compare_Transaction_Price_Yearly(2014,average=False, mode="Count", title="Transaction count of ")
Compare_Transaction_Price_Yearly(2015,average=False, mode="Count", title="Transaction count of ")
Compare_Transaction_Price_Yearly(2016,average=False, mode="Count", title="Transaction count of ")
Compare_Transaction_Price_Yearly(2017,average=False, mode="Count", title="Transaction count of ")

Compare_Transaction_Price_Yearly(2012, mode="Price", title="Average transaction price of ")
Compare_Transaction_Price_Yearly(2013, mode="Price", title="Average transaction price of ")
Compare_Transaction_Price_Yearly(2014, mode="Price", title="Average transaction price of ")
Compare_Transaction_Price_Yearly(2015, mode="Price", title="Average transaction price of ")
Compare_Transaction_Price_Yearly(2016, mode="Price", title="Average transaction price of ")
Compare_Transaction_Price_Yearly(2017, mode="Price", title="Average transaction price of ")