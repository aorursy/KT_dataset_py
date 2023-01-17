import hashlib

import os

''' I have made this simple class to save the data i am getting from query, So that If I

Ever Run this program again I dont have to use my quota but get it directly from the

place where I have saved it... :##'''

class DataSaver:

    def __init__(self, bq_assistant):

        self.bq_assistant=bq_assistant

        

    def Run_Query(self, query, max_gb_scanned=1):

        hashed_query=''.join(query.split()).encode("ascii","ignore")

        query_hash=hashlib.md5(hashed_query).hexdigest()

        query_hash+=".csv"

        if query_hash in os.listdir(os.getcwd()):

            print ("Data Already present getting it from file")

            return pd.read_csv(query_hash)

        else:

            data=self.bq_assistant.query_to_pandas_safe(query, max_gb_scanned=max_gb_scanned)

            data.to_csv(query_hash, index=False,encoding='utf-8')

            return data
import time

start_time=time.time()

import numpy as np

import operator

from collections import Counter

import pandas as pd 

from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt

from plotly import __version__

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly as py

import plotly.graph_objs as go

init_notebook_mode(connected=True)

plt.rcParams['figure.figsize']=(12,5)

import numpy as np

import operator

import pandas as pd 

from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt

from plotly import __version__

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly as py

import plotly.graph_objs as go

init_notebook_mode(connected=True)

plt.rcParams['figure.figsize']=(12,5)

from google.cloud import bigquery

from bq_helper import BigQueryHelper

client = bigquery.Client()

bq_assistant = BigQueryHelper("bigquery-public-data", "bitcoin_blockchain")

def satoshi_to_bitcoin(satoshi):

    return float(float(satoshi)/ float(100000000))

test_data=bq_assistant.head("transactions")





bq=DataSaver(bq_assistant)





def Create_Bar_plotly(list_of_tuples, items_to_show=40, title=""):

    #list_of_tuples=list_of_tuples[:items_to_show]

    data = [go.Bar(

            x=[val[0] for val in list_of_tuples],

            y=[val[1] for val in list_of_tuples]

    )]

    layout = go.Layout(

    title=title,xaxis=dict(

        autotick=False,

        tickangle=290 ),)

    fig = go.Figure(data=data, layout=layout)

    py.offline.iplot(fig)
x=test_data.iloc[2].inputs[0]

x["input_pubkey_base58"]="1KEH32noJFb3tiBbWzLZo9nie6C4VhNP7Y"

print (x)

#MODIFIED THE INPUT BECAUSE HEAD VALUE KEEPS ON CHANGING
x=test_data.iloc[3].outputs[0]

x["output_satoshis"]=4000000

x["output_pubkey_base58"]="1bonesF1NYidcd5veLqy1RZgF4mpYJWXZ"

print (x)

print ("-"*0)

x["output_satoshis"]=1159000

x["output_pubkey_base58"]="1KEH32noJFb3tiBbWzLZo9nie6C4VhNP7Y"

print (x)

#MODIFIED THE INPUT BECAUSE HEAD VALUE KEEPS ON CHANGING
print (test_data.iloc[2].outputs[0])
q = """

SELECT  o.output_pubkey_base58, sum(o.output_satoshis) as output_sum from 

    `bigquery-public-data.bitcoin_blockchain.transactions`JOIN

    UNNEST(outputs) as o 

    where o.output_pubkey_base58 not in (select i.input_pubkey_base58

    from UNNEST(inputs) as i)

    group by o.output_pubkey_base58 order by output_sum desc limit 1000

"""

print (str(round((bq_assistant.estimate_query_size(q)),2))+str(" GB"))



results2=bq.Run_Query(q, max_gb_scanned=70)

results2["output_sum"]=results2["output_sum"].apply(lambda x: float(x/100000000))
import matplotlib.pyplot as plt; plt.rcdefaults()

import numpy as np

import matplotlib.pyplot as plt

objects = results2["output_pubkey_base58"][:10]

y_pos = np.arange(len(objects))

performance = results2["output_sum"][:10]

plt.bar(y_pos, performance, align='center', alpha=0.5)

plt.xticks(y_pos, objects, rotation=90)

plt.ylabel('Bitcoins')

plt.title('Bitcoins Addresses Who received Most number of bitcoins')

plt.show()
q = """

SELECT  TIMESTAMP_MILLIS((timestamp - MOD(timestamp,

          86400000))) AS day,o.output_pubkey_base58, o.output_satoshis as output_max from 

    `bigquery-public-data.bitcoin_blockchain.transactions`JOIN

    UNNEST(outputs) as o order by output_max desc limit 1000

"""

print (str(round((bq_assistant.estimate_query_size(q)),2))+str(" GB"))

results3=bq.Run_Query(q, max_gb_scanned=56)



#CONVERT SATOSHIS TO BITCOINS

results3["output_max"]=results3["output_max"].apply(lambda x: float(x/100000000))

results3.head()
results4=results3.sort_values(by="day")

layout = go.Layout(title="Time Series of Highest single output transaction")

data = [go.Scatter(x=results4.day, y=results4.output_max)]

fig = go.Figure(data=data, layout=layout)

py.offline.iplot(fig)
results4["day"]=results4["day"].apply(lambda x: x.year)

years_output={}

years_max_output_count={}

for i,x in results4.iterrows():

    if x["day"] not in years_output:

        years_output[x["day"]]=[]

    if x["day"] not in years_max_output_count:

        years_max_output_count[x["day"]]=[]

    years_output[x["day"]].append(x["output_max"])

    years_max_output_count[x["day"]].append(x["output_pubkey_base58"])

years_output_final={}

for x in years_output.keys():

    years_output_final[str(x)]=np.mean(years_output[x])

years_max_output_count_final={}

for x in years_max_output_count.keys():

    years_max_output_count_final[str(x)]=len(years_max_output_count[x])
print (results3.iloc[len(results3)-1]["output_max"])

print (results3.iloc[len(results3)-1]["day"])
d=Counter(years_output_final)

d.most_common(1)

Create_Bar_plotly(d.most_common(), title="Single Highest Valued Transaction Average Per Year")
d=Counter(years_max_output_count_final)

Create_Bar_plotly(d.most_common(), title="Most number of high transaction yearwise")
results4[results4.day==2018]

results3[results3.index==970]
results3.iloc[list(results4[results4.day==2010].index)]
results3.iloc[list(results4[results4.day==2016].index)]
q = """

SELECT  TIMESTAMP_MILLIS((timestamp - MOD(timestamp,

          86400000))) AS day, avg(o.output_satoshis) as output_avg from 

    `bigquery-public-data.bitcoin_blockchain.transactions`JOIN

    UNNEST(outputs) as o group by day order by output_avg desc

"""

print (str(round((bq_assistant.estimate_query_size(q)),2))+str(" GB"))



results5=bq.Run_Query(q, max_gb_scanned=89)



#CONVERT SATOSHIS TO BITCOINS

results5["output_avg"]=results5["output_avg"].apply(lambda x: float(x/100000000))

results5.head()
results6=results5.sort_values(by="day")

layout = go.Layout(title="Time Series of AVERAGE in no of bitcoins transacted per day")

data = [go.Scatter(x=results6.day, y=results6.output_avg)]

fig = go.Figure(data=data, layout=layout)

py.offline.iplot(fig)
q = """

SELECT  TIMESTAMP_MILLIS((timestamp - MOD(timestamp,

          86400000))) AS day, count(o.output_satoshis) as counts from 

    `bigquery-public-data.bitcoin_blockchain.transactions`JOIN

    UNNEST(outputs) as o group by day order by counts

"""

print (str(round((bq_assistant.estimate_query_size(q)),2))+str(" GB"))

results7=bq.Run_Query(q,max_gb_scanned=180)

results7.tail()
results7=results7.sort_values(by="day")

layout = go.Layout(title="Time Series of transaction COUNT in Bitcoins")

data = [go.Scatter(x=results7.day, y=results7.counts)]

fig = go.Figure(data=data, layout=layout)

py.offline.iplot(fig)
q = """

SELECT  count(o.output_pubkey_base58) from

    `bigquery-public-data.bitcoin_blockchain.transactions`JOIN

    UNNEST(outputs)as o, UNNEST(inputs) as i where ARRAY_LENGTH(outputs)=1 and

    ARRAY_LENGTH(inputs)=1 and i.input_pubkey_base58=o.output_pubkey_base58

"""

print (str(round((bq_assistant.estimate_query_size(q)),2))+str(" GB"))

results9=bq.Run_Query(q, max_gb_scanned=580)

print (results9)
q = """

SELECT  TIMESTAMP_MILLIS((timestamp - MOD(timestamp,

          86400000))) AS day,o.output_pubkey_base58 as key, o.output_satoshis as price from 

    `bigquery-public-data.bitcoin_blockchain.transactions`JOIN

    UNNEST(outputs)as o, UNNEST(inputs) as i where ARRAY_LENGTH(outputs)=1 and

    ARRAY_LENGTH(inputs)=1 and i.input_pubkey_base58=o.output_pubkey_base58

    order by price desc limit 1000

"""

print (str(round((bq_assistant.estimate_query_size(q)),2))+str(" GB"))

results9=bq.Run_Query(q, max_gb_scanned=588)

#CONVERT SATOSHIS TO BITCOINS

results9["price"]=results9["price"].apply(lambda x: float(x/100000000))

results9.head()
results9=results9.sort_values(by="day")

layout = go.Layout(title="Self Transactions price by bitcoins holders TimeSeries")

data = [go.Scatter(x=results9.day, y=results9.price)]

fig = go.Figure(data=data, layout=layout)



py.offline.iplot(fig, image="png")
QUERY = """

SELECT

    inputs.input_pubkey_base58 AS input_key, count(*)

FROM `bigquery-public-data.bitcoin_blockchain.transactions`

    JOIN UNNEST (inputs) AS inputs

WHERE inputs.input_pubkey_base58 IS NOT NULL

GROUP BY inputs.input_pubkey_base58 order by count(*) desc limit 1000

"""

bq_assistant.estimate_query_size(QUERY)

ndf=bq.Run_Query(QUERY, max_gb_scanned=238)

import matplotlib.pyplot as plt; plt.rcdefaults()

import numpy as np

import matplotlib.pyplot as plt

objects = ndf["input_key"][:10]

y_pos = np.arange(len(objects))

performance = ndf["f0_"][:10] 

plt.bar(y_pos, performance, align='center', alpha=0.5)

plt.xticks(y_pos, objects, rotation=90)

plt.ylabel('Number of transactions')

plt.title('BITCOIN ADDRESSES WITH MOST NUMBER OF TRANSACTIONS')

plt.show()
ndf.iloc[0]
q = """

SELECT  TIMESTAMP_MILLIS((timestamp - MOD(timestamp,

          86400000))) AS day,o.output_pubkey_base58, o.output_satoshis as output_max from 

    `bigquery-public-data.bitcoin_blockchain.transactions`JOIN

    UNNEST(outputs) as o

    where o.output_pubkey_base58 not in 

    (

    select i.input_pubkey_base58 from 

    `bigquery-public-data.bitcoin_blockchain.transactions`,

    UNNEST(inputs) as i where i.input_pubkey_base58 is not null

    )

    order by output_max desc limit 10000

"""

print (str(round((bq_assistant.estimate_query_size(q)),2))+str(" GB"))

inactive_wallets=bq.Run_Query(q, max_gb_scanned=69)

inactive_wallets["output_max"]=inactive_wallets["output_max"].apply(lambda x: float(x/100000000))

inactive_wallets_2=inactive_wallets.sort_values(by="day")

layout = go.Layout(title="Value of unspent outputs with respect to time")

data = [go.Scatter(x=inactive_wallets_2.day, y=inactive_wallets.output_max)]

fig = go.Figure(data=data, layout=layout)

py.offline.iplot(fig)
inactive_wallets_2["day"]=inactive_wallets_2["day"].apply(lambda x: x.year)

years_outputs={}

for i,x in inactive_wallets_2.iterrows():

    if x["day"] not in years_outputs:

        years_outputs[x["day"]]=0

    years_outputs[x["day"]]+=1

years=Counter(years_outputs)

Create_Bar_plotly(years.most_common(),title="Count of outputs never spent with respect to year")
yearly_unspent={}

for i,x in inactive_wallets_2.iterrows():

    if x["day"] not in yearly_unspent:

        yearly_unspent[x["day"]]={}

    if x["output_pubkey_base58"] not in yearly_unspent[x["day"]]:

        yearly_unspent[x["day"]][x["output_pubkey_base58"]]=0    

    yearly_unspent[x["day"]][x["output_pubkey_base58"]]+=x["output_max"]
yearly=Counter(yearly_unspent[2010])

yearly.most_common()
np.sum(inactive_wallets["output_max"])
data_f=pd.read_csv("../input/bitstampUSD_1-min_data_2012-01-01_to_2018-03-27.csv")

data_f["Timestamp"]=pd.to_datetime(data_f["Timestamp"], unit='s')

data_f.Timestamp=data_f.Timestamp.apply(lambda x: x.replace(hour=0, minute=0, second=0))

data_f=data_f[["Timestamp", "Weighted_Price"]]

data_f=data_f.drop_duplicates(keep="first")

data_f=data_f.drop_duplicates(subset='Timestamp', keep="last")

data_f.head()
import time



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
end_time=time.time()

print ("TIME TAKEN for the kernel")

print (end_time-start_time)