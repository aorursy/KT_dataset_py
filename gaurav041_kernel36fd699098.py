# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from IPython.core.display import display, HTML



import base64

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from google.cloud import bigquery

from bq_helper import BigQueryHelper



def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv()

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)





bq_assistant = BigQueryHelper("patents-public-data", "google_patents_research")

# bq_assistant.list_tables()

client = bigquery.Client()

# display(create_download_link(publication_df))
query = ('SELECT * FROM `patents-public-data.google_patents_research.publications` LIMIT 10')

query_job = client.query(query) 

query_job.to_dataframe()
query = ('SELECT * FROM `patents-public-data.google_patents_research.publications` where top_terms is not null LIMIT 5')

publication_job = client.query(query)

[job for job in publication_job]
query = ('SELECT * FROM `patents-public-data.patents.publications`  order by filing_date desc LIMIT 10')

filing_date_job = client.query(query)

[job for job in filing_date_job]
query = ('SELECT AVG(num_inventors), COUNT(*) AS cnt, country_code, filing_year, STRING_AGG(publication_number LIMIT 10) AS example_publications FROM (  SELECT ANY_VALUE(publication_number) AS publication_number, ANY_VALUE(ARRAY_LENGTH(inventor)) AS num_inventors, ANY_VALUE(country_code) AS country_code, ANY_VALUE(CAST(FLOOR(filing_date / (5*10000)) AS INT64))*5 AS filing_year   FROM `patents-public-data.patents.publications` AS pubs  WHERE filing_date > 19000000 AND ARRAY_LENGTH(inventor) > 0  GROUP BY application_number) GROUP BY filing_year, country_code HAVING cnt > 100 ORDER BY filing_year')

avg_job = client.query(query)

# [job for job in avg_job]

avg_inventor = avg_job.result().to_dataframe()
avg_inventor.plot()
count = avg_inventor['cnt'].tolist()

country_codes = avg_inventor['country_code'].tolist()
country_codes
import numpy as np                                                               

import matplotlib.pyplot as plt



ys = count

xs = country_codes



width = 10

plt.bar(xs, ys, width, align='center')

# plt.xticks(np.arange(len(country_codes)), xs) #Replace default x-ticks with xs, then replace xs with labels

plt.show()

query = 'select publication_date from `patents-public-data.patents.publications` order by publication_date desc limit 5;'

job = client.query(query)

job.result().to_dataframe()
query = 'select * from `patents-public-data.google_patents_research.publications` where ARRAY_LENGTH(top_terms) > 0 limit 1000'

job = client.query(query)

result = job.result().to_dataframe()
top_terms_all = result['top_terms']
terms = [term for term in top_terms_all]
freq = {} 

for sublist in terms:

    for item in sublist:

        if (item in freq):

            freq[item] +=1

        else:

            freq[item] = 1
sorted(freq)
# sorted(freq.iteritems(), key = lambda x : x[1])

import operator

import collections

sorted_freq = sorted(freq.items(), key=operator.itemgetter(1), reverse=True)

# collections.OrderedDict(freq)
sorted_freq[:3]