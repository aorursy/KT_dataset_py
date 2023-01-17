# Start by importing the bq_helper module and calling on the specific active_project and dataset_name for the BigQuery dataset.

import bq_helper

from bq_helper import BigQueryHelper

# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package



from google.cloud import bigquery

import pandas as pd

patents_research = bq_helper.BigQueryHelper(active_project="patents-public-data",

                                   dataset_name="google_patents_research")
# View table names under the google_patents_research data table

bq_assistant = BigQueryHelper("patents-public-data", "google_patents_research")

bq_assistant.list_tables()
# View the first three rows of the publications data table

bq_assistant.head("publications", num_rows=3)
# View the last 3 rows of the publications data table

# bq_assistant.last("publications", num_rows=3) ##no such command? 
# View information on all columns in the trials data table

bq_assistant.table_schema("publications")
query_count = """

SELECT count(*)

FROM

  `patents-public-data.google_patents_research.publications`

"""

print(bq_assistant.estimate_query_size(query_count))

patents_research.query_to_pandas_safe(query_count)



### 121 million rows/patents
# query_country = """

# SELECT DISTINCT

#   country

# FROM

#   `patents-public-data.google_patents_research.publications`

# LIMIT

#   500;

#         """

# query_country = patents_research.query_to_pandas_safe(query_country, max_gb_scanned=25)

# query_country



#### 'United States' , 'Eurasian Patent Office' , 'United Kingdom' , 'WIPO (PCT)' , 'EUIPO' , 'USSR - Soviet Union' ...
query1 = """

SELECT 

  publication_number, top_terms, title

FROM

  `patents-public-data.google_patents_research.publications`

WHERE

(ARRAY_LENGTH(top_terms)> 0) AND (title_translated = FALSE) 

AND (CHAR_LENGTH(title)>2)

AND (country = "United States" OR country = "USA")

AND (publication_description LIKE "Patent%")



LIMIT 4123456 OFFSET 12123456

;

        """



# cpc, ## cpc.code causes error, and returning the cpc seems to cause memory issues + be much slower for some reason  

##   (LEN(top_terms)>1)

# (NOT isnull(title))

# AND (publication_description LIKE "%atent%")



# publication_description - Patent , [china :  Granted Patent , Granted patent for invention ..  ]



#  LIMIT

#    27123456



## could filter by last letter(2) of publication_number , corresponds to patent kind / type. (e.g. A = patent, granted)



print(bq_assistant.estimate_query_size(query1))

df = patents_research.query_to_pandas_safe(query1, max_gb_scanned=50)



df.head(20) ## with an offset of 13123456 , and no special other filtering, we see the first patent in 2012
# response1 = df

print(df.shape)
df.tail(30)
# print(df["publication_description"].value_counts())
print(df.shape[0])

# df = response1.drop(["title_translated","country"],axis=1,errors="ignore")

df = df.drop_duplicates(subset=["publication_number","title"])

print(df.shape[0])

df
df.to_csv("sample_usa_patents_research_k_bq_terms_last.csv.gz",index=False,compression="gzip")
# response1
### https://www.kaggle.com/shaileshshettyd/china-patents-contributions

top_terms = pd.DataFrame(df["top_terms"].tolist())



top_terms = pd.DataFrame(top_terms.values.flatten())

top_terms.columns = ['top_terms']



top_terms = top_terms.dropna(axis=0,how='all')

top_terms.shape
# 9 millions terms

top_terms.sample(5)
#  unique terms

top_terms.top_terms.nunique()
df_agg = pd.DataFrame(top_terms.sample(frac=0.02).groupby('top_terms')['top_terms'].count())



df_agg.columns = ['counter']

df_agg = df_agg.sort_values('counter', ascending=False)

# df_agg = df_agg.head(30)



# df_agg.tail(5)



df_agg.head(50)
df_agg.tail(30)
# # from pandas import json_normalize

# from pandas.io.json import json_normalize

def get_nested_codes(row):

    ls = []

#     row = json_normalize(row)

    for i in row:

        ls.append(i["code"])

    return(ls)
# print(df["cpc"][0])

# df["cpc"] = df["cpc"].apply(get_nested_codes)

# print(df["cpc"][0])