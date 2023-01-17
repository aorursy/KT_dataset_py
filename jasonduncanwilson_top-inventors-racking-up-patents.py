from IPython.display import Image
url = 'https://patentimages.storage.googleapis.com/pages/USD253711-2.png'
Image(url=url,width=800, height=600)
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper
from bq_helper import BigQueryHelper
import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.figure_factory as ff

# prepare bigQuery helper
patents = bq_helper.BigQueryHelper(active_project="patents-public-data",
                                   dataset_name="patents")
bq_assistant = BigQueryHelper("patents-public-data", "patents")
bq_assistant.list_tables()
#patents.table_schema("publications")
#patents.head("publications")
#patents.head("publications",selected_columns="application_number_formatted,inventor,country_code", num_rows=10)
#import os
#print(os.listdir("../input"))
USPTO_patent_codes = pd.read_csv('../input/USPTO patent codes.csv')
USPTO_patent_codes.head(5)
# create query to be run
query1 = """
WITH temp1 AS (
    SELECT
      DISTINCT
      PUB.country_code,
      PUB.application_number AS patent_number,
      inventor_name
    FROM
      `patents-public-data.patents.publications` PUB
    CROSS JOIN
      UNNEST(PUB.inventor) AS inventor_name
    WHERE
          PUB.grant_date > 0
      AND PUB.country_code IS NOT NULL
      AND PUB.application_number IS NOT NULL
      AND PUB.inventor IS NOT NULL
)
SELECT
  *
FROM (
    SELECT
     temp1.country_code AS country,
     temp1.inventor_name AS inventor,
     COUNT(temp1.patent_number) AS count_of_patents
    FROM temp1
    GROUP BY
     temp1.country_code,
     temp1.inventor_name
     )
WHERE
 count_of_patents > 100
;
"""
# Check size of data being examined by query
#patents.estimate_query_size(query1)

# Run query 
query_results = patents.query_to_pandas_safe(query1, max_gb_scanned=6)
print("Number of records:", len(query_results.index))
query_results.head(5)
# reduce results down to the top 50 inventors in the US
top_50_inventors = query_results.loc[query_results['country'] == "US"].nlargest(50,'count_of_patents')

# show the top 50 in a plotly table
table1 = ff.create_table(top_50_inventors)
py.iplot(table1, filename='jupyter-table1')