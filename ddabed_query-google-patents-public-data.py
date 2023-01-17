# Start by importing the bq_helper module and calling on the specific active_project and dataset_name for the BigQuery dataset.
import bq_helper
from bq_helper import BigQueryHelper
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package

patents = bq_helper.BigQueryHelper(active_project="patents-public-data",
                                   dataset_name="patents")
# View table names under the patents data table
bq_assistant = BigQueryHelper("patents-public-data", "patents")
bq_assistant.list_tables()
bq_assistant.table_schema("publications")
# View the first three rows of the publications data table
data_head = bq_assistant.head("publications_201912", num_rows=5)
data_head[[ 'publication_number','family_id']]

# View the first three rows of the publications data table
bq_assistant.head("publications_201710", num_rows=5)
query1 = """
SELECT
  country_code 
  abstract_localized
FROM
  `patents-public-data.patents.publications`
LIMIT
  20;
        """
#response1 = patents.query_to_pandas_safe(query1)
#response1.head(20)
query2 = """
SELECT 
  publication_number,
  family_id,
  title_localized,
  abstract_localized
FROM
  `patents-public-data.patents.publications`
WHERE
  country_code = 'WO'
LIMIT
  20;
        """
query21 =  """
SELECT
  publication_number,
  family_id,
  title_localized,
  abstract_localized,
  claims_localized,
  citation
FROM
  `patents-public-data.patents.publications_201912`
WHERE
  country_code = 'US' AND
  grant_date > 0;
        """
query3 = """
SELECT FIRST(publication_number)
FROM
  `patents-public-data.patents.publications_201912`;
        """
#response1 = patents.query_to_pandas_safe(query3)
#response1
query4 = """
SELECT * FROM (
  SELECT
    ROW_NUMBER() OVER (ORDER BY publication_number ASC) AS rownumber
    FROM `patents-public-data.patents.publications_201912`
)
WHERE rownumber IN (2,5);
"""
response4 = patents.query_to_pandas_safe(query4)
response4
bq_assistant.estimate_query_size(query4)