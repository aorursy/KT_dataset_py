import bq_helper
from bq_helper import BigQueryHelper
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package
nppes = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nppes")
bq_assistant = BigQueryHelper("bigquery-public-data", "nppes")
bq_assistant.list_tables()
bq_assistant.head("healthcare_provider_taxonomy_code_set", num_rows=20)
query1 = """
SELECT
  healthcare_provider_taxonomy_1_specialization,
  COUNT(DISTINCT npi) AS number_specialist
FROM
  `bigquery-public-data.nppes.npi_optimized`
WHERE
 provider_business_practice_location_address_city_name = "MOUNTAIN VIEW"
 AND provider_business_practice_location_address_state_name = "CA"
  AND healthcare_provider_taxonomy_1_specialization > ""
GROUP BY
  healthcare_provider_taxonomy_1_specialization
ORDER BY
  number_specialist DESC
LIMIT
  20;
        """
response1 = nppes.query_to_pandas_safe(query1)
response1.head(20)
query2 = """
SELECT
  provider_credential_text,
  provider_first_name,
  provider_business_practice_location_address_telephone_number
FROM
  `bigquery-public-data.nppes.npi_optimized`
WHERE
  provider_business_mailing_address_state_name = 'CA'
  AND healthcare_provider_taxonomy_1_grouping = 'Dental Providers'
  AND REPLACE(provider_credential_text, ".","") LIKE '%MPH%';
        """
response2 = nppes.query_to_pandas_safe(query2, max_gb_scanned=10)
response2.head(20)
