import bq_helper
from bq_helper import BigQueryHelper
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package
RxNorm = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nlm_rxnorm")
bq_assistant = BigQueryHelper("bigquery-public-data", "nlm_rxnorm")
bq_assistant.list_tables()
bq_assistant.head("rxnsat_current", num_rows=20)
bq_assistant.head("rxnsab_current", num_rows=20)
bq_assistant.table_schema("rxnsab_current")
query1 = """SELECT
  *
FROM
  `bigquery-public-data.nlm_rxnorm.rxn_all_pathways_current`
WHERE
  SOURCE_NAME IN ('Vancomycin 100 MG/ML',
    'Ceftriaxone 250 MG Injection',
    'Metronidazole Pill',
    '1 ML Morphine Sulfate 15 MG/ML Cartridge',
    '{14 (Estrogens, Conjugated (USP) 0.625 MG / medroxyprogesterone acetate 5 MG Oral Tablet) / 14 (Estrogens, Conjugated (USP) 0.625 MG Oral Tablet) } Pack [Premphase 28 Day]',
    'Warfarin Sodium 4 MG Oral Tablet',
    'Diphenhydramine',
    'Metformin / pioglitazone',
    'Ibuprofen 800 MG Extended Release Oral Tablet',
    'Valium' )
  AND TARGET_TTY = 'IN'
ORDER BY
  SOURCE_RXCUI;
        """
response1 = RxNorm.query_to_pandas_safe(query1)
response1.head(20)
query2 = """SELECT
  SOURCE_NAME AS IN_NAME,
  COUNT (*) AS DF_COUNT
FROM
  `bigquery-public-data.nlm_rxnorm.rxn_all_pathways_current`
WHERE
  SOURCE_TTY='IN'
  AND TARGET_TTY='DF'
GROUP BY
  IN_NAME
ORDER BY
  DF_COUNT DESC
LIMIT
  10;
        """
response2 = RxNorm.query_to_pandas_safe(query2, max_gb_scanned=10)
response2.head(20)
query3 = """SELECT
  *
FROM
  `bigquery-public-data.nlm_rxnorm.rxn_all_pathways_current`
WHERE
  SOURCE_NAME='Phenylephrine'
  AND target_TTY = 'DF';
        """
response3 = RxNorm.query_to_pandas_safe(query3, max_gb_scanned=10)
response3.head(20)