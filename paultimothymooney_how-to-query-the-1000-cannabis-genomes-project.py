import bq_helper
from bq_helper import BigQueryHelper
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package
genomes = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="genomics_cannabis")
bq_assistant = BigQueryHelper("bigquery-public-data", "genomics_cannabis")
bq_assistant.list_tables()
bq_assistant.head("MNPR01_201703", num_rows=10)
bq_assistant.table_schema("MNPR01_201703")
query1 = """SELECT
  i.Sample_Name_s AS sample_name,
  call.call_set_name AS call_set_name,
  COUNT(call.call_set_name) AS call_count_for_call_set
FROM
  `bigquery-public-data.genomics_cannabis.sample_info_201703` i,
  `bigquery-public-data.genomics_cannabis.MNPR01_201703` v,
  v.call
WHERE
  call.call_set_name = i.SRA_Sample_s
  AND reference_name = 'gi|1098492959|gb|MNPR01002882.1|'
  AND EXISTS (
  SELECT
    1
  FROM
    UNNEST(v.alternate_bases) AS alt
  WHERE
    alt NOT IN ("",
      "<*>"))
  AND v.dp >= 10
  AND v.start >= 12800
  AND v.end <= 14600
GROUP BY
  call_set_name,
  Sample_Name_s
ORDER BY
  call_set_name;
        """
response1 = genomes.query_to_pandas_safe(query1, max_gb_scanned=10)
response1.head(10)
query2 = """SELECT
  call.call_set_name
FROM
  `bigquery-public-data.genomics_cannabis.MNPR01_201703` v,
  v.call
GROUP BY
  call.call_set_name;
        """
response2 = genomes.query_to_pandas_safe(query2, max_gb_scanned=10)
response2.head(10)
query3 = """SELECT
  *
FROM (
  SELECT
    reference_name,
    COUNT(reference_name) / r.length AS variant_density,
    COUNT(reference_name) AS variant_count,
    r.length AS reference_length
  FROM
    `bigquery-public-data.genomics_cannabis.MNPR01_201703` v,
    `bigquery-public-data.genomics_cannabis.MNPR01_reference_201703` r
  WHERE
    v.reference_name = r.name
    AND EXISTS (
    SELECT
      1
    FROM
      UNNEST(v.call) AS call
    WHERE
      EXISTS (
      SELECT
        1
      FROM
        UNNEST(call.genotype) AS gt
      WHERE
        gt > 0))
  GROUP BY
    reference_name,
    r.length ) AS d
ORDER BY
  variant_density DESC;
        """
response3 = genomes.query_to_pandas_safe(query3, max_gb_scanned=10)
response3.head(10)