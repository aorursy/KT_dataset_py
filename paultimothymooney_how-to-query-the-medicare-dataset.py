import bq_helper
from bq_helper import BigQueryHelper
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package
medicare = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="cms_medicare")
bq_assistant = BigQueryHelper("bigquery-public-data", "cms_medicare")
bq_assistant.list_tables()
bq_assistant.head("inpatient_charges_2015", num_rows=15)
bq_assistant.table_schema("inpatient_charges_2015")
query1 = """SELECT
  nppes_provider_state AS state,
  ROUND(SUM(total_claim_count) / 1e6) AS total_claim_count_millions
FROM
  `bigquery-public-data.cms_medicare.part_d_prescriber_2014`
GROUP BY
  state
ORDER BY
  total_claim_count_millions DESC
LIMIT
  5;
        """
response1 = medicare.query_to_pandas_safe(query1)
response1.head(10)
query2 = """SELECT
  A.state,
  drug_name,
  total_claim_count,
  day_supply,
  ROUND(total_cost_millions) AS total_cost_millions
FROM (
  SELECT
    generic_name AS drug_name,
    nppes_provider_state AS state,
    ROUND(SUM(total_claim_count)) AS total_claim_count,
    ROUND(SUM(total_day_supply)) AS day_supply,
    ROUND(SUM(total_drug_cost)) / 1e6 AS total_cost_millions
  FROM
    `bigquery-public-data.cms_medicare.part_d_prescriber_2014`
  GROUP BY
    state,
    drug_name) A
INNER JOIN (
  SELECT
    state,
    MAX(total_claim_count) AS max_total_claim_count
  FROM (
    SELECT
      nppes_provider_state AS state,
      ROUND(SUM(total_claim_count)) AS total_claim_count
    FROM
      `bigquery-public-data.cms_medicare.part_d_prescriber_2014`
    GROUP BY
      state,
      generic_name)
  GROUP BY
    state) B
ON
  A.state = B.state
  AND A.total_claim_count = B.max_total_claim_count
ORDER BY
  A.total_claim_count DESC
LIMIT
  5;
        """
response2 = medicare.query_to_pandas_safe(query2, max_gb_scanned=10)
response2.head(10)
query3 = """SELECT
  OP.provider_state AS State,
  OP.provider_city AS City,
  OP.provider_id AS Provider_ID,
  ROUND(OP.average_OP_cost) AS Average_OP_Cost,
  ROUND(IP.average_IP_cost) AS Average_IP_Cost,
  ROUND(OP.average_OP_cost + IP.average_IP_cost) AS Combined_Average_Cost
FROM (
  SELECT
    provider_state,
    provider_city,
    provider_id,
    SUM(average_total_payments*outpatient_services)/SUM(outpatient_services) AS average_OP_cost
  FROM
    `bigquery-public-data.cms_medicare.outpatient_charges_2014`
  GROUP BY
    provider_state,
    provider_city,
    provider_id ) AS OP
INNER JOIN (
  SELECT
    provider_state,
    provider_city,
    provider_id,
    SUM(average_medicare_payments*total_discharges)/SUM(total_discharges) AS average_IP_cost
  FROM
    `bigquery-public-data.cms_medicare.inpatient_charges_2014`
  GROUP BY
    provider_state,
    provider_city,
    provider_id ) AS IP
ON
  OP.provider_id = IP.provider_id
  AND OP.provider_state = IP.provider_state
  AND OP.provider_city = IP.provider_city
ORDER BY
  combined_average_cost DESC
LIMIT
  10;
        """
response3 = medicare.query_to_pandas_safe(query3, max_gb_scanned=10)
response3.head(10)
query4 = """SELECT
  drg_definition AS Diagnosis,
  provider_city AS City,
  provider_state AS State,
  cityrank AS City_Rank,
  CAST(ROUND(citywise_avg_total_payments) AS INT64) AS Citywise_Avg_Payments,
  CONCAT(CAST(ROUND(citywise_avg_total_payments /national_avg_total_payments * 100, 0) AS STRING), " %") AS Avg_Payments_City_vs_National
FROM (
  SELECT
    drg_definition,
    provider_city,
    provider_state,
    cityrank,
    national_num_cases,
    citywise_avg_total_payments,
    national_sum_total_payments,
    (national_sum_total_payments /national_num_cases) AS national_avg_total_payments
  FROM (
    SELECT
      drg_definition,
      provider_city,
      provider_state,
      citywise_avg_total_payments,
      RANK() OVER (PARTITION BY drg_definition ORDER BY citywise_num_cases DESC ) AS cityrank,
      SUM(citywise_num_cases) OVER (PARTITION BY drg_definition ) AS national_num_cases,
      SUM(citywise_sum_total_payments) OVER (PARTITION BY drg_definition ) AS national_sum_total_payments
    FROM (
      SELECT
        drg_definition,
        provider_city,
        provider_state,
        SUM(total_discharges) AS citywise_num_cases,
        SUM(average_total_payments * total_discharges)/ SUM(total_discharges) AS citywise_avg_total_payments,
        SUM(average_total_payments * total_discharges) AS citywise_sum_total_payments
      FROM
        `bigquery-public-data.cms_medicare.inpatient_charges_2014`
      GROUP BY
        drg_definition,
        provider_city,
        provider_state))
  WHERE
    cityrank <=3)  # Limit to top 3 cities for each Diagnosis
ORDER BY
  national_num_cases DESC,
  cityrank
LIMIT
  9;  # Limit Results to the top 3 cities for the top 3 diagnosis;
        """
response4 = medicare.query_to_pandas_safe(query4, max_gb_scanned=10)
response4.head(10)
