import bq_helper
from bq_helper import BigQueryHelper
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package
usfs = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="usfs_fia")
bq_assistant = BigQueryHelper("bigquery-public-data", "usfs_fia")
bq_assistant.list_tables()
bq_assistant.head("plot_tree", num_rows=15)
bq_assistant.table_schema("plot_tree")
# Note: State and county are FIPS state codes.
query1 = """
SELECT
    plot_sequence_number,
    plot_state_code,
    plot_county_code,
    measurement_year,
    latitude,
    longitude,
    tree_sequence_number,
    species_code,
    current_diameter
FROM
    `bigquery-public-data.usfs_fia.plot_tree`
WHERE
    plot_state_code = 53
    AND plot_county_code = 33
;
        """
response1 = usfs.query_to_pandas_safe(query1, max_gb_scanned=10)
response1.head(20)
query2 = """
SELECT
    plot_sequence_number,
    plot_state_code,
    plot_state_code_name
    plot_county_code,
    measurement_year,
    latitude,
    longitude,
    tree_sequence_number,
    species_code,
    species_common_name,
    species_scientific_name,
    current_diameter
FROM
    `bigquery-public-data.usfs_fia.plot_tree`
WHERE
    plot_state_code = 53
    AND plot_county_code = 33
;
        """
response2 = usfs.query_to_pandas_safe(query2, max_gb_scanned=10)
response2.head(10)
query5 = """
Select  
 pt.plot_sequence_number as plot_sequence_number,
 p.evaluation_type evaluation_type,
 p.evaluation_group as evaluation_group,
 p.evaluation_description as evaluation_description,
 pt.plot_state_code_name as state_name,
 p.inventory_year as inventory_year,
 p.state_code as state_code, 
 #calculate area - this replaces the "decode" logic in example from Oracle
 CASE
  WHEN c.proportion_basis = 'MACR' and p.adjustment_factor_for_the_macroplot > 0
  THEN
    (p.expansion_factor * c.condition_proportion_unadjusted * p.adjustment_factor_for_the_macroplot) 
  ELSE 0
 END as macroplot_acres,
 CASE
  WHEN c.proportion_basis = 'SUBP' and p.adjustment_factor_for_the_subplot > 0
  THEN
    (p.expansion_factor * c.condition_proportion_unadjusted * p.adjustment_factor_for_the_subplot) 
  ELSE 0
 END as subplot_acres
FROM 
  `bigquery-public-data.usfs_fia.condition`  c
JOIN 
  `bigquery-public-data.usfs_fia.plot_tree`  pt
        ON pt.plot_sequence_number = c.plot_sequence_number
JOIN 
  `bigquery-public-data.usfs_fia.population`  p
      ON p.plot_sequence_number = pt.plot_sequence_number
WHERE 
  p.evaluation_type = 'EXPCURR'
  AND c.condition_status_code = 1
GROUP BY 
 plot_sequence_number,
 evaluation_type,
 evaluation_group,
 evaluation_description,
 macroplot_acres,
 subplot_acres,
 inventory_year,
 state_code,
 state_name
;
        """
response5 = usfs.query_to_pandas_safe(query5, max_gb_scanned=50)
response5.head(50)
query6 = """
Select  
 pt.plot_sequence_number as plot_sequence_number,
 p.evaluation_type evaluation_type,
 p.evaluation_group as evaluation_group,
 p.evaluation_description as evaluation_description,
 pt.plot_state_code_name as state_name,
 p.inventory_year as inventory_year,
 p.state_code as state_code, 
 #calculate area - this replaces the "decode" logic in example from Oracle
 CASE
  WHEN c.proportion_basis = 'MACR' and p.adjustment_factor_for_the_macroplot > 0
  THEN
    (p.expansion_factor * c.condition_proportion_unadjusted * p.adjustment_factor_for_the_macroplot) 
  ELSE 0
 END as macroplot_acres,
 CASE
  WHEN c.proportion_basis = 'SUBP' and p.adjustment_factor_for_the_subplot > 0
  THEN
    (p.expansion_factor * c.condition_proportion_unadjusted * p.adjustment_factor_for_the_subplot) 
  ELSE 0
 END as subplot_acres
FROM 
  `bigquery-public-data.usfs_fia.condition`  c
JOIN 
  `bigquery-public-data.usfs_fia.plot_tree`  pt
        ON pt.plot_sequence_number = c.plot_sequence_number
JOIN 
  `bigquery-public-data.usfs_fia.population`  p
      ON p.plot_sequence_number = pt.plot_sequence_number
WHERE 
  p.evaluation_type = 'EXPCURR'
  AND c.condition_status_code = 1
GROUP BY 
 plot_sequence_number,
 evaluation_type,
 evaluation_group,
 evaluation_description,
 macroplot_acres,
 subplot_acres,
 inventory_year,
 state_code,
 state_name;
        """
response6 = usfs.query_to_pandas_safe(query6, max_gb_scanned=50)
response6.head(50)
query3 = """
#standardSQL

SELECT
state_code,
evaluation_group,
evaluation_description,
state_name,
sum(macroplot_acres) + sum(subplot_acres) as total_acres,
latest
FROM (SELECT
        state_code,
        evaluation_group,
        evaluation_description,
        state_name,
        macroplot_acres,
        subplot_acres,
        MAX(evaluation_group) OVER (PARTITION By state_code) as latest                   
        FROM `bigquery-public-data.usfs_fia.estimated_timberland_acres` )
WHERE evaluation_group = latest
GROUP by state_code, state_name, evaluation_description, evaluation_group, latest
order by state_name
;
        """
response3 = usfs.query_to_pandas_safe(query3, max_gb_scanned=10)
response3.head(50)
query4 = """
SELECT
state_code,
evaluation_group,
evaluation_description,
state_name,
sum(macroplot_acres) + sum(subplot_acres) as total_acres,
latest
FROM (SELECT
        state_code,
        evaluation_group,
        evaluation_description,
        state_name,
        macroplot_acres,
        subplot_acres,
        MAX(evaluation_group) OVER (PARTITION By state_code) as latest                   
        FROM `bigquery-public-data.usfs_fia.estimated_forestland_acres` )
WHERE evaluation_group = latest
GROUP by state_code, state_name, evaluation_description, evaluation_group, latest
order by state_name
;
        """
response4 = usfs.query_to_pandas_safe(query4, max_gb_scanned=10)
response4.head(50)