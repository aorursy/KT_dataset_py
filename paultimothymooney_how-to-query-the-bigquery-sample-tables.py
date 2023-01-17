import bq_helper
from bq_helper import BigQueryHelper
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package
sampleTables = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="samples")
bq_assistant = BigQueryHelper("bigquery-public-data", "samples")
bq_assistant.list_tables()
bq_assistant.head("wikipedia", num_rows=20)
query1 = """SELECT
  word
FROM
  `bigquery-public-data.samples.shakespeare`
WHERE
  word LIKE 'prais%' AND word LIKE '%ing' OR
  word LIKE 'laugh%' AND word LIKE '%ed';
        """
response1 = sampleTables.query_to_pandas_safe(query1, max_gb_scanned=10)
response1.head(10)
query2 = """SELECT word
FROM `bigquery-public-data.samples.shakespeare`
WHERE RAND() < 20/164656;
        """
response2 = sampleTables.query_to_pandas_safe(query2, max_gb_scanned=10)
response2.head(20)
query3 = """SELECT
  word,
  corpus,
  COUNT(word)
FROM
  `bigquery-public-data.samples.shakespeare`
WHERE
  word LIKE 'th%'
GROUP BY
  word,
  corpus;
        """
response3 = sampleTables.query_to_pandas_safe(query3, max_gb_scanned=10)
response3.head(10)
query4 = """SELECT
  word,
  corpus,
  COUNT(word)
FROM
  `bigquery-public-data.samples.shakespeare`
WHERE
  word LIKE '%th'
GROUP BY
  word,
  corpus;
        """
response4 = sampleTables.query_to_pandas_safe(query4, max_gb_scanned=10)
response4.head(10)
query5 = """SELECT
  mother_age,
  COUNT(mother_age) total
FROM
  `bigquery-public-data.samples.natality`
WHERE
  state IN (SELECT
              state
            FROM
              (SELECT
                 state,
                 COUNT(state) total
               FROM
                 `bigquery-public-data.samples.natality`
               GROUP BY
                 state
               ORDER BY
                 total DESC
               LIMIT 20))
  AND mother_age > 50
GROUP BY
  mother_age
ORDER BY
  mother_age DESC;
        """
response5 = sampleTables.query_to_pandas_safe(query5, max_gb_scanned=10)
response5.head(10)
query6 = """SELECT
  mother_age,
  COUNT(mother_age) total
FROM
  `bigquery-public-data.samples.natality`
WHERE
  state NOT IN (SELECT
                  state
                FROM
                  (SELECT
                     state,
                     COUNT(state) total
                   FROM
                     `bigquery-public-data.samples.natality`
                   GROUP BY
                     state
                   ORDER BY
                     total DESC
                   LIMIT 10))
  AND mother_age > 50
GROUP BY
  mother_age
ORDER BY
  mother_age DESC;
        """
response6 = sampleTables.query_to_pandas_safe(query6, max_gb_scanned=10)
response6.head(10)
query8 = """SELECT
  year,
  is_male,
  COUNT(1) as count
FROM
  `bigquery-public-data.samples.natality`
WHERE
  year >= 2000
  AND year <= 2002
GROUP BY
  ROLLUP(year, is_male)
ORDER BY
  year,
  is_male;
        """
response8 = sampleTables.query_to_pandas_safe(query8, max_gb_scanned=10)
response8.head(10)
query9 = """SELECT
  cigarette_use,
  /* Finds average and standard deviation */
  AVG(weight_pounds) baby_weight,
  STDDEV(weight_pounds) baby_weight_stdev,
  AVG(mother_age) mother_age
FROM
  `bigquery-public-data.samples.natality`
WHERE
  year=2003 AND state='OH'
/* Group the result values by those */
/* who smoked and those who didn't.  */
GROUP BY
  cigarette_use;
        """
response9 = sampleTables.query_to_pandas_safe(query9, max_gb_scanned=10)
response9.head(10)
query10 = """SELECT
  state,
  /* If 'is_male' is True, return 'Male', */
  /* otherwise return 'Female' */
  IF (is_male, 'Male', 'Female') AS sex,
  /* The count value is aliased as 'cnt' */
  /* and used in the HAVING clause below. */
  COUNT(*) AS cnt
FROM
  `bigquery-public-data.samples.natality`
WHERE
  state != ''
GROUP BY
  state, sex
HAVING
  cnt > 3000000
ORDER BY
  cnt DESC;
        """
response10 = sampleTables.query_to_pandas_safe(query10, max_gb_scanned=10)
response10.head(10)



