import bq_helper
from bq_helper import BigQueryHelper
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package
stackOverflow = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="stackoverflow")
bq_assistant = BigQueryHelper("bigquery-public-data", "stackoverflow")
bq_assistant.list_tables()
bq_assistant.head("comments", num_rows=20)
bq_assistant.table_schema("comments")
query1 = """SELECT
  EXTRACT(YEAR FROM creation_date) AS Year,
  COUNT(*) AS Number_of_Questions,
  ROUND(100 * SUM(IF(answer_count > 0, 1, 0)) / COUNT(*), 1) AS Percent_Questions_with_Answers
FROM
  `bigquery-public-data.stackoverflow.posts_questions`
GROUP BY
  Year
HAVING
  Year > 2008 AND Year < 2016
ORDER BY
  Year;
        """
response1 = stackOverflow.query_to_pandas_safe(query1)
response1.head(10)
query2 = """SELECT User_Tenure,
       COUNT(1) AS Num_Users,
       ROUND(AVG(reputation)) AS Avg_Reputation,
       ROUND(AVG(num_badges)) AS Avg_Num_Badges
FROM (
  SELECT users.id AS user,
         ROUND(TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), ANY_VALUE(users.creation_date), DAY)/365) AS user_tenure,
         ANY_VALUE(users.reputation) AS reputation,
         SUM(IF(badges.user_id IS NULL, 0, 1)) AS num_badges
  FROM `bigquery-public-data.stackoverflow.users` users
  LEFT JOIN `bigquery-public-data.stackoverflow.badges` badges
  ON users.id = badges.user_id
  GROUP BY user
)
GROUP BY User_Tenure
ORDER BY User_Tenure;
        """
response2 = stackOverflow.query_to_pandas_safe(query2)
response2.head(10)
query3 = """SELECT badge_name AS First_Gold_Badge,
       COUNT(1) AS Num_Users,
       ROUND(AVG(tenure_in_days)) AS Avg_Num_Days
FROM
(
  SELECT
    badges.user_id AS user_id,
    badges.name AS badge_name,
    TIMESTAMP_DIFF(badges.date, users.creation_date, DAY) AS tenure_in_days,
    ROW_NUMBER() OVER (PARTITION BY badges.user_id
                       ORDER BY badges.date) AS row_number
  FROM
    `bigquery-public-data.stackoverflow.badges` badges
  JOIN
    `bigquery-public-data.stackoverflow.users` users
  ON badges.user_id = users.id
  WHERE badges.class = 1
)
WHERE row_number = 1
GROUP BY First_Gold_Badge
ORDER BY Num_Users DESC
LIMIT 10;
        """
response3 = stackOverflow.query_to_pandas_safe(query3, max_gb_scanned=10)
response3.head(10)
query4 = """SELECT
  Day_of_Week,
  COUNT(1) AS Num_Questions,
  SUM(answered_in_1h) AS Num_Answered_in_1H,
  ROUND(100 * SUM(answered_in_1h) / COUNT(1),1) AS Percent_Answered_in_1H
FROM
(
  SELECT
    q.id AS question_id,
    EXTRACT(DAYOFWEEK FROM q.creation_date) AS day_of_week,
    MAX(IF(a.parent_id IS NOT NULL AND
           (UNIX_SECONDS(a.creation_date)-UNIX_SECONDS(q.creation_date))/(60*60) <= 1, 1, 0)) AS answered_in_1h
  FROM
    `bigquery-public-data.stackoverflow.posts_questions` q
  LEFT JOIN
    `bigquery-public-data.stackoverflow.posts_answers` a
  ON q.id = a.parent_id
  WHERE EXTRACT(YEAR FROM a.creation_date) = 2016
    AND EXTRACT(YEAR FROM q.creation_date) = 2016
  GROUP BY question_id, day_of_week
)
GROUP BY
  Day_of_Week
ORDER BY
  Day_of_Week;
        """
response4 = stackOverflow.query_to_pandas_safe(query4, max_gb_scanned=10)
response4.head(10)
