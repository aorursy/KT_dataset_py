import bq_helper
from bq_helper import BigQueryHelper
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package
stackOverflow = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="stackoverflow")
stackOverflow.list_tables()
stackOverflow.table_schema("users")
stackOverflow.table_schema("badges")
stackOverflow.table_schema("comments")
query = """SELECT reputation FROM `bigquery-public-data.stackoverflow.users` LIMIT 10;"""
response1 = stackOverflow.query_to_pandas_safe(query)
response1.head(10)
stackOverflow.table_schema("votes")
stackOverflow.table_schema("badges")
stackOverflow.table_schema("posts_questions")
query = """SELECT question_id, question FROM (
SELECT
q.title AS question,
q.id AS question_id,
FROM `bigquery-public-data.stackoverflow.posts_questions` q
JOIN `bigquery-public-data.stackoverflow.badges` b
ON q.id = b.id 
WHERE b.name = 'Popular Question' AND q.body LIKE '% home router %');"""
homeResponse = stackOverflow.query_to_pandas_safe(query, max_gb_scanned=500)
responseAsArray = homeResponse.values
for item in responseAsArray:
    print(item)
stackOverflow.table_schema("posts_answers")
stackOverflow.table_schema("posts_questions")
query = """SELECT tag_name FROM `bigquery-public-data.stackoverflow.tags` WHERE tag_name = 'networking';"""
response1 = stackOverflow.query_to_pandas_safe(query)
response1.head(10)
query = """SELECT title, accepted_answer_id FROM `bigquery-public-data.stackoverflow.posts_questions` WHERE body LIKE '% home network %' AND accepted_answer_id <> 'nan';"""
homeResponse = stackOverflow.query_to_pandas_safe(query, max_gb_scanned=500)
responseAsArray = homeResponse.values
for item in responseAsArray:
    print(item)
stackOverflow.table_schema("posts_answers")
stackOverflow.table_schema("comments")
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
query_1 = "SELECT title FROM `bigquery-public-data.stackoverflow.posts_questions` WHERE body LIKE '% home network %' AS test COUNT (test);"
query_2 = "COUNT(test)"
response1 = stackOverflow.query_to_pandas_safe(query_1)
response1.head(10)
response2 = stackOverflow.query_to_pandas_safe(query_2)
response2.head(10)
query4 = "SELECT User_Tenure;"
response4 = stackOverflow.query_to_pandas_safe(query4)
response4.head(10)
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
testQuery = """SELECT User_Tenure FROM `bigquery-public-data.stackoverflow.posts_users`;"""
testResponse = stackOverflow.query_to_pandas_safe(testQuery, max_gb_scanned=10)
testResponse.head(10)
testQuery = """SELECT creation_date, title, answer_count FROM `bigquery-public-data.stackoverflow.posts_questions` ORDER BY creation_date LIMIT 1;"""
testResponse = stackOverflow.query_to_pandas_safe(testQuery, max_gb_scanned=10)
testResponse.head(10)
testQuery = """SELECT creation_date, title, answer_count FROM `bigquery-public-data.stackoverflow.posts_questions` ORDER BY creation_date DESC LIMIT 1;"""
testResponse = stackOverflow.query_to_pandas_safe(testQuery, max_gb_scanned=10)
testResponse.head(10)
testQuery = """SELECT creation_date FROM `bigquery-public-data.stackoverflow.posts_answers` ORDER BY creation_date LIMIT 1;"""
testResponse = stackOverflow.query_to_pandas_safe(testQuery, max_gb_scanned=10)
testResponse.head(10)
testQuery = """SELECT creation_date FROM `bigquery-public-data.stackoverflow.posts_answers` ORDER BY creation_date DESC LIMIT 1;"""
testResponse = stackOverflow.query_to_pandas_safe(testQuery, max_gb_scanned=10)
testResponse.head(10)
homeQuery = """SELECT body FROM `bigquery-public-data.stackoverflow.posts_questions` 
                WHERE EXTRACT(YEAR FROM creation_date) BETWEEN 2014 AND 2019
                AND body LIKE '% slow internet %'
                ORDER BY creation_date DESC;"""
homeResponse = stackOverflow.query_to_pandas_safe(homeQuery, max_gb_scanned=30)
responseAsArray = homeResponse.values
for item in responseAsArray:
    print(item)

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
matchingQuery = """SELECT comment_user_id, owner_user_id, question, question_id, question_body, answer_body, user_comment
FROM
(
  SELECT 
    c.user_id AS comment_user_id,
    q.owner_user_id AS owner_user_id,
    q.title AS question,
    q.body AS question_body,
    a.body AS answer_body,
    c.text AS user_comment,
    q.id AS question_id,
  FROM
    `bigquery-public-data.stackoverflow.posts_questions` q
  JOIN
    `bigquery-public-data.stackoverflow.posts_answers` a
  ON q.id = a.parent_id
  JOIN
    `bigquery-public-data.stackoverflow.comments` c
  ON q.id = c.post_id
  WHERE ((c.user_id = q.owner_user_id) OR (c.user_display_name = q.owner_display_name)) AND ((q.body LIKE '% home router %' AND q.accepted_answer_id = a.id) OR (q.body LIKE '% home router %' AND c.text LIKE '% worked %'))
);
"""

matchingResponse = stackOverflow.query_to_pandas_safe(matchingQuery, max_gb_scanned=70)
matchingResponse.head(70)
matchingQuery = """SELECT question_id, question_date, question, answer_body, user_comment
FROM
(
  SELECT 
    c.user_id AS comment_user_id,
    q.owner_user_id AS owner_user_id,
    q.title AS question,
    q.body AS question_body,
    a.body AS answer_body,
    c.text AS user_comment,
    q.id AS question_id,
    q.creation_date AS question_date,
  FROM
    `bigquery-public-data.stackoverflow.posts_questions` q
  JOIN
    `bigquery-public-data.stackoverflow.posts_answers` a
  ON q.id = a.parent_id
  JOIN
    `bigquery-public-data.stackoverflow.comments` c
  ON q.id = c.post_id
  WHERE ((c.user_id = q.owner_user_id) OR (c.user_display_name = q.owner_display_name)) AND ((q.body LIKE '% home router %' AND q.accepted_answer_id = a.id) OR (q.body LIKE '% home router %' AND c.text LIKE '% worked %'))
);
"""

matchingResponse = stackOverflow.query_to_pandas_safe(matchingQuery, max_gb_scanned=70)
matchingResponse.to_csv('/kaggle/working/results.csv', index=False)
matchingResponse.head(70)
matchingQuery = """SELECT question_id, question_monthandyear, question
FROM
(
  SELECT 
    c.user_id AS comment_user_id,
    q.owner_user_id AS owner_user_id,
    q.title AS question,
    q.body AS question_body,
    a.body AS answer_body,
    c.text AS user_comment,
    q.id AS question_id,
    CONCAT((EXTRACT(MONTH FROM q.creation_date)), ' ', (EXTRACT(YEAR FROM q.creation_date))) AS question_monthandyear,
  FROM
    `bigquery-public-data.stackoverflow.posts_questions` q
  JOIN
    `bigquery-public-data.stackoverflow.posts_answers` a
  ON q.id = a.parent_id
  JOIN
    `bigquery-public-data.stackoverflow.comments` c
  ON q.id = c.post_id
  WHERE ((c.user_id = q.owner_user_id) OR (c.user_display_name = q.owner_display_name)) AND ((q.body LIKE '% lan %' AND q.accepted_answer_id = a.id) OR (q.body LIKE '% lan %' AND c.text LIKE '% worked %')) AND (EXTRACT(YEAR FROM q.creation_date) > 2018) AND (EXTRACT(YEAR FROM q.creation_date) < 2021));
"""

matchingResponse = stackOverflow.query_to_pandas_safe(matchingQuery, max_gb_scanned=70)
matchingResponse.head(70)
matchingQuery = """SELECT question_id, time, question_body, answer_body, user_comment
FROM
(
  SELECT 
    c.user_id AS comment_user_id,
    q.owner_user_id AS owner_user_id,
    q.title AS question,
    q.body AS question_body,
    a.body AS answer_body,
    c.text AS user_comment,
    q.id AS question_id,
    q.creation_date AS time,
    EXTRACT(MONTH FROM q.creation_date) AS month,
    EXTRACT(YEAR FROM q.creation_date) AS year,
  FROM
    `bigquery-public-data.stackoverflow.posts_questions` q
  JOIN
    `bigquery-public-data.stackoverflow.posts_answers` a
  ON q.id = a.parent_id
  JOIN
    `bigquery-public-data.stackoverflow.comments` c
  ON q.id = c.post_id
  JOIN
    `bigquery-public-data.stackoverflow.badges` b
    ON b.user_id = q.owner_user_id
  WHERE ((c.user_id = q.owner_user_id) OR (c.user_display_name = q.owner_display_name)) AND 
  (q.title LIKE '%wifi%') AND 
  (b.name = 'Yearling'));
"""
matchingResponse = stackOverflow.query_to_pandas_safe(matchingQuery, max_gb_scanned=800)
matchingResponse.to_csv('/kaggle/working/results.csv', index=False)
matchingResponse.head(800)
matchResponsesAsArray = matchingResponse.values
for item in matchResponsesAsArray:
    print(item)