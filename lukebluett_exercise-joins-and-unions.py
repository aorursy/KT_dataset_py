# Set up feedback system

from learntools.core import binder

binder.bind(globals())

from learntools.sql_advanced.ex1 import *

print("Setup Complete")
from google.cloud import bigquery



# Create a "Client" object

client = bigquery.Client()



# Construct a reference to the "stackoverflow" dataset

dataset_ref = client.dataset("stackoverflow", project="bigquery-public-data")



# API request - fetch the dataset

dataset = client.get_dataset(dataset_ref)



# Construct a reference to the "posts_questions" table

table_ref = dataset_ref.table("posts_questions")



# API request - fetch the table

table = client.get_table(table_ref)



# Preview the first five lines of the table

client.list_rows(table, max_results=5).to_dataframe()
# Construct a reference to the "posts_answers" table

table_ref = dataset_ref.table("posts_answers")



# API request - fetch the table

table = client.get_table(table_ref)



# Preview the first five lines of the table

client.list_rows(table, max_results=5).to_dataframe()
first_query = """

              SELECT q.id AS q_id,

                  MIN(TIMESTAMP_DIFF(a.creation_date, q.creation_date, SECOND)) as time_to_answer

              FROM `bigquery-public-data.stackoverflow.posts_questions` AS q

                  INNER JOIN `bigquery-public-data.stackoverflow.posts_answers` AS a

              ON q.id = a.parent_id

              WHERE q.creation_date >= '2018-01-01' and q.creation_date < '2018-02-01'

              GROUP BY q_id

              ORDER BY time_to_answer

              """



first_result = client.query(first_query).result().to_dataframe()

print("Percentage of answered questions: %s%%" % \

      (sum(first_result["time_to_answer"].notnull()) / len(first_result) * 100))

print("Number of questions:", len(first_result))

first_result.head()
# Your code here

correct_query = """

                SELECT q.id AS q_id,

                  MIN(TIMESTAMP_DIFF(a.creation_date, q.creation_date, SECOND)) as time_to_answer

                FROM `bigquery-public-data.stackoverflow.posts_questions` AS q

                  LEFT JOIN `bigquery-public-data.stackoverflow.posts_answers` AS a

                  ON q.id = a.parent_id

                WHERE q.creation_date >= '2018-01-01' and q.creation_date < '2018-02-01'

                GROUP BY q_id

                ORDER BY time_to_answer

                """



# Check your answer

q_1.check()



# Run the query, and return a pandas DataFrame

correct_result = client.query(correct_query).result().to_dataframe()

print("Percentage of answered questions: %s%%" % \

      (sum(correct_result["time_to_answer"].notnull()) / len(correct_result) * 100))

print("Number of questions:", len(correct_result))
# Lines below will give you a hint or solution code

#q_1.hint()

#q_1.solution()
# Your code here

q_and_a_query = """

                SELECT q.owner_user_id AS owner_user_id,

                    MIN(q.creation_date) AS q_creation_date,

                    MIN(a.creation_date) AS a_creation_date

                FROM `bigquery-public-data.stackoverflow.posts_questions` AS q

                    FULL JOIN `bigquery-public-data.stackoverflow.posts_answers` AS a

                ON q.owner_user_id = a.owner_user_id 

                WHERE q.creation_date >= '2019-01-01' AND q.creation_date < '2019-02-01' 

                    AND a.creation_date >= '2019-01-01' AND a.creation_date < '2019-02-01'

                GROUP BY owner_user_id

                """



# Check your answer

q_2.check()
# Lines below will give you a hint or solution code

#q_2.hint()

#q_2.solution()
# Your code here

three_tables_query = """

                        SELECT u.id AS id, 

                                MIN(q.creation_date) AS q_creation_date,

                                MIN(a.creation_date) AS a_creation_date

                        FROM `bigquery-public-data.stackoverflow.posts_questions` AS q

                        FULL JOIN `bigquery-public-data.stackoverflow.posts_answers` AS a

                             ON q.owner_user_id = a.owner_user_id 

                        RIGHT JOIN `bigquery-public-data.stackoverflow.users` AS u

                             ON q.owner_user_id = u.id

                        WHERE u.creation_date >= '2019-01-01' and u.creation_date < '2019-02-01'

                        GROUP BY id

                     """



# Check your answer

q_3.check()
# Lines below will give you a hint or solution code

q_3.hint()

#q_3.solution()
# Your code here

all_users_query = """

                    SELECT q.owner_user_id

                    FROM `bigquery-public-data.stackoverflow.posts_questions` AS q

                    WHERE EXTRACT(DATE FROM q.creation_date) = "2019-01-01"

                    UNION DISTINCT

                    SELECT a.owner_user_id

                    FROM `bigquery-public-data.stackoverflow.posts_answers` AS a

                    WHERE EXTRACT(DATE FROM a.creation_date) = "2019-01-01"

                  """



# Check your answer

q_4.check()
# Lines below will give you a hint or solution code

#q_4.hint()

#q_4.solution()