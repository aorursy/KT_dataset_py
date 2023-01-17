from google.cloud import bigquery



# Create a "Client" object

client = bigquery.Client()
query = """

         SELECT 'table: USERS' as table,

                 id AS user_id,

                 MIN(creation_date) as min_date,

                 'date of joinning Stack Overflow' as date_description

             FROM `bigquery-public-data.stackoverflow.users`

             WHERE id IN (11486952, 10904707, 10977933, 11040013, 10600000)

             GROUP BY table, user_id

         UNION ALL

         SELECT 'table: POSTS_QUESTIONS' as table,

                 owner_user_id AS user_id,

                 MIN(creation_date) as min_date,

                 'date of first question' as date_description

             FROM `bigquery-public-data.stackoverflow.posts_questions`

             WHERE owner_user_id IN (11486952, 10904707, 10977933, 11040013, 10600000)

             GROUP BY table, user_id

         UNION ALL

         SELECT 'table: POSTS_ANSWERS' as table,

                 owner_user_id AS user_id,

                 MIN(creation_date) as min_date,

                 'date of first answer' as date_description

             FROM `bigquery-public-data.stackoverflow.posts_answers`                     

             WHERE owner_user_id IN (11486952, 10904707, 10977933, 11040013, 10600000)

             GROUP BY table, user_id

                    """



user = client.query(query).result().to_dataframe()

user.sort_values('user_id')
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

             AND u.id IN (11486952, 10904707, 10977933, 11040013, 10600000)

             GROUP BY id

                    """



client.query(three_tables_query).result().to_dataframe()

three_tables_query = """

         SELECT q.owner_user_id AS q_id,

             a.owner_user_id as a_id,

             MIN(q.creation_date) AS q_creation_date,

             MIN(a.creation_date) AS a_creation_date

         FROM `bigquery-public-data.stackoverflow.posts_questions` AS q

             FULL JOIN `bigquery-public-data.stackoverflow.posts_answers` AS a

                 ON q.owner_user_id = a.owner_user_id 

         WHERE q.owner_user_id IN (11486952, 10904707, 10977933, 11040013, 10600000)

             or a.owner_user_id IN (11486952, 10904707, 10977933, 11040013, 10600000)

         GROUP BY q_id,a_id

                    """



client.query(three_tables_query).result().to_dataframe()

three_tables_query = """

             SELECT u.id AS id,

                 MIN(q.creation_date) AS q_creation_date,

                 MIN(a.creation_date) AS a_creation_date

             FROM `bigquery-public-data.stackoverflow.posts_questions` AS q

                 FULL JOIN `bigquery-public-data.stackoverflow.posts_answers` AS a

                     ON q.owner_user_id = a.owner_user_id 

                 RIGHT JOIN `bigquery-public-data.stackoverflow.users` AS u

                     ON COALESCE(q.owner_user_id, a.owner_user_id) = u.id

             WHERE u.creation_date >= '2019-01-01' and u.creation_date < '2019-02-01'

             AND u.id IN (11486952, 10904707, 10977933, 11040013, 10600000)

             GROUP BY id

                    """



client.query(three_tables_query).result().to_dataframe()

# Your code here

three_tables_query = """

            SELECT u.id,

                MIN(q.creation_date) AS q_creation_date,

                MIN(a.creation_date) AS a_creation_date

            FROM `bigquery-public-data.stackoverflow.users` AS u

            LEFT JOIN `bigquery-public-data.stackoverflow.posts_questions` AS q

                ON u.id = q.owner_user_id

            LEFT JOIN `bigquery-public-data.stackoverflow.posts_answers` AS a

                ON u.id = a.owner_user_id 

            WHERE u.creation_date >= '2019-01-01' AND u.creation_date < '2019-02-01' 

            AND u.id IN (11486952, 10904707, 10977933, 11040013, 10600000)

            GROUP BY u.id

                     """





client.query(three_tables_query).result().to_dataframe()
