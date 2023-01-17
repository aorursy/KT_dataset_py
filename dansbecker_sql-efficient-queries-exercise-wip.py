from google.cloud import bigquery



# Create client object to access database

client = bigquery.Client()



# Specify dataset for high level overview of data

dataset_ref = client.dataset("stackoverflow", "bigquery-public-data")

dataset = client.get_dataset(dataset_ref)



# List all the tables

tables = client.list_tables(dataset)

for table in tables:  

    print(table.table_id)
table_ref = dataset_ref.table("posts_answers")

table = client.get_table(table_ref)

# See the first five rows of data

client.list_rows(table, max_results=5).to_dataframe()
table_ref = dataset_ref.table("posts_questions")

table = client.get_table(table_ref)

client.list_rows(table, max_results=5).to_dataframe()
bytes_per_gb = 2**30

job_config = bigquery.QueryJobConfig(dry_run=True)

result = client.query(

    (

"""

WITH bq_questions as (

SELECT title, accepted_answer_id 

FROM `bigquery-public-data.stackoverflow.posts_questions` 

WHERE tags like '%bigquery%' and accepted_answer_id is not NULL

)

SELECT ans.* 

FROM bq_questions inner join `bigquery-public-data.stackoverflow.posts_answers` ans

ON ans.Id = bq_questions.accepted_answer_id

"""

    ),

    job_config=job_config

)

print("This query will process {} GB.".format(result.total_bytes_processed // bytes_per_gb))
bytes_per_gb = 2**30

job_config = bigquery.QueryJobConfig(maximum_bytes_billed=1)

result = client.query(

    (

"""

WITH bq_questions as (

SELECT title, accepted_answer_id 

FROM `bigquery-public-data.stackoverflow.posts_questions` 

WHERE tags like '%bigquery%' and accepted_answer_id is not NULL

)

SELECT ans.* 

FROM bq_questions inner join `bigquery-public-data.stackoverflow.posts_answers` ans

ON ans.Id = bq_questions.accepted_answer_id

"""

    ),

    job_config=job_config

)

result.to_dataframe()
query = """

        WITH bq_questions AS 

        (

            SELECT title, accepted_answer_id 

            FROM `bigquery-public-data.stackoverflow.posts_questions` 

            WHERE tags like '%bigquery%' and accepted_answer_id is not NULL

        )

        SELECT ans.* 

        FROM bq_questions 

        INNER JOIN `bigquery-public-data.stackoverflow.posts_answers` ans

            ON ans.Id = bq_questions.accepted_answer_id

        """



result = client.query(query).result().to_dataframe()
result.head()