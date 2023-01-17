from google.cloud import bigquery
client = bigquery.Client()
dataset_ref = client.dataset('hacker_news', project = 'bigquery-public-data')
dataset = client.get_dataset(dataset_ref)
tables = list(client.list_tables(dataset))
for i in tables:
    print(i.table_id)
# we will need twi tables
### Comments and stories
table_ref = dataset_ref.table('comments')
table = client.get_table(table_ref)
# view the top rows of table
client.list_rows(table, max_results=5).to_dataframe()
# this is comments table
# same like this show the stories table
table_ref = dataset_ref.table('stories')
table = client.get_table(table_ref)
client.list_rows(table, max_results = 5).to_dataframe()
# this is stories table
### Approach
##### If someone has commented then its is 100% sure that he has commented on someones story
#### but it is also possible that a story does not have any comment
#### Therfore we will have 'stories' as our left table and 'comments' as our right table
# to make your query run faster try to use your right table in left join in CTE
# and vice versa

query = """
             WITH c AS
             (
             SELECT parent, COUNT(*) as num_comments
             FROM `bigquery-public-data.hacker_news.comments` 
             GROUP BY parent
             )
             
             SELECT s.id as story_id, s.by, s.title, c.num_comments
             FROM `bigquery-public-data.hacker_news.stories` AS s
             LEFT JOIN c
             ON s.id = c.parent
             WHERE EXTRACT(DATE FROM s.time_ts) = '2012-01-01'
             ORDER BY c.num_comments DESC
    
        """

join_result = client.query(query).result().to_dataframe()
join_result.head()
join_result.tail()
union_query = """
              SELECT c.by
              FROM `bigquery-public-data.hacker_news.comments` AS c
              WHERE EXTRACT(DATE FROM c.time_ts) = '2014-01-01'
              UNION DISTINCT
              SELECT s.by
              FROM `bigquery-public-data.hacker_news.stories` AS s
              WHERE EXTRACT(DATE FROM s.time_ts) = '2014-01-01'
              """

# Run the query, and return a pandas DataFrame
union_result = client.query(union_query).result().to_dataframe()
union_result.head()
# To get the number of users who posted on January 1, 2014, we need only take the length of the DataFrame.

len(union_result)
