import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("comments")

# Which Hacker News comments generated the most discussion?
query = """SELECT parent, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY parent
            HAVING COUNT(id) > 10
        """
popular_stories = hacker_news.query_to_pandas_safe(query)
popular_stories.head()
#How many stories (use the "id" column) are there of each type (in the "type" column) in the full table?
hacker_news.head("full")
query="SELECT type,COUNT(id) as no_of_stories FROM `bigquery-public-data.hacker_news.full` group by type"
no_of_stories=hacker_news.query_to_pandas_safe(query)
no_of_stories.head()
#How many comments have been deleted? 
hacker_news.head("comments")
query="""SELECT COUNT(deleted) as no_of_comments_deleted 
        from `bigquery-public-data.hacker_news.comments`
        where deleted=True """
no_of_comments_deleted=hacker_news.query_to_pandas_safe(query)
no_of_comments_deleted.head()
#Optional credit
query="""SELECT COUNTIF(deleted=True) AS no_of_comments_deleted 
         from `bigquery-public-data.hacker_news.comments` """
no_of_comments_deleted=hacker_news.query_to_pandas_safe(query)
no_of_comments_deleted.head()