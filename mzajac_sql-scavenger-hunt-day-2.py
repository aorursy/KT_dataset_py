# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# task 1
query1 = """SELECT type as Type, 
                    COUNT(id) as NbStories
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """

popular_stories = hacker_news.query_to_pandas_safe(query1)

print('How many stories (use the "id" column) are there of each type (in the "type" column) in the full table?')
print (popular_stories)
# task 2
query2 = """SELECT COUNT(id) as DeletedComments
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE deleted IS NOT NULL
        """

deleted_comments = hacker_news.query_to_pandas_safe(query2)

print('How many comments have been deleted?')
for DeletedComments in deleted_comments.DeletedComments.unique():
    print (DeletedComments)
# additional task
query3 = """SELECT `by` as Author,
                    type as Type,
                    AVG(score) as AverageScore
            FROM `bigquery-public-data.hacker_news.full`
            WHERE TIMESTAMP_TRUNC(timestamp, MONTH) BETWEEN '2017-01-01' AND '2017-12-01'
                    AND type !='comment'
            GROUP BY Author, Type
            HAVING AverageScore > 1000
            ORDER BY AverageScore desc            
        """

author_avg_score = hacker_news.query_to_pandas_safe(query3)

print('What is the Average Score for Author in FY2017 (above 1k average score)?')
print(author_avg_score)