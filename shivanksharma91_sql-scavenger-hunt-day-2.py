# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("comments")
# query to pass to 
query = """SELECT parent, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY parent
            HAVING COUNT(id) > 10
        """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
popular_stories = hacker_news.query_to_pandas_safe(query)
popular_stories.head()
#import package with helper function
import bq_helper

#creating helper object for data set
hacker_news=bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                    dataset_name="hacker_news")

query1=""" SELECT count(id) as Nbr_of_Stories,type
           FROM `bigquery-public-data.hacker_news.full`
           GROUP BY 2
       """
nbr_of_stories=hacker_news.query_to_pandas_safe(query1)
nbr_of_stories


#Counting number of comments deleted.
query2=""" SELECT count(id) AS Nbr_Of_Comments_Deleted,deleted
           FROM `bigquery-public-data.hacker_news.comments`
           WHERE deleted=True
           GROUP BY 2
       """
nbr_of_deletion=hacker_news.query_to_pandas_safe(query2)
nbr_of_deletion
#Extra points. Calculating Maximum score

query3=""" SELECT avg(score) AS Average_Score,author
           FROM `bigquery-public-data.hacker_news.stories`
           WHERE descendants>4 and text IS NOT NULL
          GROUP BY 2
       """
avg_score=hacker_news.query_to_pandas_safe(query3)
avg_score