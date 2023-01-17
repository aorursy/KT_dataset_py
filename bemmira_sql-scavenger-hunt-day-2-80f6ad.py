# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("comments")
maRequete = """SELECT count (id), parent 
                 FROM `bigquery-public-data.hacker_news.comments`
                 GROUP By parent
                 HAVING count (parent) > 10
                 
            """

populaire_stories = hacker_news.query_to_pandas_safe(maRequete)
populaire_stories.head()
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
popular_stories.head ()
#  How many stories (use the "id" column) are there of each type
# (in the "type" column) in the full table?

maRequete = """SELECT  type, COUNT (id)
                 FROM `bigquery-public-data.hacker_news.full`
                 GROUP By type
            """
hacker_news.head("full")
stories_by_type = hacker_news.query_to_pandas_safe(maRequete)
stories_by_type.head ()
#* How many comments have been deleted? (If a comment was deleted the "deleted" 
# column in the comments table will have the value "True".)
requeteDelCmt = """SELECT id, count (parent), deleted
 FROM `bigquery-public-data.hacker_news.comments`
                     GROUP BY id, deleted
                     HAVING deleted = 'True'
                 """

requete_Deltd_comments =  hacker_news.query_to_pandas_safe(requeteDelCmt)
requete_Deltd_comments.head ()