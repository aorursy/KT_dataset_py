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
# Your code goes here :)

#How many stories (use the "id" column) are there of each type (in the "type" column) in the full table?
hacker_news.head("full")
    
# Qry1 to count stories for each type from full table
qry1 = """SELECT count(distinct id) as id,type
            FROM `bigquery-public-data.hacker_news.full`
            --WHERE type='story' 
            GROUP BY type            
        """
print('Verify query estimationmy query',hacker_news.estimate_query_size(qry1))
pd_numofstory = hacker_news.query_to_pandas_safe(qry1)
print ('Number of stories for each Type - ' , pd_numofstory.head())

#Qry2 to count deleted comments from comment table
qry2="""SELECT count(deleted)
            FROM `bigquery-public-data.hacker_news.full`
            WHERE deleted =True            
        """
#estomate query result
hacker_news.estimate_query_size(qry2)

#load query result to panda dataframe
pd_numofdel = hacker_news.query_to_pandas_safe(qry2)

#list the result
print('Number of deleted comments are - ',pd_numofdel.head())

#qry3 Use other aggregate function (I want to try avg and having function)

qry3 = """
            SELECT distinct type,avg(time) as time,count(parent) as prnt
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
            HAVING count(parent)>0
        """
#estomate query result
hacker_news.estimate_query_size(qry3)
#load query result to panda dataframe
pd_testqry = hacker_news.query_to_pandas_safe(qry3)
#list the result
print('Average time for each type - \n',pd_testqry.head())
