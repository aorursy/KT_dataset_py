#setup
import bq_helper
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                      dataset_name="hacker_news")
#view first several rows of the comments table
hacker_news.head("comments")
#The "parent" column has information on the comment that each comment was a reply to and 
#the "id" column has the unique id used to identify each comment.
#So group by the "parent" column and count the "id" column in order to figure out 
#the number of comments that were made as responses to a specific comment.

#Show popular comments only, thus filtering having count(id)>10
query = """SELECT parent, COUNT(ID)
           FROM `bigquery-public-data.hacker_news.comments`
           GROUP BY parent
           HAVING COUNT(ID)>10
"""
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
popular_stories = hacker_news.query_to_pandas_safe(query)

#show the head of the dataframe
popular_stories.head()

#Why is the column with the COUNT() data called f0_? It's called this because COUNT() 
#is the first (and in our case, only) aggregate function we used in this query. 
#If we'd used a second one, it would be called "f1_", the third would be called "f2_", and so on. 
#Q1: how many stories (use ID col) are there in each type ("type" col) in the full table
#check what's in full table
hacker_news.head("full")

#write the query
query1="""SELECT type, COUNT(ID)
          FROM `bigquery-public-data.hacker_news.full`
          GROUP BY type
"""
#types number
types_count = hacker_news.query_to_pandas_safe(query1)

#show the dataframe
types_count
#Q2: How many comments have been deleted?
#(If a comment was deleted the "deleted" column in the comments table will have the value "True".)

query2="""SELECT COUNT(ID)
          FROM `bigquery-public-data.hacker_news.comments`
          WHERE deleted = True
"""
#types number
deleted_count = hacker_news.query_to_pandas_safe(query2)

#print the result
deleted_count

hacker_news.head("full")
#Q3: Optional extra credit: read about aggregate functions other than COUNT() and modify 
#one of the queries you wrote above to use a different aggregate function.

#Average score (score col) per comment type (type col) in full table 

#write the query
query3= """SELECT type, AVG(score)
          FROM `bigquery-public-data.hacker_news.full`
          GROUP BY type"""
score_avg = hacker_news.query_to_pandas_safe(query3)
score_avg