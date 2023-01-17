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
#* check howlong it'll charge you( i.e. how BIG it is)
hacker_news.estimate_query_size(query)#0.1251610666513443(i..e. 125MBs)*
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
popular_stories = hacker_news.query_to_pandas_safe(query)

#              #* you may view the first 7 rows
popular_stories.head(7)
popular_stories.head()
# Your code goes here :)
## Now it's your turn! Here's the questions I would like you to get the data to answer:

###       * How many stories (use the "id" column) are there of each type (in the "type" column) in the full table?
# print the first couple rows of the "comments" table
hacker_news.head("full")
        #* my query to pass to 
query = """SELECT type, COUNT(id) as storyCount
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
            --HAVING COUNT(id) > 10
        """
         #* check howlong it'll charge you( i.e. how BIG it is)
hacker_news.estimate_query_size(query)#0.25361450482159853(i..e. ~254MBs)*

        #* Now that our query is ready, let's run it (safely!) and store the results in a dataframe:    
# the query_to_pandas_safe method will cancel the query if it would use too much of your quota, 
#with the limit set to 1 GB by default
story_types = hacker_news.query_to_pandas_safe(query)

#              #* you may view the first 7 rows
story_types.head(7)

#       *How many comments have been deleted? (If a comment was deleted the "deleted" column in the comments table will have the value "True".)
#       *Optional extra credit: read about aggregate functions other than COUNT() and modify one of the queries you wrote above to use a different aggregate function.

#       *How many comments have been deleted? (If a comment was deleted the "deleted" column in the comments table will have the value "True".)

# print the first couple rows of the "comments" table

hacker_news.head("comments")
        #* my query to pass to 
query = """SELECT deleted--type
                  , COUNT(id) as Comments
            FROM `bigquery-public-data.hacker_news.comments`-- `bigquery-public-data.hacker_news.full`
            --where type='comment' -- deleted='True'
            GROUP BY deleted--type
            --HAVING COUNT(id) > 10
            -- HAVING --type='comment' 
            
        """
         #* check howlong it'll charge you( i.e. how BIG it is)
hacker_news.estimate_query_size(query)#0.06279262900352478(i..e. ~63MBs)*

        #* Now that our query is ready, let's run it (safely!) and store the results in a dataframe:    
# the query_to_pandas_safe method will cancel the query if it would use too much of your quota, 
#with the limit set to 1 GB by default
deleted_comments = hacker_news.query_to_pandas_safe(query)

#              #* you may view the first 7 rows
deleted_comments.head(7)
#       *Optional extra credit: read about aggregate functions other than COUNT() and modify one of the queries you wrote above to use a different aggregate function.

# print the first couple rows of the "full" table
hacker_news.head("full")
hacker_news.head("comments")
hacker_news.head("full_201510")
hacker_news.head("stories")
        #* my query to pass to 
#query = """SELECT deleted--type
#                  , AVE(time) as Comments
#            FROM `bigquery-public-data.hacker_news.full`
#            where type='comment' -- deleted='True'
#            GROUP BY deleted--type
#            --HAVING COUNT(id) > 10
 #           -- HAVING --type='comment' 
#            
 #       """
         #* check howlong it'll charge you( i.e. how BIG it is)
#hacker_news.estimate_query_size(query)#0.2540750429034233(i..e. ~254MBs)*

        #* Now that our query is ready, let's run it (safely!) and store the results in a dataframe:    
# the query_to_pandas_safe method will cancel the query if it would use too much of your quota, 
#with the limit set to 1 GB by default
#deleted_comments = hacker_news.query_to_pandas_safe(query)

#              #* you may view the first 7 rows
#deleted_comments.head(7)