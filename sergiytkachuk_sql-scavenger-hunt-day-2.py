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
import bq_helper
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

def generate_query(columns, group_by, func = ['COUNT'], agg = 'id',
                   database = 'bigquery-public-data.hacker_news',
                   table = 'full_201510'):
    column = ', '.join(item for item in columns)
    agg_str = '(' + agg + '), '
    function = agg_str.join(item for item in func)
    group = ', '.join(item for item in group_by)
    query = "SELECT " + column + ', ' + function + '(' + agg + ')' + " FROM" + " `" + database + "." + table + "` " + "GROUP BY " + group
    return query

# How many stories (use the "id" column) are there of each type (in the "type" column) in the full table?
types_count = hacker_news.query_to_pandas_safe(generate_query(['type'], ['type']))
print(types_count.head())

#How many comments have been deleted?
#(If a comment was deleted the "deleted" column in the comments table will have the value "True".)
deleted_count_query = """SELECT COUNT(*)
                            FROM `bigquery-public-data.hacker_news.comments`
                            where deleted = True
                        """
deleted_count = hacker_news.query_to_pandas_safe(deleted_count_query)
print(deleted_count.head())

avg_score_qury = """SELECT deleted, round(AVG(ranking), 2)
                            FROM `bigquery-public-data.hacker_news.comments`
                            group by deleted
                        """
deleted_avg_score = hacker_news.query_to_pandas_safe(avg_score_qury)
print(deleted_avg_score.head())