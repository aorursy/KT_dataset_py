# import package with helper functions 
import bq_helper
import pandas as pd
# Useful functions
def converted_query_size(helper,query=''):
    """
    returns a human format for query size.
    Author: Robert Barbosa"""
    all_sizes=['GB','MB','KB','B']
    if type(helper) == bq_helper.BigQueryHelper:
        size =helper.estimate_query_size(query)
    i=0
    factor=1024
    while size<1:
        size*=factor
        i+=1
    return f'{size:.1f} {all_sizes[i]}'

def describe_tables(helper,table=None,only_name=False):
    width=104
    def print_table(t):
        print(f'Table: {t}\n{"-"*100}')
        for f in helper.table_schema(t):
                R = width - len(t) - len(f.name) - len(f.field_type)
                print(f'{f.name}: {f.description[:R]}{"..." if len(f.description)>R else ""} [{f.field_type}]')         
    if table == None:
        for t in helper.list_tables():
            print_table(t)
    else:
        if only_name:
            for f in helper.table_schema(table):
                print(f.name)
        else:
            print_table(table)

def get_tablefield_description(helper,table,field):
    for f in helper.table_schema(table):
        if field == f.name:
            return f.description   
# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("comments")
describe_tables(hacker_news,table='full')
query = """ SELECT type,count(id) as quantity 
            FROM `bigquery-public-data.hacker_news.full` group by type"""
converted_query_size(helper=hacker_news,query=query)
story_types = hacker_news.query_to_pandas_safe(query)
story_types.head()
describe_tables(hacker_news,table='comments')
 #How many comments have been deleted?
query = """ SELECT deleted,count(id) as quantity 
            FROM `bigquery-public-data.hacker_news.comments` group by deleted"""
print(converted_query_size(helper=hacker_news,query=query))
deleted_comments = hacker_news.query_to_pandas_safe(query)
deleted_comments.head()
 #How many comments have been deleted with ranking>0? 
query = """ SELECT deleted,countif(ranking>0) as quantity 
            FROM `bigquery-public-data.hacker_news.comments` group by deleted"""
print(converted_query_size(helper=hacker_news,query=query))
deleted_comments_short = hacker_news.query_to_pandas_safe(query)
deleted_comments_short.head()
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
popular_stories = hacker_news.query_to_pandas_safe(query)
popular_stories.head()