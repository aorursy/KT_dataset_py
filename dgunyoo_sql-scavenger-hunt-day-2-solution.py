import bq_helper

hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

query1 = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
number = hacker_news.query_to_pandas_safe(query1)

query2 = """SELECT COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE deleted = True
        """
deleted = hacker_news.query_to_pandas_safe(query2)

query3 = """SELECT type, sum(score)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
            having sum(score) > 0
        """
more = hacker_news.query_to_pandas_safe(query3)

print('Q1\n',number,'\n\nQ2\n',deleted,'\n\nQ3\n',more)