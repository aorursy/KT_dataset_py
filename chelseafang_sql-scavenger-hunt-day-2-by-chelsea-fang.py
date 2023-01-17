'''import bq_helper as bq
pets=bq.BigQueryHelper(active_project="bigquery-public-data",
                      dataset_name="pet_records")
query1="""
          SELECT Animal, COUNT(ID)
          FROM `bigquery-public-data.pet_records.pets`
          GROUP BY Animal
"""
result1=pets.query_to_pandas_safe(query1)
'''
import bq_helper as bq
hacker_news=bq.BigQueryHelper(active_project='bigquery-public-data',
                             dataset_name='hacker_news')
hacker_news.head('comments')

query = """SELECT parent, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY parent
            HAVING COUNT(id) > 10
        """
popular_stories = hacker_news.query_to_pandas_safe(query)
print(popular_stories)
query3="""
       SELECT COUNT(id),type
       FROM `bigquery-public-data.hacker_news.full`
       GROUP BY type
"""
result3= hacker_news.query_to_pandas_safe(query3)
print(result3)
query4="""
          SELECT COUNT(deleted)
          FROM `bigquery-public-data.hacker_news.comments`
          WHERE deleted = True
""" 
result4=hacker_news.query_to_pandas_safe(query4)
print(result4)
