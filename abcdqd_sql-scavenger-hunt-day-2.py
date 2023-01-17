import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper as bq
hacker_news = bq.BigQueryHelper(active_project = 'bigquery-public-data',
                               dataset_name = 'hacker_news')
hacker_news.list_tables()
hacker_news.head('comments', num_rows = 10)
hacker_news.head('full', num_rows = 10)
hacker_news.head('stories', num_rows = 10)
query = """
        with unique_table as 
        (select * from
        (select
            id,
            type,
            row_number() over (partition by id order by type desc) as dup_count
        from `bigquery-public-data.hacker_news.full`)
        where dup_count = 1)
        select
            type,
            count(id) as count
        from unique_table
        group by type
        order by type desc
        """
hacker_news.query_to_pandas(query)
query = """
        select 
            deleted,
            count(id) as count
        from `bigquery-public-data.hacker_news.comments`
        group by deleted
        """
hacker_news.query_to_pandas(query)