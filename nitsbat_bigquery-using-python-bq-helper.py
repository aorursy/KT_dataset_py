import bq_helper
hacker = bq_helper.BigQueryHelper(active_project = "bigquery-public-data", dataset_name = "hacker_news")
hacker.list_tables()
hacker.head('comments')
query = """    select author_name , total
               from  (select author as author_name,count(author) as total 
                from `bigquery-public-data.hacker_news.comments` group by author_name)
                where total > 500
                """
hacker.estimate_query_size(query)
pos = hacker.query_to_pandas_safe(query)
pos.head()
query1 = """    select count(DISTINCT author) 
                from `bigquery-public-data.hacker_news.comments` 
                where extract(YEAR from time_ts) = 2013 and ranking > 1
         """
hacker.estimate_query_size(query1)
hacker.query_to_pandas_safe(query1)
query = """ select type,count(id) from `bigquery-public-data.hacker_news.full`
            group by type
        """
hacker.estimate_query_size(query)
hacker.query_to_pandas_safe(query)

query = """         select `by`,type,count(`by`) as total 
                    from `bigquery-public-data.hacker_news.full`
                    group by `by`, type
                    order by total DESC
        """
hacker.estimate_query_size(query)
k = hacker.query_to_pandas_safe(query)
k.head()
query = """         WITH new_tab as
                    (
                        select `by` as author_name,type,count(`by`) as total 
                        from `bigquery-public-data.hacker_news.full`
                        group by `by`, type
                    )
                    select author_name,type,total from new_tab
                    where total = (
                    select max(total) from new_tab where type = 'comment'
                    ) or total = (
                    select max(total) from new_tab where type = 'story'
                    )
        """
hacker.estimate_query_size(query)
hacker.query_to_pandas_safe(query)

query = """         WITH new_tab as
                    (
                        select `by` as author_name,type,count(`by`) as total 
                        from `bigquery-public-data.hacker_news.full`
                        group by `by`, type
                    )
                    select a.author, b.type, b.total
                    from `bigquery-public-data.hacker_news.comments` as a 
                    join new_tab as b
                    on a.author = b.author_name
        """
hacker.estimate_query_size(query)
res = hacker.query_to_pandas_safe(query,max_gb_scanned=1)
res.head()
