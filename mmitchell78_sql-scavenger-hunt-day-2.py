import bq_helper
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")
storycountquery =     """ 
            SELECT  Type,
                    Count(ID) AS `StoryCount`
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY Type
            """
hacker_news.query_to_pandas_safe(storycountquery)            
deletedstorycountquery1 =     """ 
                                SELECT  Count(ID) AS `DeletedCommentCount`
                                FROM    `bigquery-public-data.hacker_news.comments`
                                WHERE   Deleted is not NULL
                             """
hacker_news.query_to_pandas_safe(deletedstorycountquery1)
deletedstorycountquery2 =     """ 
                                SELECT  Count(ID) AS `DeletedCommentCount`
                                FROM    `bigquery-public-data.hacker_news.full`
                                WHERE   Type LIKE "comment" 
                                        AND 
                                        Deleted is not NULL
                              """
hacker_news.query_to_pandas_safe(deletedstorycountquery2)
extracreditquery =     """ 
                                SELECT  SUM( 
                                            CASE 
                                                WHEN Deleted is not NULL 
                                                THEN 1 
                                                ELSE 0 
                                            END
                                            ) AS `Deleted`
                                FROM    `bigquery-public-data.hacker_news.comments`
                              """
hacker_news.query_to_pandas_safe(extracreditquery) 