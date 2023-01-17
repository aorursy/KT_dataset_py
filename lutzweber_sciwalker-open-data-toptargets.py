PROJECT_ID = 'sciwalker-open-data'

from google.cloud import bigquery

import pandas as pd

client = bigquery.Client(project=PROJECT_ID)
query = (

    "select count(a1.docId) as counts, a1.ocid as ocid, a2.name as name, substring(cast( a1.date as STRING ),0,4) as year " +

    "from `sciwalker-open-data.Grant_applications.NIH_grant_applications` a1 " +

    "left join `sciwalker-open-data.ontologies.proteins_preflabel` a2 on a1.ocid=a2.ocid " +

    "where a1.ocid > 101000000000 AND a1.ocid < 101810000000 AND a1.date > cast ( '2010-01-01T00:00:00' as DATETIME ) " +

    "group by a1.ocid, a2.name, year " +

    "having counts > 8 " +

    "order by year desc "

)



query_job = client.query(

    query,

    location="US",

)  



df = query_job.to_dataframe()



df.head()



#for row in query_job: 

#    print( row["ocid"], row["counts"], row["name"] )

    

    
df.head()