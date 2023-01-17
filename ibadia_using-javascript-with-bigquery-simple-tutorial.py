import os
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from bq_helper import BigQueryHelper
bq_assistant = BigQueryHelper("bigquery-public-data", "github-repos")
javascript_query='''
CREATE TEMPORARY FUNCTION 
myFunc(mystring STRING)
RETURNS STRING
LANGUAGE js AS
"""
var start=mystring.indexOf('[');
var end=mystring.indexOf(']');
var res='nothing';
if (start==-1 || end==-1){
    res='nothing';
}else{
 res = mystring.slice(start+1, end);
}

return res;
""";
'''
print (javascript_query)
my_sql_query='''SELECT myFunc(names) as EDITED_NAME 
FROM UNNEST(["H[ibbu]ah", "Max", "Jakob","HELLO WORLD[SDSDsdsd]"]) AS names'''

final_query=javascript_query+my_sql_query
print (final_query)
bq_assistant.estimate_query_size(final_query)
## Seems like we are good to go
result=bq_assistant.query_to_pandas_safe(final_query)
print (result)