# import the libraries
import pandas as pd
import bq_helper
hacker_helper = bq_helper.BigQueryHelper(active_project = 'bigquery-public-data',
                                        dataset_name='hacker_news')
# see what tables are in the space
hacker_helper.list_tables()
# Look at the top of the 'full' table to see what features are available and
# the general structure
hacker_helper.head("full")
# set up a variable for the repeated portion of the space/table name:
table = "bigquery-public-data.hacker_news.full"

# Build the query
query1 = """
        SELECT type, COUNT(id) 
        FROM `{}` 
        GROUP BY type
""".format(table)

# Let's check out the query size on this to see how much of our quota it will use:
hacker_helper.estimate_query_size(query1)
# let's run that suckah!
types = hacker_helper.query_to_pandas_safe(query1)

# and look at the results
types.columns = ['type','count']
types
# set up the query:
query2 = """
        SELECT type, COUNT(id)
        FROM `{}`
        WHERE deleted=True AND type='comment'
        GROUP BY type
""".format(table)

# Check the usage
hacker_helper.estimate_query_size(query2)
# run that suckah2!
deleted = hacker_helper.query_to_pandas_safe(query2)

# Check out the results
deleted.columns = ['type', '# deleted']
deleted
# set up the query:
query3 = """
        SELECT type, AVG(score)
        FROM `{}`
        GROUP BY type
""".format(table)

# Check the usage
hacker_helper.estimate_query_size(query3)
# run that optional suckah!
avgscores = hacker_helper.query_to_pandas_safe(query3)

# check out the results
avgscores.columns = ['type','avg score']
avgscores
import matplotlib.pyplot as plt
%matplotlib inline

colors=['pink','blue','green','yellow','red']

plt.style.use('ggplot')
plt.bar(avgscores['type'], avgscores['avg score'], color=colors)
