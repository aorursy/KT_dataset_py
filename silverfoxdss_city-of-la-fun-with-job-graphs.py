import pandas as pd                             # data processing, CSV file I/O (e.g. pd.read_csv)

import pandasql                                 # https://github.com/yhat/pandasql

from pandasql import sqldf                      

pysqldf = lambda q: sqldf(q, globals())

import matplotlib.pyplot as plt                 # https://matplotlib.org/

%matplotlib inline

from graphviz import Digraph                    # https://www.graphviz.org/      graph visualization software

import networkx as nx                           # https://networkx.github.io/   software for complex networks

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
# Let's read in the prepared csv file (for more details, see my main kernel)



# While the column titles read Job_1 and Job_2 when I was first showing promotions only, 

# the data has evolved based on the  assigned category.



job_graph = pd.read_csv('../input/city-of-la-job-graph/Job_Graph.csv', header=0)   #,nrows=125)

job_graph['Job_1'] = job_graph['Job_1'].astype(str)

job_graph['Job_2'] = job_graph['Job_2'].astype(str)

jobs =  list(job_graph.Job_1.unique())

promotions = list(job_graph.Job_2.unique())

promotions[0:5]
# Let's list the available categories

query_text = "SELECT DISTINCT Category FROM job_graph order by Category asc"

job_titles_g = sqldf(query_text)   #,(current_position,current_position))

job_titles_g



# Age = Age requirement in years

# Background_Check - is a background check required?

# Certificate/License - details of required certifications

# Citizenship - details of citizenship requirements

# Drivers_License  - details of drivers license requirements

# Education - details of education requirements

# Experience - details of prior experience requirements

# Health/Fitness/Vision/Drugs - details on medical physicals, drug testing, vision requirements, fitness requirements

# Job_Title - This is a pre-requisite job Title. If it is entry level, it will state that in the Availability Column. 

# Other - any other requirements of note that don't fit other categories
# Breaking it down

query_text = "SELECT distinct Job_1 as Age FROM job_graph where Category = 'Age' order by Job_1 asc"

ages_g = sqldf(query_text) ; ages_g

query_text = "SELECT distinct Job_1 as Background_Check FROM job_graph where Category = 'Background_Check' order by Job_1 asc"

bg_g = sqldf(query_text)  ;bg_g

query_text = "SELECT distinct Job_1 as Citizenship FROM job_graph where Category = 'Citizenship' order by Job_1 asc"

cz_g = sqldf(query_text)  ;cz_g

query_text = "SELECT distinct Job_1 as HealthFitnessVisionDrugs FROM job_graph where Category like '%Drugs%' order by Job_1 asc"

hl_g = sqldf(query_text)  ;hl_g

query_text = "SELECT distinct Job_1 as Other FROM job_graph where Category = 'Other' order by Job_1 asc"

ot_g = sqldf(query_text)  ;ot_g

query_text = "SELECT distinct Job_1 as Cert FROM job_graph where Category = 'Certificate/License' order by Job_1 asc"

ct_g = sqldf(query_text)  ;ct_g

query_text = "SELECT distinct Job_1 as DL FROM job_graph where Category like '%river%' order by Job_1 asc"

dl_g = sqldf(query_text)  ;dl_g
# What are the different types of availability?

# Availability : Open/Entry-Level  or Promotional based on the Job Trees provided by the City of LA

# Availability_Details: based on the text in the job bulletins
# Which category would you like to check?

category = "'Job_Title'"



# What is your current position? lower-case please

current_position = "'%officer%'"





### I'm going to start with just the Job_Title category

query_text = "SELECT * FROM job_graph where Category = %(category)s and (lower(Job_1) like %(current_position)s or lower(Job_2) like %(current_position)s)"%locals()

job_titles_g = sqldf(query_text)   #,(current_position,current_position))

job_titles_g.head(20)

dot = Digraph(comment='Promotions')



for index, row in job_titles_g.iterrows():

    dot.edge(str(row["Job_1"]), str(row["Job_2"]), label='')



dot
plt.figure(figsize=(50,50))

# Inspiration from  http://jonathansoma.com/lede/algorithms-2017/classes/networks/networkx-graphs-from-source-target-dataframe/

# 1. Create the graph

g = nx.from_pandas_edgelist(job_graph, source = 'Job_1', target = 'Job_2', edge_attr=None, create_using=None)



# 2. Create a layout for our nodes 

layout = nx.spring_layout(g,iterations=20)



# 3. Draw the parts we want

# Edges thin and grey

# Jobs_1 small and grey

# Promotions sized according to their number of connections

# Promotions blue

# Labels for Promotions ONLY

# Promotions that are highly connected are a highlighted color



# Go through every promotion ask the graph how many

# connections it has. Multiply that by 80 to get the circle size

promotion_size = [g.degree(Job_2) * 80 for Job_2 in promotions]

nx.draw_networkx_nodes(g, 

                       layout,

                       nodelist=promotions, 

                       node_size=promotion_size, # a LIST of sizes, based on g.degree

                       node_color='lightblue')



# Draw all jobs

nx.draw_networkx_nodes(g, layout, nodelist=jobs, node_color='#cccccc', node_size=100)



# Draw all jobs with most promotional ops

hot_jobs = [Job_1 for Job_1 in jobs if g.degree(Job_1) > 1]

nx.draw_networkx_nodes(g, layout, nodelist=hot_jobs, node_color='orange', node_size=100)



nx.draw_networkx_edges(g, layout, width=1, edge_color="#cccccc")



node_labels = dict(zip(promotions, promotions))

nx.draw_networkx_labels(g, layout, labels=node_labels)



plt.axis('off')

plt.title("Promotions")

plt.show()



# the larger the graph size (200,200) the longer it takes to run..... too small to read
g.nodes
job_graph.columns
# What are the different Availability Types?

query = """ SELECT distinct Availability from job_graph order by Availability asc"""

av = pysqldf(query)

av
count = """ SELECT count(distinct Job_1) as Possible_Promotions from job_graph where Availability like '%rom%' """

promotion_counts = pysqldf(count)

promotion_counts
count = """ SELECT count(distinct Job_2) as Open_to_Newcomers from job_graph where Availability like '%Entry%' """

open_counts = pysqldf(count)

open_counts
# Please answer the following questions



# Which category would you like to check?

category = "'Job_Title'"



# What is your current position?

current_position = "'%Officer%'"

### I'm going to start with just the Job_Title category

query_text = "SELECT * FROM job_graph where Category = %(category)s and (Job_1 like %(current_position)s or Job_2 like %(current_position)s)"%locals()

job_titles_g = sqldf(query_text)   #,(current_position,current_position))

job_titles_g.head(20)

from graphviz import Digraph

dot = Digraph(comment='Promotions')



for index, row in job_titles_g.iterrows():

    dot.edge(str(row["Job_1"]), str(row["Job_2"]), label='')



dot

# Which category would you like to check?

category = "'Job_Title'"



# What is your current position?

current_position = "'%Accountant%'"

# Run this cell for the results



#################################################################

query_text = "SELECT * FROM job_graph where Category = %(category)s and (Job_1 like %(current_position)s or Job_2 like %(current_position)s)"%locals()

job_titles_g = sqldf(query_text)   #,(current_position,current_position))

job_titles_g.head()

dot = Digraph(comment='Promotions')



for index, row in job_titles_g.iterrows():

    dot.edge(str(row["Job_1"]), str(row["Job_2"]), label='')

dot
# All Categories

# What is your current position?  all lower-case please

current_position = "'%officer%'"

# Run this cell for the results



#################################################################

query_text = "SELECT * FROM job_graph where  (lower(Job_1) like %(current_position)s or lower(Job_2) like %(current_position)s)"%locals()

job_titles_g = sqldf(query_text)   #,(current_position,current_position))

job_titles_g.head()

dot = Digraph(comment='Promotions')



for index, row in job_titles_g.iterrows():

    dot.edge(str(row["Job_1"]), str(row["Job_2"]), label='')

dot
# All Categories

# What is your current position or interest? all-lower case please

current_position = "'%fire%'"



# Run this cell for the results



#################################################################

query_text = "SELECT * FROM job_graph where  (lower(Job_1) like '%Fire%' or lower(Job_2) like '%Fire%')"

#query_text = "SELECT * FROM job_graph where  (lower(Job_1) like %(current_position)s or lower(Job_2) like %(current_position)s)"%locals()

job_titles_g = sqldf(query_text)   #,(current_position,current_position))

job_titles_g.head()

dot = Digraph(comment='Promotions')



for index, row in job_titles_g.iterrows():

    dot.edge(str(row["Job_1"]), str(row["Job_2"]), label='')

dot
job_graph.head(10)
# All Categories

# What is your current position or interest? all-lower case please

current_position = "'%nurse%'"



#################################################################

query_text = "SELECT * FROM job_graph where  (lower(Job_1) like %(current_position)s or lower(Job_2) like %(current_position)s)"%locals()

job_titles_g = sqldf(query_text)   #,(current_position,current_position))

job_titles_g.head()

dot = Digraph(comment='Promotions')



for index, row in job_titles_g.iterrows():

    dot.edge(str(row["Job_1"]), str(row["Job_2"]), label='')

dot
# All Categories

# What is your current position or interest? all-lower case please

availability = "'%entry%'"



#################################################################

query_text = "SELECT * FROM job_graph where lower(Availability like %(availability)s)"%locals()

job_titles_g = sqldf(query_text)   #,(current_position,current_position))

job_titles_g.head()

dot = Digraph(comment='Promotions')



for index, row in job_titles_g.iterrows():

    dot.edge(str(row["Job_1"]), str(row["Job_2"]), label='')

dot