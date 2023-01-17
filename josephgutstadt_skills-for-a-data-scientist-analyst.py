# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
jobs = pd.read_csv('/kaggle/input/data-jobs/all_jobs.csv')

skills = pd.read_csv('/kaggle/input/data-skills/skills_taxonomy.csv')
import re

import matplotlib.pyplot as plt

from mlxtend.frequent_patterns import apriori, association_rules

import copy
jobs
jobs.drop('Unnamed: 0', axis = 1, inplace = True)

jobs = jobs.drop_duplicates(subset = ['Job Description','Job Title','Location'], keep = 'first') 

#Let's remove Capitals

jobs['Job Description'] = jobs['Job Description'].str.lower()
#Let's remove all non-word charachters



regex = re.compile('[^a-zA-Z\']')



jobs['Job Description'] = jobs['Job Description'].apply(lambda x: regex.sub(' ', x))
jobs
#The Equal Oppritunity tagline may skew our results, let's remove it

equal_emp = 'Kelly is an equal opportunity employer committed to employing a diverse workforce, including, but not limited to, minorities, females, individuals with disabilities, protected veterans, sexual orientation, gender identity. Equal Employment Opportunity is The Law.'

equal_emp = equal_emp.lower().split(' ')



jobs['Job Description'] = jobs['Job Description'].apply(lambda x: [item for item in x.split() if item.lower() not in equal_emp])



#and then re-join our Job Descriptions

jobs['Job Description'] = jobs['Job Description'].apply(lambda x: ' '.join(x))

skill_types= {}



skill_types['Statistics'] = ['matlab',

 'statistical',

 'models',

 'modeling',

 'statistics',

 'analytics',

 'forecasting',

 'predictive',

 'r',

 'pandas',

 'statistics',

 'statistical',

 'Julia']



skill_types['Machine Learning'] = ['datarobot',

 'tensorflow',

 'knime',

 'rapidminer',

 'mahout',

 'logicalglue',

 'nltk',

 'networkx',

 'rapidminer',

 'scikit',

 'pytorch',

 'keras',

 'caffe',

 'weka',

 'orange',

 'qubole',

 'ai',

 'nlp',

 'ml',

 'neuralnetworks',

 'deeplearning']





skill_types['Data Visualization'] = ['tableau',

 'powerpoint',

 'Qlik',

 'looker',

 'powerbi',

 'matplotlib',

 'tibco',

 'bokeh',

 'd3',

 'octave',

 'shiny',

 'microstrategy']





skill_types['Data Engineering'] = ['etl',

 'mining',

 'warehousing',

 'cloud',

 'sap',

 'salesforce',

 'openrefine',

 'redis',

 'sybase',

 'cassandra',

 'msaccess',

 'databasemanagement',

 'aws',

 'ibmcloud',

 'azure',

 'redshift',

 's3',

 'ec2',

 'rds',

 'bigquery',

 'googlecloudplatform',

 'googlecloudplatform',

 'hadoop',

 'hive',

 'kafka',

 'hbase',

 'mesos',

 'pig',

 'storm',

 'scala',

 'hdfs',

 'mapreduce',

 'kinesis',

 'flink']





skill_types['Software Engineer'] = ['java',

 'javascript',

 'c#',

 'c',

 'docker',

 'ansible',

 'jenkins',

 'nodejs',

 'angularjs',

 'css',

 'html',

 'terraform',

 'kubernetes',

 'lex',

 'perl',

 'cplusplus']





skill_types['SQL'] = ['sql',

 'oracle',

 'mysql',

 'oraclenosql',

 'nosql',

 'postgresql',

 'plsql',

 'mongodb']









skill_types['Trait Skills'] = ['Learning',

 'TimeManagement',

 'AttentiontoDetail',

 'ProblemSolving',

 'criticalthinking']







skill_types['Social Skills']= ['teamwork',

 'team'

 'communication',

 'written',

 'verbal',

 'writing',

 'leadership',

 'interpersonal',

 'personalmotivation',

 'storytelling']



skill_types['Business'] = ['excel',

 'bi',

 'reporting',

 'reports',

 'dashboards',

 'dashboard',

 'businessintelligence'

 'business']
for k,v in skill_types.items():

    skill_types[k] = [skill.lower() for skill in skill_types.get(k)]
def refiner(desc):

    desc = desc.split()

    

    two_word = ''

    

    newskills = []

    

    for word in desc:

        two_word = two_word + word 

        for key,value in skill_types.items():

            if((word in value) or (two_word in value)):

                newskills.append(key)

                

        #check for the two worders, like 'businessintelligence'        

        two_word = word

                

    return list(set(newskills))
#Now all we have to do is apply this do our Job Description

jobs['refined skills'] = jobs['Job Description'].apply(refiner)

jobs['refined skills']
def apriori_df(series, min_support):

    lisolis =[]

    series.apply(lambda x: lisolis.append(list(x)))

    

    from mlxtend.preprocessing import TransactionEncoder



    te = TransactionEncoder()

    te_ary = te.fit(lisolis).transform(lisolis)

    df = pd.DataFrame(te_ary, columns=te.columns_)





    from mlxtend.frequent_patterns import apriori



    freq_itemsets = apriori(df, min_support=min_support, use_colnames=True)

    

    return freq_itemsets
frequent_itemsets = apriori_df(jobs['refined skills'],.1)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

import seaborn as sns



_ = frequent_itemsets[frequent_itemsets['length'] == 1]

_['itemsets'] = _['itemsets'].astype("unicode").str.replace('[\(\)\'\{\}]|frozenset','', regex = True)

ax = sns.barplot(x="itemsets", y="support", data= _);

ax.set_xticklabels(ax.get_xticklabels(), rotation=75);

temp = pd.DataFrame()
jobtypes = ['Data Analyst','Business Analyst','Data Scientist']
#If it contains Data and Analyst we will classify that as Data Analyst

data_analyst = jobs[(jobs['Job Title'].str.contains('[Aa]nalyst', flags=re.IGNORECASE)) & (jobs['Job Title'].str.contains('[dD]ata ', flags=re.IGNORECASE))]

data_analyst['jobtype'] = jobtypes[0]

temp = temp.append(data_analyst)
business_analyst = jobs[(jobs['Job Title'].str.contains('[Aa]nalyst|[Ii]ntelligence|BI', regex = True, flags=re.IGNORECASE)) & (jobs['Job Title'].str.contains('Business |BI', flags=re.IGNORECASE))]

business_analyst['jobtype'] = jobtypes[1]

temp = temp.append(business_analyst)
data_scientist = jobs[(jobs['Job Title'].str.contains('[Ss]cientist|Science', regex = True, flags=re.IGNORECASE)) & (jobs['Job Title'].str.contains('[dD]ata ', flags=re.IGNORECASE))]

data_scientist['jobtype'] = jobtypes[2]

temp = temp.append(data_scientist)
jobs = temp
jobs.reset_index(inplace = True)

jobs.drop(['index'], axis =1, inplace = True)
# We want to retain the categories we just made, run apriori, and then recombine them

frequent_itemsets = pd.DataFrame()





for job in jobtypes:

    temp_frequent_itemsets = apriori_df(jobs.loc[jobs['jobtype'] == job,'refined skills'],.02)

    temp_frequent_itemsets['jobtype'] = job

    frequent_itemsets = frequent_itemsets.append(temp_frequent_itemsets)







frequent_itemsets
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
single_items = frequent_itemsets[frequent_itemsets['length'] == 1]

single_items['itemsets'] = single_items['itemsets'].astype("unicode").str.replace('[\(\)\'\{\}]|frozenset','', regex = True)

single_items['itemsets'] = single_items['itemsets'].str.replace(' ','\n', regex = True)

plt.figure(figsize =(25, 20)) 



plt.title('Chance of a skill appearing in a given Job Description', size = 35)

ax = sns.barplot(x="itemsets", y="support", hue = 'jobtype', data= single_items);

ax.set_xticklabels(ax.get_xticklabels(), rotation=25, size =20);

ax.set_yticklabels([0,0.2,0.4,0.6,0.8], size =25);

plt.xlabel('Skill', size = 20)

ax.legend(fontsize = 20, frameon= False)

plt.ylabel('Probability', size = 22)

plt.savefig('chart1.png', dpi = 500, format = 'png')
import networkx as nx 
typ = 'Data Analyst'
rules = association_rules(apriori_df(jobs.loc[jobs['jobtype'] == typ,'refined skills'],.2), metric="conviction", min_threshold=1.4)
# 1 to 1 

rules['alength'] = rules['antecedents'].apply(lambda x: len(x))

rules['clength'] = rules['consequents'].apply(lambda x: len(x))

rules = rules[(rules['alength'] == 1) & (rules['clength'] == 1)]

rules['antecedents'] = rules['antecedents'].astype("unicode").str.replace('[\(\)\'\{\}]|frozenset','', regex = True)

rules['antecedents'] = rules['antecedents'].str.replace(' ','\n', regex = True)

rules['consequents'] = rules['consequents'].astype("unicode").str.replace('[\(\)\'\{\}]|frozenset','', regex = True)

rules['consequents'] = rules['consequents'].str.replace(' ','\n', regex = True)

rules
rules.sort_values(by = 'conviction')
#Make some edges, now that we've run apriori

weighted_edges = []



for x in range(len(rules)):

    weighted_edges.append((rules.iloc[x,0], rules.iloc[x,1], rules.iloc[x,8]))
G = nx.DiGraph() 

G.add_weighted_edges_from(weighted_edges)
#Change node size according to support

for i in list(G.nodes()): 

    G.nodes[i]['support'] = single_items.loc[(single_items['jobtype'] == typ) & (single_items['itemsets'] == i), 'support'].values

node_size = [50000*nx.get_node_attributes(G, 'support')[v] for v in G] 

edge_width = [(G[u][v]['weight']- 1)*12 if((G[u][v]['weight']- 1) > .4) else 0 for u, v in G.edges() ]

pos = nx.circular_layout(G)

plt.figure(figsize=(20,20))



pos_shadow = copy.deepcopy(pos)

shift_amount = 0.008

for idx in pos_shadow:

    pos_shadow[idx][0] += shift_amount

    pos_shadow[idx][1] -= shift_amount



nx.draw_networkx_nodes(G, pos_shadow, node_color='k', alpha=0.3,  node_size = node_size)    

    

nx.draw_networkx_nodes(G, pos, with_label = True, node_size = node_size,connectionstyle='arc3, rad = .03',arrowsize=60, width = edge_width)



nx.draw_networkx_labels(G, pos, with_label = True, node_size = node_size,connectionstyle='arc3, rad = .03',arrowsize=60, size = 25, width = edge_width)



nx.draw_networkx_edges(G, pos, with_label = True, node_size = node_size,connectionstyle='arc3, rad = .03',arrowsize=60, width = edge_width)

plt.title('Data Analyst Skills', size = 50)

plt.axis('off')

plt.plot();
def make_network_graph(jobtype, min_conviction):

    rules = association_rules(apriori_df(jobs.loc[jobs['jobtype'] == jobtype,'refined skills'],.15), metric="conviction", min_threshold=.5)

    # 1 to 1 

    rules['alength'] = rules['antecedents'].apply(lambda x: len(x))

    rules['clength'] = rules['consequents'].apply(lambda x: len(x))

    rules = rules[(rules['alength'] == 1) & (rules['clength'] == 1)]

    rules['antecedents'] = rules['antecedents'].astype("unicode").str.replace('[\(\)\'\{\}]|frozenset','', regex = True)

    rules['antecedents'] = rules['antecedents'].str.replace(' ','\n', regex = True)

    rules['consequents'] = rules['consequents'].astype("unicode").str.replace('[\(\)\'\{\}]|frozenset','', regex = True)

    rules['consequents'] = rules['consequents'].str.replace(' ','\n', regex = True)





    #Make some edges, now that we've run apriori

    weighted_edges = []



    G = nx.DiGraph()

    

    for x in range(len(rules)):

        if(rules.iloc[x,8] > min_conviction):

            weighted_edges.append((rules.iloc[x,0], rules.iloc[x,1], rules.iloc[x,8]))

        

        else:

            G.add_node(rules.iloc[x,1])

            G.add_node(rules.iloc[x,0])

        

        

     

    G.add_weighted_edges_from(weighted_edges)

    

    #Change node size according to support

    for i in list(G.nodes()): 

        G.nodes[i]['support'] = single_items.loc[(single_items['jobtype'] == jobtype) & (single_items['itemsets'] == i), 'support'].values

        

    node_size = [60000*nx.get_node_attributes(G, 'support')[v] for v in G] 

    edge_width = [(G[u][v]['weight']- 1)*10.5 if((G[u][v]['weight']- 1) > .1) else 0 for u, v in G.edges() ]



    pos = nx.circular_layout(G)

    plt.figure(figsize=(27,22))





    pos_shadow = copy.deepcopy(pos)

    shift_amount = 0.008

    for idx in pos_shadow:

        pos_shadow[idx][0] += shift_amount

        pos_shadow[idx][1] -= shift_amount



    nx.draw_networkx_nodes(G, pos_shadow, node_color='k', alpha=0.3,  node_size = node_size)   



    nx.draw_networkx_nodes(G, pos, with_label = True, node_size = node_size,connectionstyle='arc3, rad = .03',arrowsize=60, width = edge_width)



    nx.draw_networkx_labels(G, pos, with_label = True, node_size = node_size,connectionstyle='arc3, rad = .03',arrowsize=60, size = 20, font_size = 28, width = edge_width, font_weight = 'bold', font_color = 'darkorange')



    nx.draw_networkx_edges(G, pos, with_label = True, node_size = node_size,connectionstyle='arc3, rad = .03',arrowsize=60, width = edge_width)

    plt.title(jobtype + ' Skills', size = 50)

    plt.axis('off')

    plt.plot();
make_network_graph(jobtypes[1],1.4)
make_network_graph(jobtypes[0],1.45)

plt.savefig('DAgraph.svg', format='svg', dpi=1200)
make_network_graph(jobtypes[2],1.6)

plt.savefig('DSgraph.svg', format='svg', dpi=1200)