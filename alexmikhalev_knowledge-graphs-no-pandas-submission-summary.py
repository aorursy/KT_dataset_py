from utils_for_automata import *

!pip install d3fdgraph

import d3fdgraph

import pandas as pd

import dask.dataframe as dd 



edges_df=dd.read_csv("/kaggle/input/samplecord19knowledgegraph/edges.csv/edges.csv",names=["source", "destination", "skey"])

nodes_df=pd.read_csv("/kaggle/input/samplecord19knowledgegraph/nodes.csv", names=["canonical_id", "name"])
Task10="What has been published about information sharing and inter-sectoral collaboration?"

Task10detail="""Task Details What has been published about information sharing and inter-sectoral collaboration? What has been published about data standards and nomenclature? What has been published about governmental public health? What do we know about risk communication? What has been published about communicating with high-risk populations? What has been published to clarify community measures? What has been published about equity considerations and problems of inequity?"""
task10_search_terms=find_matches(Task10)

print("Search tearms for the task mapped to UMLS entities: ", task10_search_terms)
search_task=edges_df[edges_df.source.isin(['C4277663','C0016572', 'C0016572', 'C0279756'])] 

search_df=search_task.compute().head(10)
results_list=[]

articles_table=[]

for idx, row in search_df.iterrows():

    articles_table.append(row['skey'])

    sname=nodes_df[nodes_df.canonical_id==row['source']]['name'].to_string(header=False,

                  index=False)

    dname=nodes_df[nodes_df.canonical_id==row['destination']].name.to_string(header=False,

                  index=False)

    results_list.append({'source':sname,'destionation':dname,'weight':3})



results_df=pd.DataFrame(results_list,columns=['source','destionation','weight'])

d3fdgraph.plot_force_directed_graph(results_df,node_radius=15, link_distance=20, collision_scale=4) 
articles_table
Task9="What has been published about ethical and social science considerations?"

Task9detail="""

What has been published concerning ethical considerations for research? What has been published concerning social sciences at the outbreak response?

"""
task9_search_terms=find_matches(Task9)

print("Search tearms for the task mapped to UMLS entities: ", task9_search_terms)
search_list=[]

for each_term in task9_search_terms:

    search_list.append(each_term[0])
search_task=edges_df[edges_df.source.isin(search_list)] 

search_df=search_task.compute().head(10)
results_list=[]

articles_table=[]

for idx, row in search_df.iterrows():

    articles_table.append(row['skey'])

    sname=nodes_df[nodes_df.canonical_id==row['source']]['name'].to_string(header=False,

                  index=False)

    dname=nodes_df[nodes_df.canonical_id==row['destination']].name.to_string(header=False,

                  index=False)

    results_list.append({'source':sname,'destionation':dname,'weight':3})



results_df=pd.DataFrame(results_list,columns=['source','destionation','weight'])

d3fdgraph.plot_force_directed_graph(results_df,node_radius=15, link_distance=20, collision_scale=4) 
articles_table
Task8="What do we know about diagnostics and surveillance?"

Task8detail="What do we know about diagnostics and surveillance? What has been published concerning systematic, holistic approach to diagnostics (from the public health surveillance perspective to being able to predict clinical outcomes)?"
task8_search_terms=find_matches(Task8)

print("Search tearms for the task mapped to UMLS entities: ", task8_search_terms)
search_list=[]

for each_term in task8_search_terms:

    search_list.append(each_term[0])
search_task=edges_df[edges_df.source.isin(search_list)] 

search_df=search_task.compute().head(10)
results_list=[]

articles_table=[]

for idx, row in search_df.iterrows():

    articles_table.append(row['skey'])

    sname=nodes_df[nodes_df.canonical_id==row['source']]['name'].to_string(header=False,

                  index=False)

    dname=nodes_df[nodes_df.canonical_id==row['destination']].name.to_string(header=False,

                  index=False)

    results_list.append({'source':sname,'destionation':dname,'weight':3})



results_df=pd.DataFrame(results_list,columns=['source','destionation','weight'])

d3fdgraph.plot_force_directed_graph(results_df,node_radius=15, link_distance=20, collision_scale=4) 
articles_table
Task7="What do we know about non-pharmaceutical interventions?"

Task7Details="What do we know about the effectiveness of non-pharmaceutical interventions? What is known about equity and barriers to compliance for non-pharmaceutical interventions?"
task7_search_terms=find_matches(Task7)

print("Search tearms for the task mapped to UMLS entities: ", task7_search_terms)
search_list=[]

for each_term in task7_search_terms:

    search_list.append(each_term[0])
search_task=edges_df[edges_df.source.isin(search_list)] 

search_df=search_task.compute().head(10)
results_list=[]

articles_table=[]

for idx, row in search_df.iterrows():

    articles_table.append(row['skey'])

    sname=nodes_df[nodes_df.canonical_id==row['source']]['name'].to_string(header=False,

                  index=False)

    dname=nodes_df[nodes_df.canonical_id==row['destination']].name.to_string(header=False,

                  index=False)

    results_list.append({'source':sname,'destionation':dname,'weight':3})



results_df=pd.DataFrame(results_list,columns=['source','destionation','weight'])

d3fdgraph.plot_force_directed_graph(results_df,node_radius=15, link_distance=20, collision_scale=4) 
articles_table
Task6="What has been published about medical care?"
task6_search_terms=find_matches(Task6)

print("Search tearms for the task mapped to UMLS entities: ", task6_search_terms)
search_list=[]

for each_term in task6_search_terms:

    search_list.append(each_term[0])



search_task=edges_df[edges_df.source.isin(search_list)] 

search_df=search_task.compute().head(10)

results_list=[]

articles_table=[]

for idx, row in search_df.iterrows():

    articles_table.append(row['skey'])

    sname=nodes_df[nodes_df.canonical_id==row['source']]['name'].to_string(header=False,

                  index=False)

    dname=nodes_df[nodes_df.canonical_id==row['destination']].name.to_string(header=False,

                  index=False)

    results_list.append({'source':sname,'destionation':dname,'weight':3})



results_df=pd.DataFrame(results_list,columns=['source','destionation','weight'])

d3fdgraph.plot_force_directed_graph(results_df,node_radius=15, link_distance=20, collision_scale=4) 

articles_table
Task5="What do we know about vaccines and therapeutics?"
task5_search_terms=find_matches(Task5)

print("Search tearms for the task mapped to UMLS entities: ", task5_search_terms)
search_list=[]

for each_term in task5_search_terms:

    search_list.append(each_term[0])



search_task=edges_df[edges_df.source.isin(search_list)] 

search_df=search_task.compute().head(10)

results_list=[]

articles_table=[]

for idx, row in search_df.iterrows():

    articles_table.append(row['skey'])

    sname=nodes_df[nodes_df.canonical_id==row['source']]['name'].to_string(header=False,

                  index=False)

    dname=nodes_df[nodes_df.canonical_id==row['destination']].name.to_string(header=False,

                  index=False)

    results_list.append({'source':sname,'destionation':dname,'weight':3})



results_df=pd.DataFrame(results_list,columns=['source','destionation','weight'])

d3fdgraph.plot_force_directed_graph(results_df,node_radius=15, link_distance=20, collision_scale=4) 
articles_table
Task4="What do we know about virus genetics, origin, and evolution?"
task4_search_terms=find_matches(Task4)

print("Search tearms for the task mapped to UMLS entities: ", task4_search_terms)
search_list=[]

for each_term in task4_search_terms:

    search_list.append(each_term[0])



search_task=edges_df[edges_df.source.isin(search_list)] 

search_df=search_task.compute().head(10)

results_list=[]

articles_table=[]

for idx, row in search_df.iterrows():

    articles_table.append(row['skey'])

    sname=nodes_df[nodes_df.canonical_id==row['source']]['name'].to_string(header=False,

                  index=False)

    dname=nodes_df[nodes_df.canonical_id==row['destination']].name.to_string(header=False,

                  index=False)

    results_list.append({'source':sname,'destionation':dname,'weight':3})



results_df=pd.DataFrame(results_list,columns=['source','destionation','weight'])

d3fdgraph.plot_force_directed_graph(results_df,node_radius=15, link_distance=20, collision_scale=4) 
articles_table
Task3="What do we know about COVID-19 risk factors?"

task3_search_terms=find_matches(Task3)

print("Search tearms for the task mapped to UMLS entities: ", task3_search_terms)
search_list=[]

for each_term in task3_search_terms:

    search_list.append(each_term[0])



search_task=edges_df[edges_df.source.isin(search_list)] 

search_df=search_task.compute().head(10)

results_list=[]

articles_table=[]

for idx, row in search_df.iterrows():

    articles_table.append(row['skey'])

    sname=nodes_df[nodes_df.canonical_id==row['source']]['name'].to_string(header=False,

                  index=False)

    dname=nodes_df[nodes_df.canonical_id==row['destination']].name.to_string(header=False,

                  index=False)

    results_list.append({'source':sname,'destionation':dname,'weight':3})



results_df=pd.DataFrame(results_list,columns=['source','destionation','weight'])

d3fdgraph.plot_force_directed_graph(results_df,node_radius=15, link_distance=20, collision_scale=4) 
articles_table
Task2="What is known about transmission, incubation, and environmental stability?"
task2_search_terms=find_matches(Task2)

print("Search tearms for the task mapped to UMLS entities: ", task2_search_terms)
search_list=[]

for each_term in task2_search_terms:

    search_list.append(each_term[0])



search_task=edges_df[edges_df.source.isin(search_list)] 

search_df=search_task.compute().head(10)

results_list=[]

articles_table=[]

for idx, row in search_df.iterrows():

    articles_table.append(row['skey'])

    sname=nodes_df[nodes_df.canonical_id==row['source']]['name'].to_string(header=False,

                  index=False)

    dname=nodes_df[nodes_df.canonical_id==row['destination']].name.to_string(header=False,

                  index=False)

    results_list.append({'source':sname,'destionation':dname,'weight':3})



results_df=pd.DataFrame(results_list,columns=['source','destionation','weight'])

d3fdgraph.plot_force_directed_graph(results_df,node_radius=15, link_distance=20, collision_scale=4) 
articles_table