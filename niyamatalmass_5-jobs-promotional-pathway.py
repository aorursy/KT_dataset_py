!echo Y | apt-get install graphviz libgraphviz-dev pkg-config

!pip install pygraphviz

!pip install pyvis
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import networkx as nx

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



%matplotlib inline
df_jobs_struct = pd.read_csv('../input/2-raw-job-bulletins-to-structured-csv/jobs.csv')

df_jobs_struct[['JOB_CLASS_TITLE', 'JOB_CLASS_NO', 'EXP_JOB_CLASS_TITLE']].head(20)
def get_upper_positions(current_job_title):

    return df_jobs_struct.loc[

        df_jobs_struct['EXP_JOB_CLASS_TITLE'].str.upper()\

        == current_job_title.upper()]



def build_graph(job_title,list_relations=None):

    # job title is the lower position

    # we will find upper position of this job_title

    

    # first get all the position this job_title as a experience

    

    

    upper_positions = get_upper_positions(job_title)

    

    if len(upper_positions) == 0:

        return

    else:

        if list_relations is None:

            list_relations = list(

                zip(

                    upper_positions.JOB_CLASS_TITLE.str.upper(),

                    upper_positions.EXP_JOB_CLASS_TITLE.str.upper()))



            if len(upper_positions) >= 1:

                for index, row in upper_positions.iterrows():

                    build_graph(row['JOB_CLASS_TITLE'], list_relations)

                    break

            else:

                return

        else:

            temp_list_relations = list(

                zip(

                    upper_positions.JOB_CLASS_TITLE.str.upper(),

                    upper_positions.EXP_JOB_CLASS_TITLE.str.upper()))

            for t in temp_list_relations:

                list_relations.append(t)

                

            if len(upper_positions) >= 1:

                for index, row in upper_positions.iterrows():

                    build_graph(row['JOB_CLASS_TITLE'])

                    break

            else:

                return

                

    return list_relations
network_tuple = build_graph('MANAGEMENT ANALYST')

network_tuple
%config InlineBackend.figure_format = 'retina'

from networkx.drawing.nx_agraph import write_dot, graphviz_layout



network_tuple = build_graph('MANAGEMENT ANALYST')

# build a networkx graph

G = nx.DiGraph() 

# add main node our given job title

G.add_node('MANAGEMENT ANALYST') 

# add edges of our graph 

if network_tuple is not None:

    G.add_edges_from(network_tuple)

    

    

# now plot that graph using networkx build in function

plt.figure(figsize=(20, 20))

pos =graphviz_layout(G, prog='dot')

nx.draw(G.reverse(), pos,with_labels=False, arrows=True, arrowstyle='simple', arrowsize=18)

text = nx.draw_networkx_labels(G,pos)

for _,t in text.items():

    t.set_rotation(45)

    t.set_fontsize(22)

plt.show()
#########################################

# function for ploting promotions graph very easily

#########################################



def plot_promotion_graph(job_title):

    network_tuple = build_graph(job_title)

    # build a networkx graph

    G = nx.DiGraph() 

    # add main node our given job title

    G.add_node(job_title) 

    # add edges of our graph 

    if network_tuple is not None:

        G.add_edges_from(network_tuple)





    # now plot that graph using networkx build in function

    plt.figure(figsize=(20, 20))

    pos =graphviz_layout(G, prog='dot')

    nx.draw(G.reverse(), pos,with_labels=False, arrows=True, arrowstyle='simple', arrowsize=18)

    text = nx.draw_networkx_labels(G,pos)

    for _,t in text.items():

        t.set_rotation(45)

        t.set_fontsize(20)

    plt.show()
plot_promotion_graph('SENIOR MANAGEMENT ANALYST')
plot_promotion_graph("DETENTION OFFICER")
plot_promotion_graph("ANIMAL KEEPER")
def show_promotion_details(job_title):

    temp = df_jobs_struct.loc[

        df_jobs_struct["JOB_CLASS_TITLE"] == job_title.upper()]

    return temp[['EXP_JOB_CLASS_TITLE', 'EXP_JOB_CLASS_FUNCTION',

                 'EXPERIENCE_LENGTH', 'EDUCATION_MAJOR', 'EXP_JOB_COMPANY']]
show_promotion_details('SENIOR MANAGEMENT ANALYST')
from pyvis.network import Network
def plot_interactive_graph(job_title):

    network_tuple = build_graph(job_title)

    # build a networkx graph

    G = nx.DiGraph() 

    # add main node our given job title

    G.add_node(job_title) 

    # add edges of our graph 

    if network_tuple is not None:

        G.add_edges_from(network_tuple)



    nt = Network(height="800px",

                     width="750px",

                     directed=True,

                     notebook=True,

                     bgcolor="#ffffff",

                     font_color=False,

                     layout=True)

    nt.from_nx(G)



    neighbor_map = nt.get_adj_list()

    # add neighbor data to node hover data

    for node in nt.nodes:

        node["title"] = "Experience Job Title: "+ df_jobs_struct.loc[

            df_jobs_struct['JOB_CLASS_TITLE'].str.upper() == node['title'], 'EXP_JOB_CLASS_TITLE'].str.upper(

        ).str.cat(sep=', ') + "; Experience Years: " + df_jobs_struct.loc[

            df_jobs_struct['JOB_CLASS_TITLE'].str.upper() == node['title'], 'EXPERIENCE_LENGTH'].str.upper(

        ).str.cat(sep=', ')

        node["value"] = len(neighbor_map[node["id"]])

    return nt

    
network = plot_interactive_graph('MANAGEMENT ANALYST')

network.show("mygraph.html")
network = plot_interactive_graph("ANIMAL KEEPER")

network.show("mygraph2.html")