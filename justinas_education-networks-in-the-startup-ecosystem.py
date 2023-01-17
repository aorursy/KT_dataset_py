import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns



import holoviews as hv

from holoviews import opts

from holoviews.operation.datashader import datashade, bundle_graph



import networkx as nx
hv.extension('bokeh')



defaults = dict(width=600, height=600, padding=0.1, yaxis=None, xaxis=None, show_frame=False)

hv.opts.defaults(

    opts.EdgePaths(**defaults), opts.Graph(**defaults), opts.Nodes(**defaults))
people = pd.read_csv('../input/startup-investments/people.csv', index_col=0)

degrees = pd.read_csv('../input/startup-investments/degrees.csv', index_col=0)



df = people.merge(degrees, on='object_id')
df.info()
df['full_name'] = df['first_name'].str.cat(df['last_name'],sep=" ")



df['institution'] = df['institution'].replace('Harvard Business School' ,'Harvard University')

df['institution'] = df['institution'].replace('Stanford University Graduate School of Business' ,'Stanford University')
df = df[df['affiliation_name'] != 'Unaffiliated']
df = df[['object_id', 'full_name', 'birthplace', 'institution', 'degree_type', 'subject', 'graduated_at', 'affiliation_name']]
def count_plots(df, col_count):

    for i, col in enumerate(df.columns):

        plt.figure(i, figsize=(10,5))

        sns.countplot(x=col, data=df, order=pd.value_counts(df[col]).iloc[:col_count].index)

        plt.xticks(rotation=70)

        

count_columns = df[['institution', 'degree_type', 'subject', 'affiliation_name']]



count_plots(count_columns, 10)
def dual_degree_flag_generator(df):

    group = df.groupby(['object_id', 'institution', 'graduated_at'], as_index=False)['full_name'].count()

    group = group[group['full_name'] > 1]

    object_ids = group['object_id']

    

    df['dual_degree_flag'] = np.where(df['object_id'].isin(object_ids), 1, 0)

    

    return df



df = dual_degree_flag_generator(df)
institution_occurance_count = df['institution'].value_counts()

important_institutions = institution_occurance_count[institution_occurance_count >= 5].index.values

df = df[df['institution'].isin(important_institutions)]
df = df.dropna()
df = df[:5000]
# Create the graph object

G = nx.Graph()
G = nx.from_pandas_edgelist(df, source='full_name', target='institution', 

                            edge_attr=['degree_type', 'subject'])
nx.set_node_attributes(G, pd.Series(df['affiliation_name'].values, index=df['full_name']).to_dict(), 'company')

nx.set_node_attributes(G, pd.Series(np.nan, index=df['institution']).to_dict(), 'company')
list(G.edges(data=True))[:5]
list(G.nodes(data=True))[:5]
print(nx.info(G))
components = nx.connected_components(G)

largest_component = max(components, key=len)

subgraph = G.subgraph(largest_component)

diameter = nx.diameter(subgraph)

print("Network diameter of largest component:", diameter)
triadic_closure = nx.transitivity(G)

print("Triadic closure:", triadic_closure)
simple_graph = hv.Graph.from_networkx(G, positions=nx.spring_layout(G))

simple_graph.opts(title="Education Network", node_color='company', cmap='set3', edge_color='degree_type', edge_cmap='set3')