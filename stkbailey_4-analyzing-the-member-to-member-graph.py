import pandas as pd
import networkx as nx

edges = pd.read_csv('../input/member-edges.csv')
g = nx.from_pandas_edgelist(edges, 'member1', 'member2', 'weight')

print('There are', len(g.nodes), 'nodes and', len(g.edges), 'edges.')
members = pd.read_csv('../input/meta-members.csv', index_col='member_id')
members.head()
# Compute measures for each member... WARNING: takes a long time (>15 min)!
members['degree'] = pd.Series(dict(nx.degree(g)))
members['clustering'] = pd.Series(nx.clustering(g))
members['centrality'] = pd.Series(nx.betweenness_centrality(g, k=500)) # Use a subset of size
# Create a metadata-rich member-to-group dataframe
mem2gp = ( pd.read_csv('../input/rsvps.csv')
              .groupby(['member_id', 'group_id']).size()
              .reset_index().rename(columns={0: 'weight'}) )
groups = pd.read_csv('../input/meta-groups.csv', index_col='group_id')
mem2gp = ( mem2gp.merge(members[['name']], left_on='member_id', right_index=True)
              .merge(groups[['group_name', 'category_name']], 
                     left_on='group_id', right_index=True) )

# Print out the group affiliations for the top five most central people
print('The top five most central people in Nashville (+ their top meetups) are...')
top_five = members.sort_values(by='centrality', ascending=False).index[0:5]
for mid in top_five:
    print(members.loc[mid, 'name'], '({})'.format(mid))
    print('\t' + '\n\t'.join(mem2gp.loc[mem2gp.member_id == mid]
         .sort_values(by='weight', ascending=False)
         .group_name[0:5].tolist() ))
import seaborn as sns
import matplotlib.pyplot as plt

grid = sns.pairplot(members[['degree', 'clustering', 'centrality']].dropna())

plt.show()