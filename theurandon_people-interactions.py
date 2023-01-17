%matplotlib inline
import pandas as pd
import networkx as nx
from collections import Counter
import mpld3
from mpld3 import plugins
%pylab inline
df_aliases = pd.read_csv('../input/Aliases.csv', index_col=0)
df_emails = pd.read_csv('../input/Emails.csv', index_col=0)
df_email_receivers = pd.read_csv('../input/EmailReceivers.csv', index_col=0)
df_persons = pd.read_csv('../input/Persons.csv', index_col=0)
top = df_email_receivers.PersonId.value_counts().head(n=10).to_frame()
top.columns = ['Emails received']
top = pd.concat([top, df_persons.loc[top.index]], axis=1)
top.plot(x='Name', kind='barh', figsize=(12, 8), grid=True, color='r')
show()
import re

aliases = {df_aliases.Alias[idx]: df_aliases.PersonId[idx]
           for idx in df_aliases.index}
aliases['nan'] = 'nan'
# little fix for Aliases.csv
aliases['rosemarie.howe'] = aliases['rosemarie howe']
punkt = re.compile("[',\-]")

def tokenize(s):
    rec_list = punkt.sub('', s).split(';')
    return filter(lambda x: x != '', [x.lower().strip() for x in rec_list])

def id_ze(s):
    return [aliases[x] for x in tokenize(s)]
df_rel = df_emails[['MetadataFrom', 'MetadataTo']].dropna()
from_ = df_rel.MetadataFrom.apply(id_ze)
to_ = df_rel.MetadataTo.apply(id_ze)
def cross(iter1, iter2):
    for x in iter1:
        for y in iter2:
            if x < y:
                yield x, y
            elif x > y:
                yield y, x
emails = Counter()
emails_unary = Counter()
for (src, dest) in zip(from_, to_):
    for x, y in cross(src, dest):
        emails[x, y] += 1
        emails_unary[x] += 1
        emails_unary[y] += 1
g_sent = nx.Graph()
g_sent.add_weighted_edges_from(((u, v, log(n_emails+1))
                                for ((u, v), n_emails)
                                in emails.items()))
components = list(nx.connected_components(nx.Graph(g_sent)))
components.sort(key=lambda c: len(c), reverse=True)
figure(figsize=(12, 8))
connecteds_len = [len(comp) for comp in components]
bar(range(1, len(connecteds_len) + 1), connecteds_len)
ylabel('Connected component size')
show()
for idx, comp in enumerate(components[1:]):
    if len(comp) < 2: continue
    print('Component #{}'.format(idx + 2))
    print('\n'.join(['\t' + df_persons.Name.loc[al_id] for al_id in comp]))
g = g_sent.subgraph(components[0])

cf = plt.gcf()
cf.set_size_inches((12, 12))
ax = cf.add_axes((0, 0, 1, 1))

pos = nx.spring_layout(g)

nodelist = g.nodes()
centrality = nx.current_flow_betweenness_centrality(g)
labels = [df_persons.Name.loc[node_id] for node_id in nodelist]
node_size = [200 + 800 * centrality[node_id] for node_id in nodelist]

n_emails = array([log(emails_unary[node_id]) for node_id in nodelist])
n_emails /= max(n_emails)

weights = array([g.edge[u][v]['weight'] for (u, v) in g.edges()])
weights = list(1.0 + 4.0 * weights / max(weights))

node_collection = nx.draw_networkx_nodes(
    g, pos, alpha=0.5, node_size=node_size,
    cmap='plasma', node_color=n_emails)
edge_collection = nx.draw_networkx_edges(
    g, pos, alpha=0.3, arrows=False, width=weights)

ax.set_axis_off()

# Add interactive tooltips
tooltip = plugins.PointLabelTooltip(node_collection, labels)
plugins.connect(cf, tooltip)

mpld3.display()