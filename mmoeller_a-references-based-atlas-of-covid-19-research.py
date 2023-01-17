import pandas as pd



df = pd.read_csv('../input/CORD-19-research-challenge/metadata.csv')

df.dropna(subset=['pubmed_id'], inplace=True)

df.drop_duplicates(subset="pubmed_id", keep=False, inplace=True)

df.info()
from Bio import Entrez



Entrez.api_key = '14e49850c0c543c6475d9a78943173bc8508'

Entrez.email = 'pubmed.query@gmail.com'
refs = []

allids = [int(id) for id in df['pubmed_id'].tolist()]

idbatches = [allids[x:x+100] for x in range(0, len(allids), 100)]

for i, ids in enumerate(idbatches):

    print('Processing batch %i' % i)

    handle = Entrez.elink(dbfrom='pubmed', id=ids, linkname='pubmed_pubmed_refs')

    results = Entrez.read(handle)

    

    for res in results:

        if res["LinkSetDb"] == []:

            pmids = []

        else:

            pmids = [int(link["Id"]) for link in res["LinkSetDb"][0]["Link"]]

        refs.append(pmids)

        

df['refs'] = refs

allrefs = [ref for reflist in df['refs'].tolist() for ref in reflist] # might contain duplicates
seen = {}

commonrefs = []



for x in allrefs:

    if x not in seen:

        seen[x] = 1

    else:

        if seen[x] == 1:

            commonrefs.append(x)

        seen[x] += 1

print('There are %i refs that occur more than once.' % len(commonrefs))
shared = []

cross = {id:1 for id in allids}

for index, row in df.iterrows():

    sharedrefs = [ref for ref in row['refs'] if seen[ref] > 1 or ref in cross]

    shared.append(sharedrefs)
df['sharedrefs'] = shared

df['nsharedrefs'] = df['sharedrefs'].apply(lambda x: len(x))

df = df[df['nsharedrefs'] > 0]
nodes = allids + commonrefs

node_types = [1 for pub in allids] + [0 for ref in commonrefs]

node_titles = [title for title in df['title'].tolist()] + ['' for ref in commonrefs]

df_nodes = pd.DataFrame(list(zip(nodes, node_types, node_titles)), columns =['Id', 'Type', 'Label'])

df_nodes.to_csv('nodes.csv', index=False)  
sources = []

targets = []

for _,row in df.iterrows():

    source = int(row['pubmed_id'])

    for ref in row['sharedrefs']:

        target = ref

        sources.append(source)

        targets.append(target)

        

df_edges = pd.DataFrame(list(zip(sources, targets)), columns =['Source', 'Target'])

df_edges.to_csv('edges.csv', index=False)  