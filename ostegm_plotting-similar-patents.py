%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import bq_helper

patents_helper = bq_helper.BigQueryHelper(
    active_project="patents-public-data",
    dataset_name="patents"
)
# A list of patents we care about. In this case we're going to manually enter a set of patents related to codecs.
input_patents = [
    'US-7292636-B2',
    'US-6115503-A',
    'US-6812873-B1',
    'US-6825782-B2',
    'US-6850175-B1',
    'US-6934331-B2',
    'US-6967601-B2',
    'US-7068721-B2',
    'US-7183951-B2',
    'US-7190289-B2',
    'US-7199836-B1',
    'US-7298303-B2',
    'US-7310372-B2',
    'US-7339991-B2',
    'US-7346216-B2',
    'US-7474699-B2',
]
# Converts input list into a string for use in queries.
input_patents_str = '(' + str(input_patents)[1:-1] + ')'
input_patents_str
query = '''
#standardSQL
SELECT DISTINCT cpc_code 
FROM (
    SELECT
    publication_number,
    c.code as cpc_code

    FROM `patents-public-data.patents.publications`
    ,UNNEST(cpc) as c

    where publication_number in {}
)
'''.format(input_patents_str)

all_cpcs = patents_helper.query_to_pandas(query=query)
# Convert into helper string for limiting CPCs
all_cpcs_str = '(' + str(list(all_cpcs.cpc_code.values))[1:-1] + ')'
# Get sample of patents not in our input set, but sharing at least 1 cpc.
query = '''
SELECT DISTINCT publication_number
FROM `patents-public-data.patents.publications`
,UNNEST(cpc) as cpc
where publication_number not in {}
and cpc.code in {}
and rand() < 0.2
limit 100
'''.format(input_patents_str, all_cpcs_str)
shared_cpc = patents_helper.query_to_pandas_safe(query, max_gb_scanned=5)
shared_cpc.loc[:, 'source'] = 'shared_cpc'
shared_cpc.head()
# Get sample of 100 random patents not sharing any CPC's.
query = '''
SELECT DISTINCT publication_number
FROM `patents-public-data.patents.publications`
,UNNEST(cpc) as cpc
where publication_number not in {}
and cpc.code not in {}
and rand() < 0.2
limit 100
'''.format(input_patents_str, all_cpcs_str)
no_shared_cpc = patents_helper.query_to_pandas_safe(query, max_gb_scanned=5)
no_shared_cpc.loc[:, 'source'] = 'no_shared_cpc'
no_shared_cpc.head()
# Pull all the "similar patents" from Patents Research dataset.
# Each of our patents in the input list should have ~25 similar patents listed, so we get back 12*25 rows
query = '''
SELECT distinct
s.publication_number

FROM `patents-public-data.patents.publications` p
JOIN `patents-public-data:google_patents_research.publications` r
  on p.publication_number = r.publication_number
, UNNEST(similar) as s
where p.publication_number in {}
'''.format(input_patents_str)
similar = patents_helper.query_to_pandas_safe(query, max_gb_scanned=36)
similar.loc[:, 'source'] = 'similar_to_input'
print(len(similar))
# Lets constuct our dataframe by concatenating our input list, the close negatives and 
# the list of "similar patents" according to the patents research table.
df = pd.DataFrame(input_patents, columns=['publication_number'])
df.loc[:, 'source'] = 'input'
df = pd.concat(
    [df, similar, shared_cpc, no_shared_cpc]).drop_duplicates('publication_number', keep='first')
df.source.value_counts()
all_patents_str = '(' + str(list(df.publication_number.unique()))[1:-1] + ')'
query = r'''
CREATE TEMPORARY FUNCTION convert_embedding_to_string(embedding ARRAY<FLOAT64>)
RETURNS STRING
LANGUAGE js AS """
let embedding_str = ''
for (i = 0; i < embedding.length; i++) { 
  embedding_str += embedding[i].toFixed(6) + ',';
} 
return embedding_str
"""; 

SELECT 
publication_number,
convert_embedding_to_string(embedding_v1) embedding
FROM `patents-public-data.google_patents_research.publications` 
where publication_number in %s
''' % (all_patents_str)

results = patents_helper.query_to_pandas_safe(query, max_gb_scanned=50).drop_duplicates('publication_number')
# Put the string embedding into 64 float cols.
embeddings = pd.DataFrame(
    data=[e for e in results.embedding.apply(lambda x: x.split(',')[:64]).values],
    columns = ['x{}'.format(i) for i in range(64)],
    index=results.publication_number
)
embeddings = embeddings.astype(float).reset_index()
embeddings.head()
# Merge the embeddings into the dataframe.
df = df.merge(embeddings, on='publication_number').drop_duplicates('publication_number')
df.head()
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
principal_components = pca.fit_transform(df.iloc[:, 2:].values)
pca_df = pd.DataFrame(
    data = principal_components
    ,columns = ['principal component 1', 'principal component 2']
)

plot_df = pd.concat([pca_df, df[['source']]], axis = 1)
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = plot_df.source.unique()
colors = ['r', 'g', 'b', 'y']
for source, color in zip(targets,colors):
    indicesToKeep = plot_df['source'] == source
    ax.scatter(plot_df.loc[indicesToKeep, 'principal component 1']
               , plot_df.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 12)
ax.legend(targets)
ax.grid()










