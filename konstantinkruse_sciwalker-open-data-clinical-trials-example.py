PROJECT_ID = 'sciwalker-open-data'

from google.cloud import bigquery

client = bigquery.Client(project=PROJECT_ID)
query = (

"select distinct ct.NI_COMPOUND_NAME as compound, ct.NI_CONDITION_NAME as disease, ct.SOURCE as ct_id "+

"from ontologies.chemistry chem "+

"inner join ( "+

" select distinct ocid "+ 

" from ontologies.compound_classes "+

" where name='sesquiterpene derivatives' ) cls on cls.ocid=chem.ancestorid "+

"inner join clinical_trials_aact.aact_relations ct on ct.OCID_SUBJECT_COMPOUND=chem.ocid" )



query_job = client.query(query, location="US")  

df = query_job.to_dataframe()
from pandas import crosstab

tab = crosstab(index=df['disease'], columns=df['compound'])
import matplotlib.pyplot as plt

import seaborn as sns

sns.set(font_scale=1.5) 

fig, (ax) = plt.subplots(1, 1, figsize=(20,15))

hm = sns.heatmap(tab, ax=ax, cmap="YlOrRd", annot=True, vmin=0, vmax=10, linewidths=.05)