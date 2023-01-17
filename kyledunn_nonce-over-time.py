import numpy as np
import pandas as pd
from bq_helper import BigQueryHelper

# Helper object for BigQuery Ethereum dataset
eth = BigQueryHelper(active_project="bigquery-public-data", 
                     dataset_name="ethereum_blockchain")
query = """
SELECT
    number,
    nonce
FROM
    `bigquery-public-data.ethereum_blockchain.blocks`
ORDER BY
    1 asc
"""

print("Expected scan volume: {:.2f} GB".format(eth.estimate_query_size(query)))
# Store the results into a Pandas DataFrame
df = eth.query_to_pandas_safe(query, max_gb_scanned=1)
import altair as alt
alt.renderers.enable('notebook');
df.head()
df['inonce'] = df.nonce.map(lambda s: int(s, 0))
c = alt.Chart(df[-5000:]).mark_point(size=10, filled=True).encode(
    x=alt.X('number:T', axis=alt.Axis(labels=False, title=None)),
    y=alt.Y('inonce:Q', axis=alt.Axis(title='nonce', format=".2"))
).properties(
    title="Ethereum nonce over time",
    height=400,
    width=750
)

c.display()
#c.to_html()
