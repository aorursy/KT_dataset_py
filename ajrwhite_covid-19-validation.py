import pandas as pd

import covid19_tools as cv19
df = pd.read_csv('../input/covid19-validation-set/covid_validation_set.csv')
df['Date'] = pd.to_datetime(df.Date)
df.head()
meta = cv19.load_metadata('../input/CORD-19-research-challenge/metadata.csv')
meta, covid19_counts = cv19.add_tag_covid19(meta)
val_df = df.merge(meta, left_on='URL', right_on='url', how='inner')
val_df.shape
val_df[['Title', 'Date', 'URL', 'tag_disease_covid19']].to_csv('andy_validation_results.csv')