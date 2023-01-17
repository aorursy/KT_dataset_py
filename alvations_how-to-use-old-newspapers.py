import pandas as pd
df = pd.read_csv('../input/old-newspaper.tsv', sep='\t', error_bad_lines=False)
df.head(10)
languages = df['Language'].unique()

languages