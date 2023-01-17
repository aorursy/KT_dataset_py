class config():    

    INPUT_DIR="/kaggle/input/CORD-19-research-challenge/"

    OUTPUT_DIR="/kaggle/working/"
import pandas as pd

import re
meta_df = pd.read_csv(f'{config.INPUT_DIR}/metadata.csv',

                      usecols=['publish_time', 'title', 'abstract',

                               'pmc_json_files',

                               'doi'

                              ])

meta_df['publish_time'] = pd.to_datetime(meta_df['publish_time'])

meta_df.info()
meta_df.dropna(inplace=True)

meta_df.info()
meta_df['abstract_nw'] = meta_df.abstract.str.split().apply(len)
print(meta_df.query('abstract_nw<=3').abstract.values)
meta_df = meta_df.query('abstract_nw>3')
meta_df['title_lower'] = meta_df['title'].str.lower()
idx = meta_df.groupby(['title_lower'])['publish_time'].transform(max) == meta_df['publish_time']

meta_df = meta_df[idx]

meta_df.info()
meta_df.drop_duplicates(subset ="title_lower", keep='last', inplace=True)

meta_df.info()
meta_df.abstract.str.lower().str.extract(r'^(\w+)').iloc[:,0].value_counts().head(15)
max_loops = 5

abstract_first_words = ['abstract', 'background', 'objective', 'publisher', 'introduction', 'summary']

while max_loops:

    changed = 0

    for w in abstract_first_words:

        regex_pat = re.compile(f'^{w}s?'+ r'\s*', flags=re.IGNORECASE)

        idx = meta_df.abstract.str.contains(regex_pat, na=False)

        changed += sum(idx)

        meta_df.loc[idx, 'abstract'] = meta_df[idx].abstract.replace(regex_pat, '')

        

    regex_pat = re.compile(r'^[.:/\-&]+\s*')

    idx = meta_df.abstract.str.contains(regex_pat, na=False)

    changed += sum(idx)

    meta_df.loc[idx, 'abstract'] = meta_df[idx].abstract.replace(regex_pat, '')

        

    print(changed)

    if changed == 0:

        break

    max_loops -= 1

meta_df.abstract.str.lower().str.extract(r'^(\w+)').iloc[:,0].value_counts().head(15)
meta_df['abstract_lower'] = meta_df.abstract.str.lower()

idx = meta_df.groupby(['abstract_lower'])['publish_time'].transform(max) == meta_df['publish_time']

meta_df = meta_df[idx]

meta_df.info()
meta_df.drop(['abstract_nw', 'title_lower', 'abstract_lower', 'langid'], axis=1, inplace=True)
len(meta_df)
meta_df.reset_index(drop=True, inplace=True)
meta_df.to_pickle(config.OUTPUT_DIR+'meta_df.pkl')