import os
import numpy as np
import pandas as pd
from itertools import cycle, chain
aminer_dir = '../input/'
with open(os.path.join(aminer_dir, 'AMiner-Author.txt'), 'r') as f:
    dict_list = []
    c_dict = {}
    for i, line in enumerate(f):
        c_line = line.strip()[1:].strip()
        if len(c_line)<1:
            if len(c_dict)>0:
                dict_list += [c_dict]
            c_dict = {}
        else:
            c_frag = c_line.split(' ')
            c_dict[c_frag[0]] = ' '.join(c_frag[1:])
author_df = pd.DataFrame(dict_list)
author_df.rename({'a': 'Affiliation',
                 'n': 'Author', 
                 'pc': 'Papers',
                 'cn': 'Citations',
                  'hi': 'H-index',
                  't': 'research interests'
                 }, axis=1, inplace=True)
author_df.to_csv('author_combined.csv')
author_df.sample(3)
print(author_df.shape[0], 'authors')
zrh_match = author_df['Affiliation'].map(lambda x: 'ZURICH' in x.upper())
uzh_match = author_df['Affiliation'].map(lambda x: 'UNIVERSITY OF ZURICH' in x.upper())
print(zrh_match.sum(), 'in zurich')
print(uzh_match.sum(), 'at uzh')
author_df[uzh_match].sample(3)
major_keywords = author_df[uzh_match]['research interests'].\
    map(lambda x: x.split(';')).\
    values.tolist()
major_keywords = pd.DataFrame({'keyword': list(chain(*major_keywords))})
major_keywords.\
    groupby('keyword').\
    size().\
    reset_index(name='count').\
    sort_values('count', ascending=False).\
    head(12).\
    plot.bar(x='keyword', y='count')
author_df[uzh_match].sample(5)['Affiliation'].values
author_df[uzh_match].head(10)
author_df[zrh_match & author_df['Author'].map(lambda x: 'KEVIN' in x.upper())]
