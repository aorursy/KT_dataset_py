import covid19_tools as cv19

import pandas as pd

import re

from IPython.core.display import display, HTML

import html



METADATA_FILE = '../input/CORD-19-research-challenge/metadata.csv'



# Load metadata

meta = cv19.load_metadata(METADATA_FILE)

# Add tags

meta, covid19_counts = cv19.add_tag_covid19(meta)



pd.set_option('display.max_columns', 500)

pd.set_option('display.max_rows', 500)
n = (meta.tag_disease_covid19 & cv19.abstract_title_filter(meta, 'chloroquine')).sum()

print(f'Found {n} papers on Covid-19 and chloroquine')
for item in (meta[meta.tag_disease_covid19 &

                 cv19.abstract_title_filter(meta, 'chloroquine')].itertuples()):

    display_str = f'<b><i><a href="{item.doi}">{item.title}</a></i></b>'

    display_str += f'<br>{item.authors}<br>({item.publish_time})<br><br>'

    abstract = html.escape(str(item.abstract))

    abstract = re.sub('(?i)chloroquine', '<mark><b>chloroquine</b></mark>', abstract)

    abstract = re.sub('hydroxy<mark><b>', '<mark><b>hydroxy', abstract)

    display_str += f'{abstract}<br><br>'

    display(HTML(display_str))