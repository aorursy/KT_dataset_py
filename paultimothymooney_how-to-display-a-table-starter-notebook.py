import numpy as np 

import pandas as pd 

from IPython.display import HTML





def make_clickable(url, title):

    return '<a href="{}">{}</a>'.format(url,title)

df = pd.read_csv('/kaggle/input/aipowered-literature-review-csvs/kaggle/working/Key Scientific Questions/Human immune response to COVID-19.csv',header=None)

list_of_columns = ['Date', 

                   'URL',

                   'Result',

                   'Study Type', 

                   'Measure of Evidence Strength', 

                   'Sample (n)'] # Define columns

df = df.rename(columns=df.iloc[0]).drop(df.index[0]) # Reset header

df['URL'] = df.apply(lambda x: make_clickable(x['Study Link'], x['Study']), axis=1) # Add in link

df = df[list_of_columns] # Drop extra columns

print('Human Immune Response to COVID-19')

HTML(df.to_html(escape=False))