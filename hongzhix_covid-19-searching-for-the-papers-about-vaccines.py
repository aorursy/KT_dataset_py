from IPython.core.display import display, HTML

import pandas as pd

import re

import ipywidgets



def search_papers(title: str):

    

    df = pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv')

    

    if title != '':

        df_s = df.loc[df['title'].fillna('').str.contains(title, case=False),['title','abstract','doi']]

        df_s['title'] = [re.sub(title,'<span style="background-color:lime">' + title + '</span>', i, flags=re.IGNORECASE) for i in df_s['title']]

        df_s['abstract'] = [re.sub(title,'<span style="background-color:lime">' + title + '</span>', j, flags=re.IGNORECASE) for j in df_s['abstract'].fillna('')]

        df_s['doi'] = '<a href = "https://doi.org/' + df_s['doi'] + '" target="_blank">link</a>'

        msg = str(len(df_s)) + ' papers'

        if len(df_s) > 2000:

            df_s = df_s.head(2000)

            msg = '2000 of ' + msg

        results = HTML(msg + df_s.to_html(escape=False))

        

    else:

        msg = 'Please enter a keyword'

        results = HTML(msg)

    

    return display(results)



ipywidgets.interactive(search_papers, title='')
ipywidgets.interactive(search_papers, title='vaccine')