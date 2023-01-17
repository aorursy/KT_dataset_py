!pip install tabula-py

import numpy as np 

import pandas as pd

from tabula import read_pdf,convert_into

print('packages loaded')

print('')

file='https://www.medrxiv.org/content/10.1101/2020.04.16.20067587v1.full.pdf'

print ('this is the document we are testing ', file)

print('')

tables = read_pdf(file, pages = 'all', multiple_tables = True)



print('tables extracted')
from IPython.core.display import display, HTML



output='table dataframes returned '+str(len(tables))

display(HTML('<h3>'+output+'</h3>'))

i=0



df_names=['A','B','C','D','E','F','G','H','I','J','K','L']



for table in tables:

    df='df'+df_names[i]

    df = pd.DataFrame(table)

    df=df.fillna('missing data')

    df=df[df[df.columns[0]].str.lower().str.contains('hypertension')]

    if df.empty==False:

        df_table_show=HTML(df.to_html(escape=False,index=False))

        display(df_table_show)

    i=i+1
def get_tables(file,keyword):

    tables = read_pdf(file, pages = 'all', multiple_tables = True)

    i=0



    df_names=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']



    for table in tables:

        df='df'+df_names[i]

        df = pd.DataFrame(table)

        df=df.fillna('missing data')

        df=df[df[df.columns[0]].str.lower().str.contains(keyword)]

        if df.empty==False:

            df_table_show=HTML(df.to_html(escape=False,index=False))

            display(df_table_show)

        i=i+1



keyword='hypertension'  



file='https://www.medrxiv.org/content/10.1101/2020.04.15.20063107v1.full.pdf'



tables=get_tables(file,keyword)

file='https://www.medrxiv.org/content/10.1101/2020.04.08.20057794v1.full.pdf'



tables=get_tables(file,keyword)

file='https://www.medrxiv.org/content/10.1101/2020.03.31.20038935v1.full.pdf'



tables=get_tables(file,keyword)

file='https://www.medrxiv.org/content/10.1101/2020.03.23.20041848v2.full.pdf'



tables=get_tables(file,keyword)

file='https://www.medrxiv.org/content/10.1101/2020.03.25.20037721v2.full.pdf'



tables=get_tables(file,keyword)
file='https://www.medrxiv.org/content/10.1101/2020.03.24.20042283v1.full.pdf'



tables=get_tables(file,keyword)
file='https://www.medrxiv.org/content/10.1101/2020.04.08.20057794v1.full.pdf'



tables=get_tables(file,keyword)
