# Data libraries

import pandas as pd

import re

import pycountry



# Visualisation libraries

import plotly.express as px

import plotly.graph_objects as go



%matplotlib inline



# Load data

metadata_file = '../input/CORD-19-research-challenge/metadata.csv'

df = pd.read_csv(metadata_file,

                 dtype={'Microsoft Academic Paper ID': str,

                        'pubmed_id': str})



def doi_url(d):

    if d.startswith('http'):

        return d

    elif d.startswith('doi.org'):

        return f'http://{d}'

    else:

        return f'http://doi.org/{d}'

    

df.doi = df.doi.fillna('').apply(doi_url)



print(f'loaded DataFrame with {len(df)} records')
# Helper function for filtering df on abstract + title substring

def abstract_title_filter(search_string):

    return (df.abstract.str.lower().str.replace('-', ' ').str.contains(search_string, na=False) |

            df.title.str.lower().str.replace('-', ' ').str.contains(search_string, na=False))

def full_text_filter(search_string):

    return (df.abstract.str.lower().str.replace('-', ' ').str.contains(search_string, na=False) |

            df.title.str.lower().str.replace('-', ' ').str.contains(search_string, na=False)|

            df.full_text.str.lower().str.replace('-', ' ').str.contains(search_string, na=False))

# Helper function which counts synonyms and adds tag column to DF

def count_and_tag(df: pd.DataFrame,

                  synonym_list: list,

                  tag_suffix: str) -> (pd.DataFrame, pd.Series):

    counts = {}

    df[f'tag_{tag_suffix}'] = False

    for s in synonym_list:

        synonym_filter = abstract_title_filter(s)

        counts[s] = sum(synonym_filter)

        df.loc[synonym_filter, f'tag_{tag_suffix}'] = True

    return df, pd.Series(counts)



# Helper function which counts synonyms and adds tag column to DF

def count_and_tag_full_text(df: pd.DataFrame,

                  synonym_list: list,

                  tag_suffix: str) -> (pd.DataFrame, pd.Series):

    counts = {}

    df[f'tag_{tag_suffix}'] = False

    for s in synonym_list:

        synonym_filter = full_text_filter(s)

        counts[s] = sum(synonym_filter)

        df.loc[synonym_filter, f'tag_{tag_suffix}'] = True

    return df, pd.Series(counts)
# Helper function for Cleveland dot plot visualisation of count data

def dotplot(input_series, title, x_label='Count', y_label='Regex'):

    subtitle = '<br><i>Hover over dots for exact values</i>'

    fig = go.Figure()

    fig.add_trace(go.Scatter(

    x=input_series.sort_values(),

    y=input_series.sort_values().index.values,

    marker=dict(color="crimson", size=12),

    mode="markers",

    name="Count",

    ))

    fig.update_layout(title=f'{title}{subtitle}',

                  xaxis_title=x_label,

                  yaxis_title=y_label)

    fig.show()

# Function for printing out key passage of abstract based on key terms

def print_key_phrases(df, key_terms, n=5, chars=300):

    for ind, item in enumerate(df[:n].itertuples()):

        print(f'{ind+1} of {len(df)}')

        print(item.title)

        print('[ ' + item.doi + ' ]')

        try:

            i = len(item.abstract)

            for kt in key_terms:

                kt = kt.replace(r'\b', '')

                term_loc = item.abstract.lower().find(kt)

                if term_loc != -1:

                    i = min(i, term_loc)

            if i < len(item.abstract):

                print('    "' + item.abstract[i-30:i+chars-30] + '"')

            else:

                print('    "' + item.abstract[:chars] + '"')

        except:

            print('NO ABSTRACT')

        print('---')
covid19_synonyms = ['covid',

                    'coronavirus disease 19',

                    'sars cov 2', # Note that search function replaces '-' with ' '

                    '2019 ncov',

                    '2019ncov',

                    r'2019 n cov\b',

                    r'2019n cov\b',

                    'ncov 2019',

                    r'\bn cov 2019',

                    'coronavirus 2019',

                    'wuhan pneumonia',

                    'wuhan virus',

                    'wuhan coronavirus',

                    r'coronavirus 2\b']
df, covid19_counts = count_and_tag(df, covid19_synonyms, 'disease_covid19')
covid19_counts.sort_values(ascending=False)
novel_corona_filter = (abstract_title_filter('novel corona') &

                       df.publish_time.str.startswith('2020', na=False))

print(f'novel corona (published 2020): {sum(novel_corona_filter)}')

df.loc[novel_corona_filter, 'tag_disease_covid19'] = True
df[df.tag_disease_covid19].publish_time.str.slice(0, 4).value_counts(dropna=False)
df.loc[df.tag_disease_covid19 & ~df.publish_time.str.startswith('2020', na=True),

       'tag_disease_covid19'] = False
repr_synonyms = [

    'math',

    's.i.r', 'sir',

    'seir','s.e.i.r',

    'sis','s.i.s',

    'sirs','s.i.r.s',

    'seis', 's.e.i.s',

    'seirs','s.e.i.rs',

    'msir', 'm.s.i.r.',

    'mseir', 'm.s.e.i.r',

    'mseirs','m.s.e.i.r.s',

    'seirv','s.e.i.r.v',

    'nac', 'nac', 

    'nac seirv','nac s.e.i.r.v',

    'sirv','s.i.r.v',

    'siqr','s.i.q.r',

    'sveis','s.v.e.i.s',

    'seiqs','s.e.i.q.s',

    'iar','i.a.r',

    'network model',

]
df, repr_counts = count_and_tag(df,repr_synonyms, 'Math_models_used')

dotplot(repr_counts, 'Math models by title / abstract metadata')
repr_counts.sort_values(ascending=False)
n_math = (df.tag_disease_covid19 & df.tag_Math_models_used).sum()

n_math
msir_synonyms = [

    'msir','m.s.i.r',

]

df, msir_counts = count_and_tag(df,msir_synonyms, 'MSIR')

dotplot(msir_counts, 'MSIR models by title / abstract metadata')

msir_counts.sort_values(ascending=False)

n_msir = (df.tag_disease_covid19 & df.tag_MSIR).sum()

n_msir

n_msir_no_covid_filter = (df.tag_MSIR).sum()

n_msir_no_covid_filter
sirv_synonyms = [

    'sirv','s.i.r.v',

]

df, sirv_counts = count_and_tag(df,sirv_synonyms, 'SIRV')

dotplot(sirv_counts, 'SIRV models by title / abstract metadata')

sirv_counts.sort_values(ascending=False)

n_sirv = (df.tag_disease_covid19 & df.tag_SIRV).sum()

n_sirv

n_sirv_no_covid_filter = (df.tag_SIRV).sum()

n_sirv_no_covid_filter
seis_synonyms = [

    'seis', 's.e.i.s',

]

df, seis_counts = count_and_tag(df,seis_synonyms, 'SEIS')

dotplot(seis_counts, 'SEIS models by title / abstract metadata')

seis_counts.sort_values(ascending=False)

n_seis = (df.tag_disease_covid19 & df.tag_SEIS).sum()

n_seis

n_seis_no_covid_filter = (df.tag_SEIS).sum()

n_seis_no_covid_filter

print(df[df.tag_disease_covid19 & df.tag_SEIS]['title'])
sirs_synonyms = [

    'sirs','s.i.r.s',

]

df, sirs_counts = count_and_tag(df,sirs_synonyms, 'SIRS')

dotplot(sirs_counts, 'SIRS models by title / abstract metadata')

sirs_counts.sort_values(ascending=False)

n_sirs = (df.tag_disease_covid19 & df.tag_SIRS).sum()

n_sirs

n_sirs_no_covid_filter = (df.tag_SIRS).sum()

n_sirs_no_covid_filter 

print(df[df.tag_disease_covid19 & df.tag_SIRS]['title'])
seir_synonyms = [

    'seir','s.e.i.r',

]

df, seir_counts = count_and_tag(df,seir_synonyms, 'SEIR')

dotplot(seir_counts, 'SEIR models by title / abstract metadata')

seir_counts.sort_values(ascending=False)

n_seir = (df.tag_disease_covid19 & df.tag_SEIR).sum()

n_seir



n_seir_no_covid_filter = (df.tag_SEIR).sum()

n_seir_no_covid_filter
print(df[df.tag_disease_covid19 & df.tag_SEIR]['title'])
sir_synonyms = [

    'sir','s.i.r',

]

df, sir_counts = count_and_tag(df,sir_synonyms, 'SIR')

dotplot(sir_counts, 'SIR models by title / abstract metadata')

sir_counts.sort_values(ascending=False)

n_sir = (df.tag_disease_covid19 & df.tag_SIR).sum()

n_sir



n_sir_no_covid_filter = (df.tag_SIR).sum()

n_sir_no_covid_filter
print(df[df.tag_disease_covid19 & df.tag_SIR]['title'])
sis_synonyms = [

    'sis','s.i.s',

]

df, sis_counts = count_and_tag(df,sis_synonyms, 'SIS')

dotplot(sis_counts, 'SIS models by title / abstract metadata')

sis_counts.sort_values(ascending=False)

n_sis = (df.tag_disease_covid19 & df.tag_SIS).sum()

n_sis

n_sis_no_covid_filter = (df.tag_SIS).sum()

n_sis_no_covid_filter
print(df[df.tag_disease_covid19 & df.tag_SIS]['title'])
from bokeh.io import output_file, show, output_notebook

from bokeh.plotting import figure

output_notebook()

models_name = ['SIR','SIS','SIRV','SEIS','SEIR','SIRS','MSIR']

counts = [n_sir,n_sis,n_sirv,n_seis,n_seir,n_sirs,n_msir]

p = figure(x_range=models_name, plot_height=250, title="Model Counts")

p.xgrid.grid_line_color = None

p.y_range.start = 0

p.vbar(x=models_name, top=counts, width=0.9)

show(p)



counts

from bokeh.io import output_file, show, output_notebook

from bokeh.plotting import figure

output_notebook()

models_name = ['SIR','SIS','SIRV','SEIS','SEIR','SIRS','MSIR']

counts_no = [n_sir_no_covid_filter,n_sis_no_covid_filter,n_sirv_no_covid_filter,n_seis_no_covid_filter,n_seir_no_covid_filter,n_sirs_no_covid_filter,n_msir_no_covid_filter]

p = figure(x_range=models_name, plot_height=250, title="Model Counts without COVID19 filter")

p.xgrid.grid_line_color = None

p.y_range.start = 0

p.vbar(x=models_name, top=counts_no, width=0.9)

show(p)

counts_no
df_SIR = df[df.tag_disease_covid19 & df.tag_SIR]

df_SIR.head()

df_SIR.shape
parameter_sir_synonyms = [

    'beta','betta',

    'gama','gamma'

]

df, parameter_sir_counts = count_and_tag(df,parameter_sir_synonyms, 'parameter_sir')

dotplot(parameter_sir_counts, 'parameter_sir by text metadata')

parameter_sir_counts.sort_values(ascending=False)

n = (df.tag_disease_covid19 & df.tag_SIR & df.tag_parameter_sir).sum()

n
print_key_phrases(df[df.tag_disease_covid19 & df.tag_SIR & df.tag_parameter_sir], parameter_sir_synonyms, n=52, chars=500)