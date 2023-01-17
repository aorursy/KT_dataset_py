import os

import re

import pandas as pd

import numpy as np

import shutil

from IPython.display import display, HTML

import plotly.express as px



PATH = '/kaggle/input/the-ontario-sunshine-list-raw-data/'

YEARS = range(1996, 2020)



px.defaults.template = 'plotly_white'

px.defaults.color_discrete_sequence = ['steelblue']

MODE_BAR_BUTTONS = ['toImage', 'zoom2d', 'pan2d', 'select2d', 'lasso2d',

                    'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d',

                    'toggleSpikelines', 'hoverClosestCartesian', 'hoverCompareCartesian']

CONFIG = {

    'modeBarButtonsToRemove': ['pan2d', 'select2d', 'lasso2d', 'toggleSpikelines']

}
filenames = os.listdir(PATH)

filenames
pd.read_csv(PATH + 'en-2018-pssd-compendium.csv', nrows=5)
pd.read_csv(PATH + 'en-2018-pssd-compendium-20191223.csv', nrows=5)
filenames
p = re.compile('(?:^|-)(\d{4})\D')
def read_csv(filename):

    try:

        return pd.read_csv(PATH + filename, encoding='utf-8')

    except UnicodeDecodeError:

        return pd.read_csv(PATH + filename, encoding='latin1')
pss = {}

for filename in filenames:

    if filename == 'en-2018-pssd-compendium.csv':

        pss[2017] = read_csv(filename)

        continue

    m = p.search(filename)

    year = int(m.group(1))

    pss[year] = read_csv(filename)
for year in YEARS:

    display(HTML(pss[year].describe(include='all').to_html()))
refcols = pss[2019].columns

for year in YEARS:

    if not refcols.equals(pss[year].columns):

        print(year, pss[year].columns.tolist(), sep='\n', end='\n\n')



print("2019", refcols.tolist(), sep='\n')
pss[1996].head()
pss[1996]['Unnamed: 8'].notna().any()
pss[1996].drop(columns='Unnamed: 8', inplace=True)
for year in YEARS:

    pss[year].columns = refcols
for year in YEARS:

    print(year, pss[year].dtypes, sep='\n', end='\n\n')
pss[2016]['Calendar Year'].nunique()
wrong_year = pss[2016][pss[2016]['Calendar Year'] != '2016']

wrong_year.shape[0]
wrong_year.head(20)
pss_comb = pd.concat([pss[year] for year in YEARS]).copy()
pss_comb[pss_comb['Last Name'].str.contains('Fagan', case=False) & pss_comb['First Name'].str.contains('Thomas', case=False)]
pss_comb[pss_comb['Last Name'].str.contains('leblanc', case=False) & pss_comb['First Name'].str.contains('Laurie', case=False)]
pss_comb[pss_comb['Last Name'].str.contains('levac', case=False) & pss_comb['First Name'].str.contains('jody', case=False)]
pss[2016].loc[pss[2016]['Calendar Year'] != '2016', 'Job Title'] = wrong_year['Job Title'].str.cat(wrong_year['Calendar Year'], sep='; ')

pss[2016].loc[pss[2016]['Calendar Year'] != '2016', 'Calendar Year'] = '2016'

pss[2016]['Calendar Year'] = pss[2016]['Calendar Year'].astype('int')

pss[2016].dtypes
for year in YEARS:

    if pss[year]['Calendar Year'].nunique() > 1:

        print(year, pss[year]['Calendar Year'].unique(), sep='\n')
wrong_year_2015 = pss[2015][pss[2015]['Calendar Year'] != 2015]

wrong_year_2015
pss[2015].loc[wrong_year_2015.index, 'Calendar Year'] = 2015
pss_comb = pd.concat([pss[year] for year in YEARS]).copy().reset_index(drop=True)



salary_nonnum = pss_comb['Salary Paid'].str.extractall('([^$\d.,\s])').drop_duplicates()

salary_nonnum
tax_ben_nonnum = pss_comb['Taxable Benefits'].str.extractall('([^$\d.,\s])').drop_duplicates()

tax_ben_nonnum
idx = tax_ben_nonnum.reset_index(level='match', drop=True).index

pss_comb.loc[idx, 'Taxable Benefits'].unique()
for year in YEARS:

    pss[year]['Salary Paid'] = (pss[year]['Salary Paid']

                                .replace('[$,]', '', regex=True)

                                .replace('-', np.nan)

                                .astype('float')

                               )

    pss[year]['Taxable Benefits'] = (pss[year]['Taxable Benefits']

                                     .replace('[$,]', '', regex=True)

                                     .replace('-', np.nan)

                                     .astype('float')

                                    )
for year in YEARS:

    pss[year] = pss[year].convert_dtypes()

    print(year, pss[year].dtypes, sep='\n', end='\n\n')
for year in YEARS:

    if pss[year].isna().sum().sum() != 0:

        print(year, pss[year].isna().sum(), sep='\n', end='\n\n')
pss[2015]['Taxable Benefits'].eq(0).sum()
pss_comb = (pd.concat([pss[year] for year in YEARS])

            .copy()

            .reset_index(drop=True)

           )

no_tax_ben = (pss_comb

              .loc[pss_comb['Taxable Benefits'].eq(0), 'Calendar Year']

              .value_counts()

              .reindex(list(YEARS), fill_value=0)

              .to_frame()

              .reset_index()

              .rename(columns={'index': 'Calendar Year', 'Calendar Year': 'Number of employees'})

             )
fig = px.scatter(no_tax_ben, x='Calendar Year', y='Number of employees')

fig.update_traces(mode='lines+markers',

                  hovertemplate=

                  '<b>%{x}</b><br>'+

                  'Number of employees: <b>%{y}</b>'

                 )

fig.update_layout(title='Number of employees that did not receive taxable benefits by calendar year',

                  xaxis_title='Calendar Year',

                  yaxis_title="Number of employees",

                  yaxis_tickformat=',',

                  hoverlabel_bgcolor="white",

                  hoverlabel_font_size=14,

                  hovermode="x",

                  yaxis_zerolinecolor='grey',

                  yaxis_zerolinewidth=1

                 )

fig.show(config=CONFIG)
pss[2015]['Taxable Benefits'].fillna(0.0, inplace=True)

no_tax_ben.loc[no_tax_ben['Calendar Year'].eq(2015), 'Number of employees'] = pss[2015]['Taxable Benefits'].eq(0).sum()
fig = px.scatter(no_tax_ben, x='Calendar Year', y='Number of employees')

fig.update_traces(mode='lines+markers',

                  hovertemplate=

                  '<b>%{x}</b><br>'+

                  'Number of employees: <b>%{y}</b>'

                 )

fig.update_layout(title='Number of employees that did not receive taxable benefits by calendar year',

                  xaxis_title='Calendar Year',

                  yaxis_title="Number of employees",

                  yaxis_tickformat=',',

                  hoverlabel_bgcolor="white",

                  hoverlabel_font_size=14,

                  hovermode="x",

                  yaxis_zerolinecolor='grey',

                  yaxis_zerolinewidth=1

                 )

fig.show(config=CONFIG)
null_2016 = pss[2016][pss[2016]['Taxable Benefits'].isna()]

null_2016
pss_comb[pss_comb['Last Name'].eq('Malenfant') & pss_comb['First Name'].eq('James')]
pss[2016].loc[null_2016.index, 'Taxable Benefits'] = 0.0
null_2013 = pss[2013][pss[2013]['First Name'].isna()]

null_2013
pss_comb[pss_comb['Last Name'].str.contains('^li$', case=False) & pss_comb['Employer'].str.contains('eHealth') & pss_comb['Job Title'].str.contains('privacy', case=False)]
pss2013 = pd.read_csv(PATH + 'pssd-en-2013.csv', na_filter=False)

pss2013[pss2013['Last Name'].str.contains('^li$', case=False) & pss2013['Employer'].str.contains('eHealth') & pss2013['Job Title'].str.contains('privacy', case=False)]
pss[2013].loc[null_2013.index, 'First Name'] = 'NA'
null_1998 = pss[1998][pss[1998]['First Name'].isna()]

null_1998
pss_comb[pss_comb['Last Name'].str.contains('donnelly', case=False) & pss_comb['Employer'].str.contains('hydro', case=False)]
pss1998 = pd.read_csv(PATH + 'en-1998-pssd.csv', encoding='latin1', na_filter=False)

pss1998[pss1998['Last Name'].str.contains('donnelly', case=False) & pss1998['Employer'].str.contains('hydro', case=False)]
pss[1998][pss[1998]['First Name'].str.len().eq(2)].head()
pss[1998].loc[null_1998.index, 'First Name'] = 'NA'
null_1997 = pss[1997][pss[1997]['Job Title'].isna()]

null_1997
pss_comb[pss_comb['Last Name'].str.contains('walker', case=False) & pss_comb['Employer'].str.contains('ontario hydro', case=False)]
pss[1997].loc[null_1997.index, 'Job Title'] = 'Maintenance Superintendent'
null_1996 = pss[1996][pss[1996]['First Name'].isna()]

null_1996
pss_comb[pss_comb['Last Name'].str.contains('yearwood', case=False) & pss_comb['Calendar Year'].le(2010)]
pss1996 = pd.read_csv(PATH + 'en-1996-pssd.csv', encoding='latin1', na_filter=False)

pss1996[pss1996['Last Name'].str.contains('yearwood', case=False)]
pss[1996][pss[1996]['First Name'].str.len().eq(2)].head()
pss[1996].loc[null_1996.index, 'First Name'] = 'NA'
pss_comb = pd.concat([pss[year] for year in YEARS]).copy()
pss_comb[pss_comb['Last Name'].str.contains('malenfant', case=False) & pss_comb['First Name'].str.contains('andrew', case=False)]
pss_comb
pss_comb.sort_values(['Calendar Year', 'Sector', 'Employer', 'Last Name', 'First Name']).to_csv('pssd.csv', index=False)