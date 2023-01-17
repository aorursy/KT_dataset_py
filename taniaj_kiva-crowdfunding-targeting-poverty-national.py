import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.offline as py
py.init_notebook_mode(connected=True)
import seaborn as sns
from scipy.stats.mstats import gmean
from scipy.stats.stats import pearsonr

%matplotlib inline

sns.set(rc={"figure.figsize": (20,10), "axes.titlesize" : 18, "axes.labelsize" : 12, 
            "xtick.labelsize" : 14, "ytick.labelsize" : 14 }, 
        palette=sns.color_palette("OrRd_d", 20))

import warnings
warnings.filterwarnings('ignore')

!cp ../input/images/cell_subscription_levels.png .
# Original Kiva datasets
kiva_loans_df = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/kiva_loans.csv")
kiva_mpi_locations_df = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/kiva_mpi_region_locations.csv")
loan_theme_ids_df = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/loan_theme_ids.csv")
loan_themes_by_region_df = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/loan_themes_by_region.csv")

# Additional Kiva datasets
mpi_national_df = pd.read_csv("../input/mpi/MPI_national.csv")
# The subnational Kiva data has been enhanced with lat/long data
#mpi_subnational_df = pd.read_csv("../input/kiva-mpi-subnational-with-coordinates/mpi_subnational_coords.csv")

# Human Development Reports
hdi_df = pd.read_csv("../input/human-development/human_development.csv")

# UNDP gender inequality data
gender_development_df = pd.read_csv("../input/gender-development-inequality/gender_development_index.csv")
gender_inequality_df = pd.read_csv("../input/gender-development-inequality/gender_inequality_index.csv")

# World Bank population data
world_pop_df = pd.read_csv("../input/world-population/WorldPopulation.csv")

# World Bank Findex data
findex_df = pd.read_csv("../input/findex-world-bank/FINDEXData.csv")

# World Bank cellular subscription data
cellular_subscription_df = pd.read_csv("../input/world-telecommunications-data/Mobile cellular subscriptions.csv")
# Join datasets to get rural_pct
mpi_national_df.rename(columns={'Country': 'country'}, inplace=True)
loan_themes_country_df = loan_themes_by_region_df.drop_duplicates(subset=['country'])
mpi_national_df = mpi_national_df.merge(loan_themes_country_df[['country', 'rural_pct']], on=['country'], how='left')

# There are 52 nulls in rural_pct so lets fill these with the median value
mpi_national_df['rural_pct'].fillna(mpi_national_df['rural_pct'].median(), inplace=True)

# Calculate national mpi according to Kiva's method
mpi_national_df['MPI'] = mpi_national_df['MPI Rural']*mpi_national_df['rural_pct']/100 + mpi_national_df['MPI Urban']*(100-mpi_national_df['rural_pct'])/100
mpi_national_df.sample()
colorscale = [[0.0, 'rgb(230, 240, 255)'], [0.2, 'rgb(179, 209, 255)'], [0.4, 'rgb(102, 163, 255)'],\
              [0.6, 'rgb(26, 117, 255)'], [0.8, 'rgb(0, 71, 179)'], [1.0, 'rgb(0, 31, 77)']]
data = [dict(
        type='choropleth',
        locations= hdi_df.Country,
        locationmode='country names',
        z=hdi_df['Human Development Index (HDI)'],
        text=hdi_df.Country,
        colorscale = colorscale,
        colorbar=dict(autotick=False, tickprefix='', title='Findex'),
)]
layout = dict(
            title = 'Human Development Index',
            geo = dict(
            showframe = False, 
            showcoastlines = True, 
            projection = dict(type = 'Mercator')),
            margin = dict(t=30, b=30, l=10, r=10))
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False)
# Join
mpi_hdi_df = mpi_national_df.merge(hdi_df[['Country', 'Human Development Index (HDI)']], left_on=['country'], right_on=['Country'])
# Compare scores 
print("Correlation, p-value: ", pearsonr(mpi_hdi_df.loc[:, 'Human Development Index (HDI)'], mpi_hdi_df.loc[:, 'MPI']))
sns.regplot(x='MPI', y='Human Development Index (HDI)', data=mpi_hdi_df)
# Keep relevant indicators only
findex_key_ind_df = findex_df.loc[(findex_df['Indicator Name'] == 'Account (% age 15+) [ts]') 
                                  | (findex_df['Indicator Name'] == 'Borrowed from a financial institution (% age 15+) [ts]')
                                  | (findex_df['Indicator Name'] == 'Saved at a financial institution (% age 15+) [ts]')]
# Keep relevant Countries only (those for which we have MPI)
# Note: there are less countries available in findex than in Kiva loans.
#findex_key_ind_df['Country Name'].unique()

findex_key_ind_df = findex_key_ind_df[findex_key_ind_df['Country Name'].isin(kiva_mpi_locations_df['country'].unique())]

findex_key_ind_df.sample(5)
# Pivot
findex_pivot_df = findex_key_ind_df.pivot(index='Country Name', columns='Indicator Name', values='MRV').reset_index().rename_axis(None, axis=1)
findex_pivot_df.columns = ['country_name', 'account', 'formal_savings', 'formal_borrowing']

findex_pivot_df.sample(5)
findex_pivot_df['findex'] = gmean(findex_pivot_df.iloc[:,1:4],axis=1)
findex_pivot_df.head()
colorscale = [[0.0, 'rgb(230, 240, 255)'], [0.2, 'rgb(179, 209, 255)'], [0.4, 'rgb(102, 163, 255)'],\
              [0.6, 'rgb(26, 117, 255)'], [0.8, 'rgb(0, 71, 179)'], [1.0, 'rgb(0, 31, 77)']]
data = [dict(
        type='choropleth',
        locations= findex_pivot_df.country_name,
        locationmode='country names',
        z=findex_pivot_df['findex'],
        text=findex_pivot_df.country_name,
        colorscale = colorscale,
        colorbar=dict(autotick=False, tickprefix='', title='Findex'),
)]
layout = dict(
            title = 'Findex (Kiva Countries)',
            geo = dict(
            showframe = False, 
            showcoastlines = True, 
            projection = dict(type = 'Mercator')),
            margin = dict(t=30, b=30, l=10, r=10))
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False)
# Join
mpi_findex_national_df = mpi_national_df.merge(findex_pivot_df[['country_name', 'findex']], left_on=['country'], right_on=['country_name'])
mpi_findex_national_df.drop('country_name', axis=1, inplace=True)

mpi_findex_national_df.sample()
# Compare scores 
print("Correlation, p-value: ", pearsonr(mpi_findex_national_df.loc[:, 'findex'], mpi_findex_national_df.loc[:, 'MPI']))
sns.regplot(x='MPI', y='findex', data=mpi_findex_national_df)
plt.subplot(121).set_title("MPI distribuion")
sns.distplot(mpi_findex_national_df.MPI)

plt.subplot(122).set_title("Findex distribuion")
sns.distplot(mpi_findex_national_df.findex)

plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=2.0, rect=[0, 0, 0.95, 0.95])
# Most countries have data for 2016 but not all. Create new MRV column for most recent values. There are still a few nulls after this.
cellular_subscription_df['MRV'] = cellular_subscription_df['2016'].fillna(cellular_subscription_df['2015'])

# Keep only relevant columns
cellular_subscription_df = cellular_subscription_df[['Country Name', 'Country Code','MRV']]

cellular_subscription_df.sample(5)
colorscale = [[0.0, 'rgb(230, 240, 255)'], [0.2, 'rgb(179, 209, 255)'], [0.4, 'rgb(102, 163, 255)'],\
              [0.6, 'rgb(26, 117, 255)'], [0.8, 'rgb(0, 71, 179)'], [1.0, 'rgb(0, 31, 77)']]
data = [dict(
        type='choropleth',
        locations= cellular_subscription_df['Country Name'],
        locationmode='country names',
        z=cellular_subscription_df['MRV'],
        text=cellular_subscription_df['Country Name'],
        colorscale = colorscale,
        colorbar=dict(autotick=False, tickprefix='', title='Cellular Subscription'),
)]
layout = dict(
            title = 'Cellular Subscription Levels',
            geo = dict(
            showframe = False, 
            showcoastlines = True, 
            projection = dict(type = 'Mercator')),
            margin = dict(t=30, b=30, l=10, r=10))
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False)
# Join
mpi_cell_national_df = mpi_national_df.merge(cellular_subscription_df[['Country Name', 'MRV']], left_on=['country'], right_on=['Country Name'])
mpi_cell_national_df.drop('Country Name', axis=1, inplace=True)
# Compare scores 
print("Correlation, p-value: ", pearsonr(mpi_cell_national_df.loc[:, 'MRV'], mpi_cell_national_df.loc[:, 'MPI']))
sns.regplot(x='MPI', y='MRV', data=mpi_cell_national_df)
plt.subplot(121).set_title("MPI distribuion")
sns.distplot(mpi_cell_national_df.MPI)

plt.subplot(122).set_title("Telcomm Access distribuion")
sns.distplot(mpi_cell_national_df.MRV)

plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=2.0, rect=[0, 0, 0.95, 0.95])
gender_development_df[gender_development_df['2015'].isnull()]
gender_inequality_df[gender_inequality_df['2015'].isnull()]
# There are not many relevant missing values so lets just drop these for now. 
gender_development_df = gender_development_df.dropna(subset=['2015'])
gender_inequality_df = gender_inequality_df.dropna(subset=['2015'])
# Keep only relevant columns.
gender_development_df = gender_development_df[['Country', 'HDI Rank (2015)', '2015']]
gender_inequality_df = gender_inequality_df[['Country', 'HDI Rank (2015)', '2015']]
colorscale = [[0.0, 'rgb(255, 255, 255)'], [0.2, 'rgb(234, 250, 234)'], [0.4, 'rgb(173, 235, 173)'],\
              [0.6, 'rgb(91, 215, 91)'], [0.8, 'rgb(45, 185, 45)'], [1.0, 'rgb(31, 122, 31)']]
data = [dict(
        type='choropleth',
        locations= gender_development_df.Country,
        locationmode='country names',
        z=gender_development_df['2015'],
        text=gender_development_df.Country,
        colorscale = colorscale,
        colorbar=dict(autotick=False, tickprefix='', title='GDI'),
)]
layout = dict(
            title = 'Gender Development',
            geo = dict(
            showframe = False, 
            showcoastlines = True, 
            projection = dict(type = 'Mercator')),
            margin = dict(t=30, b=30, l=10, r=10))
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False)
data = [dict(
        type='choropleth',
        locations= gender_inequality_df.Country,
        locationmode='country names',
        z=gender_inequality_df['2015'],
        text=gender_inequality_df.Country,
        colorscale = [[0,'rgb(128, 0, 0)'],[1,'rgb(217, 179, 140)']],
        reversescale=True,
        colorbar=dict(autotick=False, tickprefix='', title='MPI'),
)]
layout = dict(
            title = 'Gender Inequality',
            geo = dict(
            showframe = False, 
            showcoastlines = True, 
            projection = dict(type = 'Mercator')),
            margin = dict(t=30, b=30, l=10, r=10))
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False)