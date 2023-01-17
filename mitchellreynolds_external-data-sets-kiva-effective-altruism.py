import os
import warnings
warnings.filterwarnings('ignore')

# Data Munging
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999
from IPython.display import HTML

# Data Visualizations
import matplotlib.pyplot as plt 
plt.style.use('fivethirtyeight')
%matplotlib inline
import seaborn as sns
import squarify
# Plotly has such beautiful graphs
import plotly
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.tools as tls
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as fig_fact
plotly.tools.set_config_file(world_readable=True, sharing='public')

# Beluga's Idea
competition_data_dir = '../input/data-science-for-good-kiva-crowdfunding/'
additional_data_dir = '../input/additional-kiva-snapshot/'

# Create DataFrames of the 4 given datasets
kiva_loans = pd.read_csv(competition_data_dir + 'kiva_loans.csv')
theme_region = pd.read_csv(competition_data_dir + 'loan_themes_by_region.csv')
theme_ids = pd.read_csv(competition_data_dir + 'loan_theme_ids.csv')
# What year is this MPI? 2017?
kiva_mpi = pd.read_csv(competition_data_dir + 'kiva_mpi_region_locations.csv')
# Addtional Snapshot Data - Beluga
all_kiva_loans = pd.read_csv(additional_data_dir + 'loans.csv')


kiva_loans_df = kiva_loans.copy()
kiva_loans_df['loan_amount_trunc'] = kiva_loans_df['loan_amount'].copy()
ulimit = np.percentile(kiva_loans_df.loan_amount.values, 99)
llimit = np.percentile(kiva_loans_df.loan_amount.values, 1)
kiva_loans_df['loan_amount_trunc'].loc[kiva_loans_df['loan_amount']>ulimit] = ulimit
kiva_loans_df['loan_amount_trunc'].loc[kiva_loans_df['loan_amount']<llimit] = llimit

kiva_loans_df['funding_amount_trunc'] = kiva_loans_df['funded_amount'].copy()
upper_limit = np.percentile(kiva_loans_df.funded_amount.values, 99)
lower_limit = np.percentile(kiva_loans_df.funded_amount.values, 1)
kiva_loans_df['funding_amount_trunc'].loc[kiva_loans_df['funded_amount']>upper_limit] = upper_limit
kiva_loans_df['funding_amount_trunc'].loc[kiva_loans_df['funded_amount']<lower_limit] = lower_limit

# Joining my dataset with Kiva's subregional dataset
# mpi_time['LocationName'] = mpi_time['region'] + ', ' + mpi_time['country']
# ez_mpi_join = pd.merge(kiva_mpi, mpi_time, on='LocationName')
# del ez_mpi_join['Unnamed: 0']
# del ez_mpi_join['country_y']
# del ez_mpi_join['region_y']
# ez_mpi_join = ez_mpi_join.rename(columns={'country_x': 'country', 'region_x': 'region'})
all_kiva_loans.head(1)
# Multidimensional Poverty Index (MPI) - The Dataset I uploaded
mpi_time = pd.read_csv('../input/multidimensional-poverty-measures/subnational_mpi_across_time.csv')
# Multidimensional Poverty Index (MPI)
national_mpi = pd.read_csv('../input/mpi/MPI_national.csv')
subnational_mpi = pd.read_csv('../input/mpi/MPI_subnational.csv')
# Google API Location Data
google_locations = pd.read_csv('../input/kiva-challenge-coordinates/kiva_locations.csv', sep='\t')
kiva_dates = pd.to_datetime(kiva_loans_df['disbursed_time'])
print("From the partial dataset from Kiva:")
print("The first loan was disbursed on ", kiva_dates.min())
print("The last loan was disbursed on ", kiva_dates.max())

snapshot_dates = pd.to_datetime(all_kiva_loans['disburse_time'])
print('\n')
print("From the additional dataset (Beluga upload):")
print("The first loan was disbursed on ", snapshot_dates.min())
print("The last loan was disbursed on ", snapshot_dates.max())
con_df = pd.DataFrame(all_kiva_loans['country_name'].value_counts()).reset_index()
con_df.columns = ['country', 'num_loans']
con_df = con_df.reset_index().drop('index', axis=1)

#Find out more at https://plot.ly/python/choropleth-maps/
data = [dict(
        type = 'choropleth',
        locations = con_df['country'],
        locationmode = 'country names',
        z = con_df['num_loans'],
        text = con_df['country'],
        colorscale = [[0,"rgb(5, 10, 172)"],[0.85,"rgb(40, 60, 190)"],[0.9,"rgb(70, 100, 245)"],\
            [0.94,"rgb(90, 120, 245)"],[0.97,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            tickprefix = '',
            title = 'Number of Loans'),
      ) ]

layout = dict(
    title = 'Number of Loans by Country',
    geo = dict(
        showframe = False,
        showcoastlines = True,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False, filename='loans-world-map')
plt.figure(figsize=(15,8))
count = all_kiva_loans['country_name'].value_counts()
squarify.plot(sizes=count.values, label=count.index, value=count.values)
plt.title('Total Distribution per Country')
plt.figure(figsize=(15,8))
count = all_kiva_loans['country_name'].value_counts().head(48)
squarify.plot(sizes=count.values, label=count.index, value=count.values)
plt.title('Distribution per Country - Top Half')
plt.figure(figsize=(15,8))
count = all_kiva_loans['country_name'].value_counts().tail(48)
squarify.plot(sizes=count.values, label=count.index, value=count.values)
plt.title('Distribution per Country - Bottom Half')



# Credit to SRK
plt.figure(figsize=(12,8))
sns.distplot(kiva_loans_df.funding_amount_trunc.values, bins=50, kde=False)
plt.xlabel('Funding Amount - USD', fontsize=12)
plt.title("Funding Amount Histogram after Outlier Truncation")
plt.show()
# Credit goes to Niyamat Ullah
# https://www.kaggle.com/niyamatalmass/who-takes-the-loan
plot_df_sector_popular_loan = pd.DataFrame(kiva_loans_df.groupby(['sector'])['funding_amount_trunc'].mean().sort_values(ascending=False)[:20]).reset_index()
plt.subplots(figsize=(15,7))
sns.barplot(x='sector',
            y='funding_amount_trunc',
            data=plot_df_sector_popular_loan,
            palette='RdYlGn_r',
            edgecolor=sns.color_palette('dark',7))
plt.ylabel('Average Funding Amount - USD', fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.xlabel('Loan sector', fontsize=20)
plt.title('Popular Sectors by Average Funding Amount (omitting outliers)', fontsize=24)
plt.savefig('popular_sectors.png')
plt.show()
trace = []
for name, group in kiva_loans_df.groupby("sector"):
    trace.append ( 
        go.Box(
            x=group["funding_amount_trunc"].values,
            name=name
        )
    )
layout = go.Layout(
    title='Loan Amount Distribution by Sector',
    width = 800,
    height = 800
)
#data = [trace0, trace1]
fig = go.Figure(data=trace, layout=layout)
py.iplot(fig, filename="LoanAmountSector")
# Clean up the all_kiva_loans dataframe
clean_df = all_kiva_loans.dropna(subset = ['disburse_time'])
clean_df['disburse_time'] = pd.to_datetime(clean_df['disburse_time'])
clean_df['cleaned_disburse_time'] = pd.DatetimeIndex(clean_df['disburse_time']).normalize()
clean_df['year'] = clean_df['cleaned_disburse_time'].dt.year

# Clean, merge and create new dataframe for country level MPI analysis over time
df1 = mpi_time.groupby(['country', 'year1']).agg({'total_population_year1': 'sum',
                                      'nb_poor_year1': 'sum',
                                      'poverty_intensity_year1': 'mean'}).reset_index()
df2 = mpi_time.groupby(['country', 'year2']).agg({'total_population_year2': 'sum',
                                      'nb_poor_year2': 'sum',
                                      'poverty_intensity_year2': 'mean'}).reset_index()
country_mpi_time = df1.merge(df2, left_on='country', right_on='country')
country_mpi_time =country_mpi_time[country_mpi_time['year1'] != country_mpi_time['year2']].reset_index()
del country_mpi_time['index']
country_mpi_time['country_mpi_year1'] = (country_mpi_time['nb_poor_year1'] / country_mpi_time['total_population_year1']) * (country_mpi_time['poverty_intensity_year1'] / 100.0)
country_mpi_time['country_mpi_year2'] = (country_mpi_time['nb_poor_year2'] / country_mpi_time['total_population_year2']) * (country_mpi_time['poverty_intensity_year2'] / 100.0)

# Find the unique set of ['country', 'year'] combinations
year_combo1 = country_mpi_time[['country', 'year1']].rename(columns={'year1': 'year'}).drop_duplicates()
year_combo2 = country_mpi_time[['country', 'year2']].rename(columns={'year2': 'year'}).drop_duplicates()
year_combo = year_combo1.append(year_combo2).drop_duplicates()

# Append country_sums to year_combos
list_of_ctry_sums = list()
for i, r in year_combo.iterrows():
        country_here, year_here = r['country'], r['year']
        yr_ctry_sum = clean_df[(clean_df['country_name'] == country_here) & (clean_df['year'] <= year_here)].funded_amount.sum()
        list_of_ctry_sums.append(yr_ctry_sum)
year_combo['country_sum'] = list_of_ctry_sums

new_df1 = country_mpi_time.merge(year_combo, left_on=['country', 'year1'], right_on=['country', 'year']).rename(columns={'country_sum': 'country_kiva_funded_sum_year1'})
new_df2 = country_mpi_time.merge(year_combo, left_on=['country', 'year2'], right_on=['country', 'year']).rename(columns={'country_sum': 'country_kiva_funded_sum_year2'})

temp_df1 = new_df1[['country', 'year1', 'year2', 'country_kiva_funded_sum_year1']]
temp_df2 = new_df2[['country', 'year1', 'year2', 'country_kiva_funded_sum_year2']]
new_df = temp_df1.merge(temp_df2, left_on=['country', 'year1', 'year2'], right_on=['country', 'year1', 'year2'])

df = country_mpi_time.merge(new_df, left_on=['country', 'year1', 'year2'], right_on=['country', 'year1', 'year2'])
df = df.drop_duplicates(subset=['country', 'year1'], keep='last')
df = df.drop_duplicates(subset=['country', 'year2'], keep='first')

df['mpi_diff'] = df['country_mpi_year2'] - df['country_mpi_year1']
df['kiva_diff'] = df['country_kiva_funded_sum_year2'] - df['country_kiva_funded_sum_year1']
df['log_kiva_diff'] = np.log1p(df['kiva_diff'].values)
df['year_diff'] = df['year2'] - df['year1']
df['annual_mpi_change'] = df['mpi_diff'] / df['year_diff']
df['annual_kiva_change'] = df['kiva_diff'] / df['year_diff']
df['log_annual_kiva_change'] = df['log_kiva_diff'] / df['year_diff']
df.sort_values(by='annual_mpi_change').head()
def correlation_heat_map(df):
    corrs = df.corr()
    
    # set figure size
    fig, ax = plt.subplots(figsize = (18, 13))
    
    #generate a mask for the upper triangle
    mask = np.zeros_like(corrs, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    
    #plot heatmap
    ax = sns.heatmap(corrs, mask=mask, annot=True)
    
    #resize labels
    ax.set_xticklabels(ax.xaxis.get_ticklabels(), fontsize=14, rotation=90)
    ax.set_yticklabels(ax.yaxis.get_ticklabels(), fontsize=14, rotation=0)
    
    plt.show()
    
correlation_heat_map(df=df[['total_population_year1', 'total_population_year2', 'nb_poor_year1', 'nb_poor_year2',
                            'poverty_intensity_year1', 'poverty_intensity_year2', 'country_mpi_year1', 'country_mpi_year2',
                            'country_kiva_funded_sum_year1', 'country_kiva_funded_sum_year1', 'mpi_diff', 'kiva_diff',
                            'annual_mpi_change', 'annual_kiva_change']])
sns.jointplot(data=df[df['log_annual_kiva_change'] > 0.0][['log_annual_kiva_change', 'annual_mpi_change']], x='log_annual_kiva_change', y='annual_mpi_change', kind="kde")
no_kiva = df[(df['kiva_diff'] == 0.0)].annual_mpi_change.mean()
some_kiva = df[(df['kiva_diff'] >= 0.0)].annual_mpi_change.mean()
delta = float(some_kiva) - float(no_kiva)
rel_delta = delta / no_kiva
print("Annualized MPI Change Average")
print("No Kiva Funding Ever: ", round(no_kiva, 6))
print("Has Had Kiva Funding: ", round(some_kiva, 6))
print("Absolute Difference: ", round(delta, 4))
print("Relative Difference", str(round(rel_delta, 2)) + "%")
loan_coords = pd.read_csv(additional_data_dir + 'loan_coords.csv')
loans = pd.read_csv(additional_data_dir + 'loans.csv')
loans_with_coords = loans[['loan_id', 'country_name', 'town_name']].merge(loan_coords, how='left', on='loan_id')

kiva_loans = kiva_loans.set_index("id")
themes = pd.read_csv(competition_data_dir + "loan_theme_ids.csv").set_index("id")
keys = ['Loan Theme ID', 'country', 'region']
locations = pd.read_csv(competition_data_dir + "loan_themes_by_region.csv",
                        encoding = "ISO-8859-1").set_index(keys)
loc_df  = kiva_loans.join(themes['Loan Theme ID'], how='left').join(locations, on=keys, rsuffix = "_")
matched_pct = 100 * loc_df['geo'].count() / loc_df.shape[0]
print("{:.1f}% of loans in kiva_loans.csv were successfully merged with loan_themes_by_region.csv".format(matched_pct))
print("We have {} loans in kiva_loans.csv with coordinates.".format(loc_df['geo'].count()))