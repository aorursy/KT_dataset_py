import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import collections
import seaborn as sns

from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import figure

from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
init_notebook_mode(connected=True)

import os
print(os.listdir("../input"))
df_loans = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding/kiva_loans.csv')
df_mpi = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding/kiva_mpi_region_locations.csv')
mpi_national = pd.read_csv("../input/mpi/MPI_national.csv")
world_countries = pd.read_csv("../input/countries-of-the-world/countries of the world.csv")
country_stats = pd.read_csv("../input/additional-kiva-snapshot/country_stats.csv")
df_loans.head()
df_loans.shape
df_loans.describe()
df_loans.isna().sum().sort_values(ascending=False)
df_mpi.head()
genders = np.array(df_loans['borrower_genders'])

genders_updated = []

for gender in genders:
    if(type(gender)==str):
        gender = gender.replace(',', '').replace("'", '').replace("[", '').replace("]", '').split(' ')
        for x in gender:
            genders_updated.append(x)

borrower_genders = collections.Counter(genders_updated)

figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
the_grid = GridSpec(3, 1)

plt.figure(figsize=(7,7))
plt.pie(borrower_genders.values(), labels=borrower_genders.keys(), autopct='%1.1f%%',)
plt.title('Borrower genders', fontsize=16)


plt.show()
fig = plt.figure(figsize=(14,8))
plt.xticks(np.arange(0, max(df_loans['loan_amount'])+1, 5000), rotation = 45)
g = sns.distplot(df_loans['loan_amount'], norm_hist=False)
g.plot()
df_loans[df_loans['loan_amount'] > 50000]
df_loans['loan_amount'].sum()
sectors = df_loans['sector'].value_counts().reset_index()
sectors = sectors.rename(columns={'index': 'sector', 'sector': 'Loan Count'})

sectors_funded = df_loans.groupby('sector').sum()['loan_amount'].reset_index().sort_values('loan_amount', ascending=False)
sns.set()
fig = plt.figure(figsize=(15,8)) # Create matplotlib figure

ax = fig.add_subplot(111) # Create matplotlib axes
ax2 = ax.twinx() # Create another axes that shares the same x-axis as a
width = .4

sectors_funded.plot.bar(x='sector', y='loan_amount',color='plum',ax=ax,width=width, position=0)
sectors.plot.bar(x='sector', y='Loan Count', color='mediumaquamarine', ax=ax2,width = width,position=1)

ax.grid(None, axis=1)
ax2.grid(None)

ax.set_ylabel('Total Loan Amount')
ax2.set_ylabel('Number of Loans')

ax.set_title('Loan Amount and Loan Count by Sector',fontsize=16)
plt.figure(figsize=(10,6))
ax = sns.boxplot(x='loan_amount',y='sector', data=df_loans, showfliers=False)
plt.title('Loan amount by sector', fontsize=16)
df = df_loans[df_loans['country'].isin(['Philippines', 'Kenya', 'El Salvador', 'Cambodia', 'Pakistan', 'Peru'])]
g = sns.FacetGrid(df, col="country", col_wrap=2)
g.set_xticklabels(rotation=90)
g.map(sns.countplot, 'sector')
plt.figure(figsize=(12,8))
activities = df_loans['activity'].value_counts().head(40)
sns.barplot(y=activities.index, x=activities.values, alpha=0.6)
plt.title("Loan Count by Activity (top 40)", fontsize=16)
plt.xlabel("Loan Count", fontsize=15)
plt.ylabel("Activity", fontsize=15)
countries = df_loans['country'].value_counts().reset_index()
countries = countries.rename(columns={'index': 'country', 'country': 'Loan Count'})

countries_total_funded = df_loans.groupby('country').sum()['loan_amount'].reset_index().sort_values('loan_amount', ascending=False)
countries_total_funded = countries_total_funded.rename(columns={'loan_amount':'Total Amount Loaned'})
countries_merged = countries.merge(countries_total_funded, on='country')
sns.set()
fig = plt.figure(figsize=(15,8)) # Create matplotlib figure

ax = fig.add_subplot(111) # Create matplotlib axes
ax2 = ax.twinx() # Create another axes that shares the same x-axis as a
width = .4

countries_merged.head(30).plot.bar(x='country', y='Total Amount Loaned',color='plum',ax=ax,width=width, position=0)
countries_merged.head(30).plot.bar(x='country', y='Loan Count', color='mediumaquamarine', ax=ax2,width = width,position=1)

ax.grid(None, axis=1)
ax2.grid(None)

ax.set_ylabel('Total Loan Amount')
ax2.set_ylabel('Number of Loans')

ax.set_title('Loan Amount and Loan Count by Country (top 30)',fontsize=16)
top_countries = countries_total_funded.head(20)['country']

loan_range=df_loans[(df_loans['country'].isin(top_countries))]
plt.figure(figsize=(12,8))
ax = sns.boxplot(x='loan_amount',y='country', data=loan_range, showfliers=False)
plt.title('Loan amount by country', fontsize=16)
df_loans['region_country'] = df_loans['region'] + ', ' + df_loans['country']

plt.figure(figsize=(12,8))
regions = df_loans['region_country'].value_counts().head(30)
sns.barplot(y=regions.index, x=regions.values, alpha=0.6)
plt.title("Loan Count by Region (top 30)", fontsize=16)
plt.xlabel("Loan Count", fontsize=15)
plt.ylabel("Region", fontsize=15)
country_avg_loan = df_loans.groupby('country')['loan_amount'].mean().reset_index()
country_avg_loan = country_avg_loan.rename(columns={'loan_amount': 'avg_loan_amount'}).sort_values('avg_loan_amount', ascending=False)
country_avg_loan.head()
df_loans.groupby('country').agg({'loan_amount': 'mean'}).sort_values('loan_amount', ascending=False).tail(10)
top_countries_df = df_loans[df_loans['country'].isin(top_countries)]
top_countries_df.groupby('country').agg({'loan_amount': 'mean'}).sort_values('loan_amount', ascending=False)
plt.figure(figsize=(12,8))
plt.title('Term in Months vs. Loan Amount', fontsize=16)
sns.scatterplot(x=df_loans['loan_amount'], y=df_loans['term_in_months'])

# df_loans[df_loans['loan_amount'] < 4000].sample(100).plot.scatter(x='loan_amount', y='term_in_months', title='Term in Months vs Loan Amount')
region_themes = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/loan_themes_by_region.csv")
region_themes[region_themes['region'] == 'Kaduna']

region_themes['number'].sum()
countries_mpi = countries_total_funded.merge(mpi_national, left_on='country', right_on='Country')
countries_mpi = countries_mpi.sort_values('MPI Urban', ascending=False)
countries_mpi = countries_mpi.drop(['Country'], axis=1)
data = [ dict(
        type = 'choropleth',
        locationmode='country names',
        locations=countries_mpi['country'],
        z=countries_mpi['MPI Rural']
      ) ]

layout = dict(
    title = 'Rural MPI'
)

fig = dict( data=data, layout=layout )
iplot( fig )
data = [ dict(
        type = 'choropleth',
        locationmode='country names',
        locations=countries_mpi['country'],
        z=countries_mpi['MPI Urban']
      ) ]

layout = dict(
    title = 'Urban MPI'
)

fig = dict( data=data, layout=layout )
iplot( fig )
data = [ dict(
        type = 'scattergeo',
        locationmode = 'ISO-3',
        lon = df_mpi['lon'],
        lat = df_mpi['lat'],
        text = df_mpi['region'] + ', ' + df_mpi['country'] + ': ' + df_mpi['MPI'].astype(str),
        mode = 'markers',
        marker = dict(
            size = 7,
            opacity = 0.8,
            color = df_mpi['MPI'],
            symbol = 'square',
            cmin = 0,
            colorbar=dict(
                title="MPI"
            )
        ))]

layout = dict(
        title = 'MPI by Region',
        colorbar = True,
        geo = dict(
            scope='world',
            #projection=dict( type='albers usa' ),
            showland = True,
            landcolor = "rgb(250, 250, 250)",
            subunitcolor = "rgb(217, 217, 217)",
            countrycolor = "rgb(217, 217, 217)",
            countrywidth = 0.5,
            subunitwidth = 0.5
        ),
    )
fig = dict( data=data, layout=layout )


iplot( fig, validate=False, filename='d3-loan-map' )
sns.scatterplot(x=countries_mpi['MPI Rural'], y=countries_mpi['Total Amount Loaned'])
# country_stats.head()
country_stats = country_stats.rename(columns={'country_name': 'country'})
country_stats = country_stats.merge(countries_total_funded, on = 'country')
country_stats[country_stats['country'] == 'Afghanistan']
x = country_stats[country_stats['population'] < 1000000000]
sns.scatterplot(x=x['population'], y=x['Total Amount Loaned'])
country_stats['Loan Amount per Capita'] = country_stats['Total Amount Loaned'] / country_stats['population']
country_stats = country_stats.sort_values('Loan Amount per Capita', ascending = False)
plt.figure(figsize=(12,8))
sns.barplot(y=country_stats.head(20)['Loan Amount per Capita'], x=country_stats.head(20)['country'], alpha=0.6)
plt.title("Loan amount per Capita", fontsize=16)
plt.xlabel("Amount per Capita", fontsize=15)
plt.ylabel("Country", fontsize=15)
plt.xticks(rotation = 45)
country_stats['number_below_poverty_line'] = country_stats['population_below_poverty_line'] * 0.01 * country_stats['population']
country_stats
plt.figure(figsize=(12,8))
country_stats = country_stats.sort_values('number_below_poverty_line', ascending=False)
sns.barplot(y=country_stats.head(20)['number_below_poverty_line'], x=country_stats.head(20)['country'], alpha=0.6)
plt.title("Citizens Below Poverty Line", fontsize=16)
plt.xlabel("Country", fontsize=15)
plt.ylabel("Citizens Below Poverty Line", fontsize=15)
plt.xticks(rotation = 45)
sns.set()
fig = plt.figure(figsize=(15,8)) # Create matplotlib figure

ax = fig.add_subplot(111) # Create matplotlib axes
ax2 = ax.twinx() # Create another axes that shares the same x-axis as a
width = .4

country_stats.head(30).plot.bar(x='country', y='number_below_poverty_line',color='plum',ax=ax,width=width, position=0,alpha=0.7)
country_stats.head(30).plot.bar(x='country', y='Total Amount Loaned', color='mediumaquamarine', ax=ax2,width = width,position=1,alpha=0.7)

ax.grid(None, axis=1)
ax2.grid(None)

ax.set_ylabel('Total Loan Amount')
ax2.set_ylabel('Number of Loans')

ax.set_title('Loans per capita and people under poverty line',fontsize=16)
fig = plt.figure(figsize=(10,8))

sns.heatmap(df_loans.corr(), annot=True)