# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns

from mpl_toolkits.basemap import Basemap


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
kiva_loans = pd.read_csv('../input/kiva_loans.csv')
loan_themes_by_region = pd.read_csv('../input/loan_themes_by_region.csv')
kiva_mpi_region_locations = pd.read_csv('../input/kiva_mpi_region_locations.csv')
loan_theme_ids = pd.read_csv('../input/loan_theme_ids.csv')
print(kiva_loans.shape)
print(loan_themes_by_region.shape)
print(kiva_mpi_region_locations.shape)
print(loan_theme_ids.shape)
kiva_loans.head()
loan_themes_by_region.head()
kiva_mpi_region_locations.head()
loan_theme_ids.head()
kiva_loans.head()
kiva_loans.describe()
sector_loan_counts = kiva_loans['sector'].value_counts()
fig = sns.barplot(sector_loan_counts.values, sector_loan_counts.index)
fig.set(xlabel= 'Number of Loans', ylabel = 'Sector')
sector_loan_total = kiva_loans.groupby('sector')['loan_amount'].sum()
sector_loan_total = sector_loan_total.sort_values(ascending=False)
fig = sns.barplot(sector_loan_total.values, sector_loan_total.index)
fig.set(xlabel= 'Total loan amount', ylabel = 'Sector')
country_loan_counts = kiva_loans['country'].value_counts()
country_loan_counts_top = country_loan_counts.sort_values(ascending=False)[:10]
fig = sns.barplot(country_loan_counts_top.values, country_loan_counts_top.index)
fig.set(xlabel= 'Number of Loans', ylabel = 'Country')
country_loan_total = kiva_loans.groupby('country')['loan_amount'].sum()
country_loan_total_top = country_loan_total.sort_values(ascending=False)[:10]
fig = sns.barplot(country_loan_total_top.values, country_loan_total_top.index)
fig.set(xlabel= 'Total loan amount', ylabel = 'Country')
country_loan_counts = kiva_loans['term_in_months']
plt.hist(country_loan_counts, bins=20)
country_loan_counts = kiva_loans['lender_count']
plt.hist(country_loan_counts, bins=20)
loan_themes_by_region.head()
fig = plt.figure(figsize=(10,10))

# draw the map background
m = Basemap(projection='cyl', resolution=None, llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180)
m.shadedrelief()

# scatter loan data, with size refecting area
lon = loan_themes_by_region['lon'].values
lat = loan_themes_by_region['lat'].values
amount = loan_themes_by_region['amount'].values

m.scatter(lon, lat, latlon=True, c='red', s=amount/10000, alpha=0.5)
partner_loan_counts = loan_themes_by_region['Field Partner Name'].value_counts()
partner_loan_counts_top = partner_loan_counts.sort_values(ascending=False)[:10]
fig = sns.barplot(partner_loan_counts_top.values, partner_loan_counts_top.index)
fig.set(xlabel= "Number Of loans", ylabel = 'Partner')
partner_loan_total = loan_themes_by_region.groupby('Field Partner Name')['amount'].sum()
partner_loan_total_top = partner_loan_total.sort_values(ascending=False)[:10]
fig = sns.barplot(partner_loan_total_top.values, partner_loan_total_top.index)
fig.set(xlabel= 'Total loan amount', ylabel = 'Partner')
kiva_mpi_region_locations.head()
fig = plt.figure(figsize=(10,10))

# draw the map background
m = Basemap(projection='cyl', resolution=None, llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180)
m.shadedrelief()

# scatter loan data, with size refecting area
lon = kiva_mpi_region_locations['lon'].values
lat = kiva_mpi_region_locations['lat'].values
MPI = kiva_mpi_region_locations['MPI'].values

m.scatter(lon, lat, latlon=True, c='red', s=MPI*100, alpha=0.5)
mpi_region_mean = kiva_mpi_region_locations.groupby('world_region')['MPI'].mean()
mpi_region_mean = mpi_region_mean.sort_values(ascending=False)
fig = sns.barplot(mpi_region_mean.values, mpi_region_mean.index)
fig.set(xlabel= 'MPI', ylabel = 'Region')
loan_theme_ids.head()
kiva_loans_with_theme = pd.merge(kiva_loans, loan_theme_ids, on='id')
kiva_loans_with_theme_counts = kiva_loans_with_theme['Loan Theme Type'].value_counts()
kiva_loans_with_theme_counts_top = kiva_loans_with_theme_counts.sort_values(ascending=False)[:10]
fig = sns.barplot(kiva_loans_with_theme_counts_top.values, kiva_loans_with_theme_counts_top.index)
fig.set(xlabel= 'Number of loans', ylabel = 'Partner')
kiva_loans_with_theme_total = kiva_loans_with_theme.groupby('Loan Theme Type')['loan_amount'].sum()
kiva_loans_with_theme_total_top = kiva_loans_with_theme_total.sort_values(ascending=False)[:10]
fig = sns.barplot(kiva_loans_with_theme_total_top.values, kiva_loans_with_theme_total_top.index)
fig.set(xlabel= 'Total loan amount', ylabel = 'Loan Theme')









